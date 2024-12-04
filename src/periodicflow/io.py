from pathlib import Path
import numpy as np
import h5py
import mpi4py.MPI as mpi
import json
import traceback
from periodicflow import logger
from .utils import periodic_bc

try:
    import shenfun as sf
except ImportError:
    raise ImportError('shenfun is required for this module')


class Params:
    def __init__(self, json_file='input.json'):
        self._load_json(json_file)
        self._check_compatibility()

    def __repr__(self):
        return f"<Params {self.__dict__}>"

    def _load_json(self, json_file):
        """Load JSON file and set attributes."""
        try:
            with open(json_file, 'r') as file:
                data = json.load(file)
            self._read_params(data)
        except Exception as e:
            raise ValueError(f"Error reading JSON file {json_file}: {e}")

    def _read_params(self, data):
        defaults = {
            'N': [256, 256],
            'domain': [[0, 2 * np.pi], [0, 2 * np.pi]],
            'dt': 0.01,
            'end_time': 1.0,
            'enable_nonlinear': False,
            'time_integrator': 'rk3',
            'viscosity': 1e-2,
            'noise_type': 'thermal',
            'noise_mag': 0.0,
            'filter_noise': False,
            'optimization': True,
            'verbose': False,
            'check_interval': np.iinfo(np.int64).max,
            'write_data': False,
            'file_name': 'data',
            'write_interval': np.iinfo(np.int64).max,
            'write_mode': 'w',
            'enforce_periodic': True,
            'write_restart': False,
            'restart_name': 'restart',
            'restart_interval': np.iinfo(np.int64).max,
            'restart_mode': 'w',
            'dealias': '3/2',
            'mask_nyquist': True,
            'fft_plan': 'FFTW_MEASURE',
            'decomposition': 'slab',
        }

        for key, default in defaults.items():
            setattr(self, key, data.get(key, default))

    def _check_compatibility(self):
        """Check compatibility between related parameters."""
        if len(self.N) != len(self.domain):
            raise ValueError("The length of N must match the number of domain dimensions.")

    def print(self):
        """Print the parameters in a human-readable format."""
        import pprint
        pprint.pprint(self.__dict__)

    def update(self, params):
        """Update parameters from a dictionary."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")

    def docs(self):
        """Print documentation for all parameters."""
        docs = {
            "N": "Number of grid points in each dimension, e.g., [Nx, Ny].",
            "domain": "Domain size for each dimension, e.g., [[xmin, xmax], [ymin, ymax]].",
            "dt": "Time step for integration.",
            "end_time": "Simulation end time.",
            "enable_nonlinear": "Enable or disable nonlinear terms.",
            "viscosity": "Viscosity of the fluid.",
            "noise_type": "Type of noise to add to the system. Options: 'thermal', 'correlated'.",
            "noise_mag": "Magnitude of the noise.",
            "filter_noise": "Wether filter noise or not.",
            "optimization": "Enable or disable numba-jit optimization.",
            "verbose": "Enable or disable verbose output.",
            "check_interval": "Interval for checking the solution.",
            "write_data": "Write velocity data to file.",
            "file_name": "Name of the output file.",
            "write_interval": "Interval for writing data to file.",
            "write_mode": "Write mode for the output file.",
            "enforce_periodic": "Enforce periodic boundary conditions.",
            "write_restart": "Write restart data to file.",
            "restart_name": "Name of the restart file.",
            "restart_interval": "Interval for writing restart data to file.",
            "restart_mode": "Write mode for the restart file.",
            "dealias": "Dealiasing factor for the nonlinear terms.",
            "mask_nyquist": "Mask the Nyquist frequency.",
            "fft_plan": "FFT plan for the FFTW library.",
            "decomposition": "Decomposition strategy for parallelization."
        }
        for key, doc in docs.items():
            print(f"{key}: {doc}")


class HDF5Writer:
    def __init__(
        self,
        mpi_comm: mpi.Comm,
        mpi_rank: int,
        mpi_size: int,
        params: Params,
        solution_data: dict = {},
        restart_data: dict = {},
        time_points: list = [0.0, 1.0]
    ):
        self.mpi_comm = mpi_comm
        self.mpi_rank = mpi_rank
        self.mpi_size = mpi_size
        self.file_name = params.file_name
        self.restart_name = params.restart_name
        self.solution_data = solution_data
        self.restart_data = restart_data
        self.time_points = time_points
        self.end_points = [x_end for _, x_end in params.domain]
        self.periodic = params.enforce_periodic

        self.solution_file = sf.ShenfunFile(
            self.file_name,
            self.solution_data['space'],
            mode='w'
        )
        self.restart_file = sf.ShenfunFile(
            self.restart_name,
            self.restart_data['space'],
            mode='w'
        )

    def write_data(self, step: int):
        self.solution_file.write(step, self.solution_data['data'], as_scalar=True)

    def write_restart(self, step: int):
        self.restart_file.write(step, self.restart_data['data'])

    def close(self):
        if self.solution_file.f:
            self.solution_file.close()

        if self.restart_file.f:
            self.restart_file.close()

    @staticmethod
    def _distribute_steps(rank: int, size: int, steps: list):
        """
        Distribute steps among MPI processes using round-robin scheme.
        """
        return [steps[i] for i in range(len(steps)) if i % size == rank]

    def _get_dataset_info(self, input_file: Path):
        """
        Get information about the dataset structure.
        """

        mpi_comm = self.mpi_comm
        rank = self.mpi_rank
        size = self.mpi_size

        try:
            with h5py.File(input_file, 'r', driver='mpio', comm=mpi_comm) as f:
                variables = list(f)
                str_dim = next(iter(f[variables[0]]), None)
                ndims = {"3d": 3, "2d": 2}.get(str_dim.lower(), 0)

                if ndims == 0:
                    raise ValueError(f"[Rank {rank}] get_dataset_info: Dimension identifier '{str_dim}' not recognized.")

                steps = sorted(f[variables[0]][str_dim], key=int)
                local_steps = HDF5Writer._distribute_steps(rank, size, steps)
                n_digits = len(str(steps[-1]))

                shape = [
                    len(f[f"{variables[0]}/mesh/{key}"]) + (1 if self.periodic else 0)
                    for key in ["x0", "x1", "x2"][:ndims]
                ]

        except Exception as e:
            logger.error(f"[Rank {rank}] get_dataset_info: {e}\n{traceback.format_exc()}")
            mpi_comm.Abort(1)

        return {
            "variables": variables,
            "str_dim": str_dim,
            "ndims": ndims,
            "shape": shape,
            "steps": steps,
            "n_digits": n_digits,
            "local_steps": local_steps,
        }

    def _prepare_temp_file(
        self,
        input_file: Path,
        temp_file: Path,
        dataset_info: dict
    ):
        """
        Prepare temporary HDF5 file for reconfiguring dataset structure.
        """

        mpi_comm = self.mpi_comm
        rank = self.mpi_rank
        time_points = self.time_points
        end_points = self.end_points

        try:
            with h5py.File(input_file, 'r', driver='mpio', comm=mpi_comm) as fr, \
                    h5py.File(temp_file, 'w', driver='mpio', comm=mpi_comm) as fw:

                variables = dataset_info["variables"]
                ndims = dataset_info["ndims"]
                shape = dataset_info["shape"]
                steps = dataset_info["steps"]

                for key, label, x_end in zip(["x0", "x1", "x2"][:ndims], ["x", "y", "z"][:ndims], end_points[:ndims]):
                    coord = fr[f"{variables[0]}/mesh/{key}"][:]
                    if self.periodic:
                        coord = np.append(coord, x_end)
                    fw.create_dataset(label, data=coord)

                fw.create_dataset("t", data=np.linspace(time_points[0], time_points[-1], len(steps)))

                n_digits = dataset_info["n_digits"]
                for var in variables:
                    grp = fw.create_group(var)
                    for step in steps:
                        step_format = str(step).zfill(n_digits)
                        grp.create_dataset(step_format, shape=shape, dtype=np.float64)

        except Exception as e:
            logger.error(f"[Rank {rank}] prepare_temp_file: {e}\n{traceback.format_exc()}")
            mpi_comm.Abort(1)

        mpi_comm.Barrier()

    def reconfigure_dataset(self):
        """
        Reconfigure/simplify dataset structure dictated by ShenfunFile.
        """

        mpi_comm = self.mpi_comm
        rank = self.mpi_rank
        periodic = self.periodic

        if rank == 0:
            logger.info("HDF5Writer.reconfigure_dataset: Start reconfiguring dataset.")

        input_file = Path(self.file_name).with_suffix(".h5")
        temp_file = Path(str(input_file).replace(".h5", "_temp.h5"))

        try:
            if not input_file.exists():
                raise FileNotFoundError(f"Solution file '{input_file}' does not exist.")
        except FileNotFoundError as e:
            if rank == 0:
                logger.error(f"HDF5Writer.reconfigure_dataset: {e}")
            mpi_comm.Abort(1)

        dataset_info = self._get_dataset_info(input_file)
        self._prepare_temp_file(input_file, temp_file, dataset_info)

        try:
            with h5py.File(input_file, 'r', driver='mpio', comm=mpi_comm) as fr, \
                    h5py.File(temp_file, 'a', driver='mpio', comm=mpi_comm) as fw:

                variables = dataset_info["variables"]
                local_steps = dataset_info["local_steps"]
                n_digits = dataset_info["n_digits"]
                str_dim = dataset_info["str_dim"]

                for var in variables:
                    grp = fw[var]
                    for step in local_steps:
                        data = fr[f"{var}/{str_dim}/{step}"][:]
                        if periodic:
                            data = periodic_bc(data)
                        step_name = str(step).zfill(n_digits)
                        grp[step_name].write_direct(data)

            mpi_comm.Barrier()
            if rank == 0:
                input_file.unlink()
                temp_file.rename(input_file)

        except Exception as e:
            logger.error(f"[Rank {rank}] reconfigure_dataset: {e}\n{traceback.format_exc()}")
            mpi_comm.Abort(1)

        if rank == 0:
            logger.info("HDF5Writer.reconfigure_dataset: Dataset reconfigured successfully.")
