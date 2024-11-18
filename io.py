from pathlib import Path
import numpy as np
import h5py
import mpi4py.MPI as mpi
import traceback
from periodicflow import logger
from .utils import periodic_bc

try:
    import shenfun as sf
except ImportError:
    raise ImportError('shenfun is required for this module')


class HDF5Writer:
    def __init__(
        self,
        comm: mpi.Comm,
        rank: int,
        size: int,
        file_name: str,
        solution: dict = {},
        time_points: list = [0.0, 1.0],
        end_points: list = [2 * np.pi, 2 * np.pi],
        periodic: bool = False,
    ):
        self.comm = comm
        self.mpi_rank = rank
        self.mpi_size = size
        self.file_name = file_name
        self.solution = solution
        self.time_points = time_points
        self.end_points = end_points
        self.periodic = periodic

        self.solution_file = sf.ShenfunFile(
            self.file_name,
            self.solution['space'],
            mode='w'
        )

    def write(self, step: int):
        self.solution_file.write(step, self.solution['data'], as_scalar=True)

    def close(self):
        if self.solution_file.f:
            self.solution_file.close()

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

        comm = self.comm
        rank = self.mpi_rank
        size = self.mpi_size

        try:
            with h5py.File(input_file, 'r', driver='mpio', comm=comm) as f:
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
            raise RuntimeError(f"[Rank {rank}] get_dataset_info: {e}\n{traceback.format_exc()}")

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

        comm = self.comm
        rank = self.mpi_rank
        time_points = self.time_points
        end_points = self.end_points

        try:
            with h5py.File(input_file, 'r', driver='mpio', comm=comm) as fr, \
                    h5py.File(temp_file, 'w', driver='mpio', comm=comm) as fw:

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
            raise RuntimeError(f"[Rank {rank}] prepare_temp_file: {e}\n{traceback.format_exc()}")

        comm.Barrier()

    def reconfigure_dataset(self):
        """
        Reconfigure dataset structure determined by ShenfunFile.

        Parameters
        ----------
        comm : mpi4py.MPI.Intracomm
            MPI communicator.
        rank : int
            Rank of the current process.
        size : int
            Total number of processes.
        file_name : str
            Name of the input HDF5 file.
        time_points : list of float
            Initial and final time points.
        periodic : bool, optional   
            Whether to enforce periodic boundary conditions.
        x_end : float, optional
            The end value for periodicity, default is 2π.
        """

        comm = self.comm
        rank = self.mpi_rank
        periodic = self.periodic

        if rank == 0:
            logger.info("HDF5Writer.reconfigure_dataset: Start reconfiguring dataset ...")

        input_file = Path(self.file_name).with_suffix(".h5")
        temp_file = Path(str(input_file).replace(".h5", "_temp.h5"))
        if not input_file.exists():
            raise FileNotFoundError(f"Solution file '{input_file}' does not exist.")

        dataset_info = self._get_dataset_info(input_file)
        self._prepare_temp_file(input_file, temp_file, dataset_info)

        try:
            with h5py.File(input_file, 'r', driver='mpio', comm=comm) as fr, \
                    h5py.File(temp_file, 'a', driver='mpio', comm=comm) as fw:

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
                        grp[step_name][...] = data

            comm.Barrier()
            if rank == 0:
                input_file.unlink()
                temp_file.rename(input_file)

        except Exception as e:
            raise RuntimeError(f"[Rank {rank}]: {e}\n{traceback.format_exc()}")

        if rank == 0:
            logger.info("HDF5Writer.reconfigure_dataset: Dataset reconfigured successfully.")
