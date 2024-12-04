import numpy as np
import mpi4py.MPI as mpi
import traceback
from .io import Params
from . import math
from .space_solver import SpaceSolver
from .time_integrator import DICT_TIME_INTEGRATORS
from .io import HDF5Writer
from .utils import Timer
from . import logger


class Solver:
    """
    Numerical solver for the incompressible Navier-Stokes equations.

    The equation is solved in Fourier space using a Galerkin pseudo-spectral method.

    Parameters:
    ----------
    params: Params
        Simulation parameters. 
    """

    def __init__(self, params: Params):

        self.mpi_comm = mpi.COMM_WORLD
        self.mpi_rank = self.mpi_comm.Get_rank()
        self.mpi_size = self.mpi_comm.Get_size()

        self.params = None
        if self.mpi_rank == 0:
            self.params = params
        self.params = self.mpi_comm.bcast(self.params, root=0)

        np.random.seed(np.random.randint(0, 2**32 - 1))

        self.space_solver = SpaceSolver(self.mpi_comm, self.mpi_rank, params=self.params)

        self.step = 0
        self.t = 0.0

        self._init_integrator()
        self._init_writer()
        self._timer = Timer(self.mpi_comm, self.params.verbose)

    def __getattr__(self, name):
        if hasattr(self.space_solver, name):
            return getattr(self.space_solver, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _init_integrator(self):
        try:
            if self.params.time_integrator not in DICT_TIME_INTEGRATORS:
                raise ValueError(f"Time integrator '{self.params.time_integrator}' not found.")
            self.time_integrator = DICT_TIME_INTEGRATORS[self.params.time_integrator](
                self.params.dt,
                self.space_solver.linear_operator,
                self.params.optimization
            )

        except Exception as e:
            if self.mpi_rank == 0:
                logger.critical(f"Failed to initialize time integrator: {e}\n{traceback.format_exc()}")
            self.mpi_comm.Abort(1)

    def _init_writer(self):
        if self.params.file_name.endswith(".h5"):
            self.params.file_name.replace(".h5", "")

        solution_dict = {"space": self.space_solver.V}
        component_names = ['u', 'v', 'w'][:self.space_solver.dim]
        solution_dict["data"] = {
            name: [self.space_solver.u[i]] for i, name in enumerate(component_names)
        }

        restart_dict = {
            "space": self.space_solver.V,
            "data": {'u_hat': [self.space_solver.u_hat]}
        }

        self.data_writer = HDF5Writer(
            self.mpi_comm,
            self.mpi_rank,
            self.mpi_size,
            params=self.params,
            solution_data=solution_dict,
            restart_data=restart_dict,
            time_points=[self.t, self.params.end_time]
        )

    def initialize(
        self,
        u0: np.ndarray,
        space: str = 'physical',
        project: bool = True,
        mask_zero_mode: bool = True
    ):
        self.space_solver._initialize(u0, space, mask_zero_mode)

        if project:
            math.leray_projection(
                self.space_solver.u_hat,
                self.space_solver.k,
                self.space_solver.k_over_k2,
                self.space_solver.p_hat
            )
            self.space_solver.backward()

    def restart(self):
        data_path, t, step = self.data_writer.read_restart_file()
        self.data_writer.restart_file.open()
        self.data_writer.restart_file.f[data_path].read_direct(
            self.space_solver.u_hat,
            source_sel=self.space_solver.local_slice_fourier
        )
        self.data_writer.restart_file.close()
        self.space_solver.backward()
        self.t = t
        self.step = step
        self.data_writer.time_points[0] = t

        dt = self.data_writer.time_points[1] - self.data_writer.time_points[0]
        if dt < 1e-8:
            logger.critical("Restart time is equal to end time.")
            self.mpi_comm.Abort(1)

    def set_linear_operator(self, L: np.ndarray):
        self.space_solver.linear_operator[:] = L
        self.time_integrator.update_linear_operator(L)

    def set_noise_correlation(self, C: np.ndarray):
        self.space_solver.correlation_func[:] = C

    def set_filter_kernel(self, kernel: np.ndarray):
        self.space_solver.filter_kernel[:] = kernel

    def solve(self):
        self.mpi_comm.Barrier()
        self._timer.start()

        dt = self.time_integrator.dt
        u_hat = self.space_solver.u_hat
        rhs_nonlinear = self.space_solver.rhs_nonlinear
        rhs_linear = self.space_solver.rhs_linear
        noise = self.space_solver.noise
        forcing = self.space_solver.forcing

        n_stages = self.time_integrator.n_stages
        noise_factor = self.time_integrator.noise_factor

        u_tmp = self.space_solver.cached_array[(u_hat, 0, True)]

        if self.params.write_data:
            self.data_writer.write_data(self.step)

        while not np.isclose(self.t, self.params.end_time, atol=1e-9):
            self.t += dt
            self.step += 1

            u_tmp[:] = u_hat
            self.space_solver.random_fields()
            for stage in range(n_stages):
                self.space_solver.compute_rhs(noise_factor[stage])
                self.time_integrator.stepping(
                    u_hat, u_tmp, stage, rhs_nonlinear, rhs_linear, forcing, noise
                )

            if self.params.write_data:
                if self.step % self.params.write_interval == 0:
                    self.space_solver.backward()
                    self.data_writer.write_data(self.step)

            if self.params.write_restart:
                if self.step % self.params.restart_interval == 0:
                    self.data_writer.write_restart(self.t, self.step)

            if self.step % self.params.check_interval == 0:
                self._timer(self.t, self.step)

        self.data_writer.close()
        self._timer.final()
