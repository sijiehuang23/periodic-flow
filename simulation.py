import numpy as np
import mpi4py.MPI as mpi
from . import math
from .space_solver import SpaceSolver
from .time_integrator import DICT_TIME_INTEGRATORS
from .io import HDF5Writer
from .utils import Timer


class Solver:
    """
    Numerical solver for the incompressible Navier-Stokes equations.

    The equation is solved in Fourier space using a Galerkin pseudo-spectral method.

    Parameters:
    ----------
    N (list or tuple):
        Number of grid points in each direction.
    dt (float):
        Time step size.
    end_time (float):
        End time of the simulation.
    file_name (str):
        Name of the output file.
    time_integrator (str, optional):
        Time integrator to use. Default is 'rk2'.
    viscosity (float, optional):
        Kinematic viscosity. Default is 1e-3.
    noise_type (str, optional):
        Type of noise to add. Options are 'thermal' and 'correlated'. Default is 'thermal'.
    noise_mag (float, optional):
        Magnitude of the noise. Default is 0.0 (so no noise).
    domain (float, list or tuple):
        Size of the domain.
    check_interval (int, optional):
        Interval at which to print simulation information. Default is 100.
    verbose (bool, optional):
        If True, prints simulation information. Default is False.
    write_interval (int, optional):
        Interval at which to write to file. Default is 100.
    dealias (str, optional):
        Dealising method to use. Options are '2/3' (truncation) and '3/2' (padding). Default is '3/2'.
    mask_nyquist (bool, optional):
        If True, set the Nyquist components to zeros. Default is False.
    fft_plan (str, optional):
        FFTW plan to use. Options are 'FFTW_ESTIMATE', 'FFTW_MEASURE', 'FFTW_PATIENT', 'FFTW_EXHAUSTIVE'. Default is 'FFTW_MEASURE'.
    decomposition (str, optional):
        Parallel decomposition to use. Options are 'slab' and 'pencil'. Default is 'slab'.
    optimization (bool, optional):
        If True, use numba-optimized time integrator. Default is False.
    """

    def __init__(
            self,
            N: list | tuple,
            dt: float,
            end_time: float,
            is_nonlinear: bool = True,
            time_integrator: str = "rk2",
            viscosity: float = 1e-3,
            noise_type: str = "thermal",
            noise_mag: float = 0.0,
            domain: float | list | tuple = 2 * np.pi,
            check_interval: int = 100,
            verbose: bool = False,
            write_solution: bool = False,
            file_name: str = None,
            file_mode: str = "w",
            write_interval: int = np.iinfo(np.int64).max,
            periodic: bool = False,
            dealias: str = "3/2",
            mask_nyquist: bool = False,
            fft_plan: str = "FFTW_MEASURE",
            decomposition: str = "slab",
            optimization: bool = False
    ):

        self.comm = mpi.COMM_WORLD
        self.mpi_rank = self.comm.Get_rank()
        self.mpi_size = self.comm.Get_size()

        self.space_solver = SpaceSolver(
            self.comm,
            self.mpi_rank,
            N,
            is_nonlinear=is_nonlinear,
            viscosity=viscosity,
            noise_type=noise_type,
            noise_mag=noise_mag,
            domain=domain,
            dealias=dealias,
            mask_nyquist=mask_nyquist,
            fft_plan=fft_plan,
            decomposition=decomposition
        )

        np.random.seed(np.random.randint(0, 2**32 - 1))

        self.end_time = end_time
        self.write_solution = write_solution
        self.write_interval = write_interval
        self.check_interval = check_interval
        self.verbose = verbose

        self.step = 0
        self.t = 0.0
        self.file_mode = file_mode

        self._init_integrator(time_integrator, dt, optimization)
        self._init_writer(file_name, periodic)
        self._timer = Timer(self.comm, verbose)

    def __getattr__(self, name):
        if hasattr(self.space_solver, name):
            return getattr(self.space_solver, name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _init_integrator(self, time_integrator, dt, optimization=False):
        try:
            if time_integrator not in DICT_TIME_INTEGRATORS:
                raise ValueError(f"Time integrator '{time_integrator}' not found.")
            self.time_integrator = DICT_TIME_INTEGRATORS[time_integrator](dt, self.space_solver.linear_operator, optimization)

        except Exception as e:
            raise RuntimeError(f"Failed to initialize time integrator: {e}")

    def _init_writer(self, file_name, periodic):
        if self.write_solution:
            if file_name.endswith(".h5"):
                file_name.replace(".h5", "")

            solution_dict = {"space": self.space_solver.V}
            component_names = ['u', 'v', 'w'][:self.space_solver.dim]
            solution_dict["data"] = {
                name: [self.space_solver.u[i]] for i, name in enumerate(component_names)
            }

            end_points = [x_end for _, x_end in self.space_solver.domain]
            self.solution_writer = HDF5Writer(
                self.comm,
                self.mpi_rank,
                self.mpi_size,
                file_name,
                solution=solution_dict,
                time_points=[0.0, self.end_time],
                end_points=end_points,
                periodic=periodic
            )

    @classmethod
    def read_config(cls, config: dict):
        """Read configurations from a dictionary."""
        return cls(**config)

    def initialize(self, u0: np.ndarray, space: str = 'physical', project: bool = False):
        self.space_solver.initialize_velocity(u0, space)

        if project:
            math.leray_projection(
                self.space_solver.u_hat,
                self.space_solver.k,
                self.space_solver.k_over_k2,
                self.space_solver.p_hat
            )
            self.space_solver.backward()

        if self.write_solution:
            self.solution_writer.write(self.step)

    def set_linear_operator(self, L: np.ndarray):
        self.space_solver.linear_operator[:] = L
        self.time_integrator.update_linear_operator(L)

    def set_correlation_function(self, C: np.ndarray):
        self.space_solver.correlation_func[:] = C

    def solve(self):
        self._timer.start()

        dt = self.time_integrator.dt
        u_hat = self.space_solver.u_hat
        rhs_nonlinear = self.space_solver.rhs_nonlinear
        rhs_linear = self.space_solver.rhs_linear
        noise = self.space_solver.noise
        forcing = self.space_solver.forcing

        n_stages = self.time_integrator.n_stages
        noise_factor = self.time_integrator.noise_factor

        u0 = self.space_solver.cached_array[(u_hat, 0, True)]

        while not np.isclose(self.t, self.end_time, atol=1e-10):
            self.t += dt
            self.step += 1

            u0[:] = u_hat
            self.space_solver.random_fields()
            for stage in range(n_stages):
                self.space_solver.compute_rhs(noise_factor[stage])
                self.time_integrator.stepping(u_hat, u0, stage, rhs_nonlinear, rhs_linear, forcing, noise)

            if self.write_solution and self.step % self.write_interval == 0:
                self.space_solver.backward()
                self.solution_writer.write(self.step)

            if self.step % self.check_interval == 0:
                self._timer(self.t, self.step)

        if self.write_solution:
            self.solution_writer.close()

        self._timer.final()
