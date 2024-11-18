import numpy as np
import numba as nb
import mpi4py.MPI as mpi
from .basis import FourierSpace
from .math import cross, curl, apply_linear_operator, leray_projection
from periodicflow import logger


try:
    import shenfun as sf
except ImportError:
    raise ImportError('shenfun is required for this module')


@nb.njit
def _symmetrize(w_hat_sym, w_hat):
    dim = w_hat_sym.ndim
    shape = w_hat_sym.shape

    factor = np.sqrt(0.5)

    if dim == 3:
        for j in range(shape[1]):
            for k in range(shape[2]):
                w_hat_sym[0, j, k] = (w_hat[0, j, k] + w_hat[0, j, k]) * factor
                w_hat_sym[1, j, k] = (w_hat[1, j, k] + w_hat[2, j, k]) * factor
                w_hat_sym[2, j, k] = (w_hat[2, j, k] + w_hat[1, j, k]) * factor
                w_hat_sym[3, j, k] = (w_hat[3, j, k] + w_hat[3, j, k]) * factor

    elif dim == 4:
        for j in range(shape[1]):
            for k in range(shape[2]):
                for l in range(shape[3]):
                    w_hat_sym[0, j, k, l] = (w_hat[0, j, k, l] + w_hat[0, j, k, l]) * factor
                    w_hat_sym[1, j, k, l] = (w_hat[1, j, k, l] + w_hat[3, j, k, l]) * factor
                    w_hat_sym[2, j, k, l] = (w_hat[2, j, k, l] + w_hat[6, j, k, l]) * factor
                    w_hat_sym[3, j, k, l] = (w_hat[3, j, k, l] + w_hat[1, j, k, l]) * factor
                    w_hat_sym[4, j, k, l] = (w_hat[4, j, k, l] + w_hat[4, j, k, l]) * factor
                    w_hat_sym[5, j, k, l] = (w_hat[5, j, k, l] + w_hat[7, j, k, l]) * factor
                    w_hat_sym[6, j, k, l] = (w_hat[6, j, k, l] + w_hat[2, j, k, l]) * factor
                    w_hat_sym[7, j, k, l] = (w_hat[7, j, k, l] + w_hat[5, j, k, l]) * factor
                    w_hat_sym[8, j, k, l] = (w_hat[8, j, k, l] + w_hat[8, j, k, l]) * factor


@nb.njit
def _noise_divergence(noise, w_hat, k):
    dim = noise.ndim
    shape = noise.shape

    if dim == 3:
        for i in range(shape[1]):
            for l in range(shape[2]):
                kx, ky = k[0, i, l], k[1, i, l]
                noise[0, i, l] = 1j * (kx * w_hat[0, i, l] + ky * w_hat[1, i, l])
                noise[1, i, l] = 1j * (kx * w_hat[2, i, l] + ky * w_hat[3, i, l])

    elif dim == 4:
        for i in range(shape[1]):
            for j in range(shape[2]):
                for l in range(shape[3]):
                    kx, ky, kz = k[0, i, j, l], k[1, i, j, l], k[2, i, j, l]
                    noise[0, i, j, l] = 1j * (kx * w_hat[0, i, j, l] + ky * w_hat[1, i, j, l] + kz * w_hat[2, i, j, l])
                    noise[1, i, j, l] = 1j * (kx * w_hat[3, i, j, l] + ky * w_hat[4, i, j, l] + kz * w_hat[5, i, j, l])
                    noise[2, i, j, l] = 1j * (kx * w_hat[6, i, j, l] + ky * w_hat[7, i, j, l] + kz * w_hat[8, i, j, l])


class SpaceSolver(FourierSpace):
    """
    Class for handling spatial discretization of the incompressible Navier-Stokes equations.

    Parameters
    ----------
    comm : mpi.Comm
        MPI communicator used for parallel processing.
    mpi_rank : int
        Rank of the current processor.
    N : list or tuple of int
        Number of grid points in each direction.
    is_nonlinear : bool, optional
        Whether to include the nonlinear term in the Navier-Stokes equations. Default is True.
    viscosity : float, optional
        Kinematic viscosity of the fluid. Default is 1e-3.
    noise_type : str, optional
        Type of noise to apply. Options are 'thermal' or 'correlated'. Default is 'thermal'.
    noise_mag : float, optional
        Magnitude of the noise. Default is 0.0 (no noise).
    **kwargs
        Additional keyword arguments passed to the parent class `FourierSpace`:

        - domain (float or list or tuple, optional)
            Domain size in each direction. Default is `2 * np.pi`.
        - dealias (str, optional)
            Dealiasing rule for Fourier transforms. Options are '2/3'/'truncate' (truncation) or '3/2'/'pad' (zero-padding). Default is '3/2'.
        - mask_nyquist (bool, optional)
            Whether to mask the Nyquist components setting them to zero. Default is False. 
        - fft_plan (str, optional)
            FFT planner effort, e.g., 'FFTW_MEASURE' for efficient computation. Default is 'FFTW_MEASURE'.
        - decomposition (str, optional)
            Parallel decomposition strategy. Options are 'slab' or 'pencil'. Default is 'slab'.

    Notes
    -----
    This solver uses a Fourier spectral method for spatial discretization. The class is designed for parallel simulations
    and makes use of MPI to handle communication between different processors.
    """

    def __init__(
        self,
        comm: mpi.Comm,
        mpi_rank: int,
        N: list | tuple,
        is_nonlinear: bool = True,
        viscosity: float = 1e-3,
        noise_type: str = 'thermal',
        noise_mag: float = 0.0,
        **kwargs
    ):
        super().__init__(comm, mpi_rank, N, **kwargs)

        if self.dim not in (2, 3):
            logger.error("NS solver only supports 2D and 3D problems.")
            raise ValueError("NS solver only supports 2D and 3D problems.")

        self._is_nonlinear = is_nonlinear
        self.viscosity = viscosity

        self.noise_type = noise_type
        self.noise_mag = noise_mag
        if self.noise_type not in ('thermal', 'correlated'):
            if self.mpi_rank == 0:
                logger.error("Invalid noise type. Options are 'thermal' or 'correlated'.")
            raise ValueError("Invalid noise type. Options are 'thermal' or 'correlated'.")

        self._define_variables()

        self._k0_mask_0 = np.where(self.k2 == 0, 0, 1)
        self._prod_n_sqrt = np.sqrt(np.prod(N))
        self._noise_normal_factor = self._k0_mask_0 / self._prod_n_sqrt / np.sqrt(2 - 2 / self.dim)

        self.linear_operator = -viscosity * self.k2
        self.correlation_func = sf.Function(self.S)

        self.cached_array = sf.CachedArrayDict()

    def _define_variables(self):
        self.u = sf.Array(self.V)
        self.u_hat = sf.Function(self.V)
        self.u_dealias = sf.Array(self.Vp)

        self.shape_local_physical = self.u.shape
        self.shape_local_fourier = self.u_hat.shape

        self.p = sf.Array(self.S)
        self.p_hat = sf.Function(self.S)

        self.vort = sf.Array(self.W)
        self.vort_hat = sf.Function(self.W)
        self.vort_dealias = sf.Array(self.Wp)

        self.forcing = sf.Function(self.V)

        field = self.T if self.noise_type == 'thermal' else self.V
        self.w1 = sf.Function(field)
        self.w2 = sf.Function(field)
        self.w_hat = sf.Function(field)
        self.noise = sf.Function(self.V)
        if self.noise_type == 'thermal':
            self.w_hat_sym = sf.Function(field)

        self.rhs_linear = sf.Function(self.V)
        self.rhs_nonlinear = sf.Function(self.V)

    def initialize_velocity(self, u0: np.ndarray, space: str = 'physical', mask_zero_mode: bool = True):
        if space.casefold() == 'physical':
            if u0.shape != self.u.shape:
                if self.mpi_rank == 0:
                    logger.error("SpaceSolver.initialize_velocity: Invalid shape for the input physical velocity.")
                raise ValueError(
                    f"SpaceSolver.initialize_velocity: Invalid shape for the input physical velocity. "
                    f"Expected {self.u.shape}, got {u0.shape}."
                )
            self.u[:] = u0
            self.forward()

        elif space.casefold() == 'fourier':
            if u0.shape != self.u_hat.shape:
                if self.mpi_rank == 0:
                    logger.error("SpaceSolver.initialize_velocity: Invalid shape for the input Fourier velocity.")
                raise ValueError(
                    f"SpaceSolver.initialize_velocity: Invalid shape for the input Fourier velocity. "
                    f"Expected {self.u_hat.shape}, got {u0.shape}."
                )

            self.u_hat[:] = u0
            if mask_zero_mode:
                self.u_hat *= self._k0_mask_0

            self.backward()
        else:
            raise ValueError("SpaceSolver.initialize_velocity: Invalid space type. Options are 'physical' or 'fourier'.")

    def forward(self):
        """
        Call the forward transform from the vector space in space_solver 
        """
        self.V.forward(self.u, self.u_hat)

    def backward(self):
        """
        Call the backward transform from the vector space in space_solver
        """
        self.V.backward(self.u_hat, self.u)

    def random_fields(self):
        """
        Generate random white Gaussian fields for the noise terms.
        """
        shape = self.w1.shape
        normal_factor = self._noise_normal_factor

        self.w1[:] = np.random.randn(*shape) + 1j * np.random.randn(*shape)
        self.w2[:] = np.random.randn(*shape) + 1j * np.random.randn(*shape)

        self.w1 *= normal_factor
        self.w2 *= normal_factor

    def add_noise(self, a: float):
        """
        Generate noise fields.
        """
        self.w_hat[:] = self.w1 + a * self.w2

        if self.noise_type == 'thermal':
            self.thermal_noise()
        elif self.noise_type == 'correlated':
            self.correlated_noise()

        leray_projection(self.noise, self.k, self.k_over_k2, self.p_hat)
        self.noise *= self.noise_mag

    def thermal_noise(self):
        """
        Generate thermal noise field.
        """
        _symmetrize(self.w_hat_sym, self.w_hat)
        _noise_divergence(self.noise, self.w_hat_sym, self.k)

    def correlated_noise(self):
        """
        Generate spatially correlated noise field.
        """
        apply_linear_operator(self.noise, self.correlation_func, self.w_hat)

    def add_forcing(self):
        """
        Generate external forcing field.
        """
        pass

    def compute_rhs_linear(self):
        """
        Compute the linear part of the right-hand side.
        """
        apply_linear_operator(self.rhs_linear, self.linear_operator, self.u_hat)

    def compute_vorticity(self):
        """
        Compute the vorticity field in Fourier space.
        """
        curl(self.vort_hat, self.k, self.u_hat)
        self.Wp.backward(self.vort_hat, self.vort_dealias)

    def compute_rhs_nonlinear(self):
        """
        Compute the nonlinear part of the right-hand side.
        """
        self.compute_vorticity()
        self.Vp.backward(self.u_hat, self.u_dealias)

        uw = self.cached_array[(self.u_dealias, 0, False)]
        cross(uw, self.u_dealias, self.vort_dealias)

        self.Vp.forward(uw, self.rhs_nonlinear)
        leray_projection(self.rhs_nonlinear, self.k, self.k_over_k2, self.p_hat)

        if self.nyquist_mask is not None:
            self.rhs_nonlinear.mask_nyquist(self.nyquist_mask)

    def compute_rhs(self, a: float = 0.0):
        """
        Assemble the overall right-hand side.
        """
        self.compute_rhs_linear()
        if self._is_nonlinear:
            self.compute_rhs_nonlinear()
        self.add_noise(a)
        self.add_forcing()
