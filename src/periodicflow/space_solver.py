import numpy as np
import numba as nb
import mpi4py.MPI as mpi
from .io import Params
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
    mpi_comm : mpi.Comm
        MPI communicator used for parallel processing.
    mpi_rank : int
        Rank of the current processor.
    params : Params
        Object containing the simulation parameters.

    Notes
    -----
    This solver uses a Fourier spectral method for spatial discretization. The class is designed for parallel simulations
    and makes use of MPI to handle communication between different processors.
    """

    def __init__(
        self,
        mpi_comm: mpi.Comm,
        mpi_rank: int,
        params: Params,
    ):
        super().__init__(
            mpi_comm,
            mpi_rank,
            params.N,
            domain=params.domain,
            dealias=params.dealias,
            mask_nyquist=params.mask_nyquist,
            fft_plan=params.fft_plan,
            decomposition=params.decomposition
        )

        if self.dim not in (2, 3):
            try:
                raise ValueError("NS equation has to be in either 2D or 3D.")
            except ValueError as e:
                if self.mpi_rank == 0:
                    logger.critical(str(e))
                self.mpi_comm.Abort(1)

        self.cached_array = sf.CachedArrayDict()

        self._enable_nonlinear = params.enable_nonlinear
        self._filter_velocity = params.filter_velocity
        self.viscosity = params.viscosity

        self.noise_type = params.noise_type
        self.noise_mag = params.noise_mag
        if self.noise_type not in ('thermal', 'correlated'):
            try:
                raise ValueError(f"Invalid noise type '{self.noise_type}'. Options are 'thermal' or 'correlated'.")
            except ValueError as e:
                if self.mpi_rank == 0:
                    logger.error(str(e))
                self.mpi_comm.Abort(1)

        self._define_variables()

        self.local_shape_physical = self.u.shape
        self.local_shape_fourier = self.u_hat.shape

        self.local_slice_physical = self.u.local_slice()
        self.local_slice_fourier = self.u_hat.local_slice()

        self._k0_mask_0 = np.where(self.k2 == 0, 0, 1)
        self._prod_n_sqrt = np.sqrt(np.prod(params.N))
        # self._noise_normal_factor = self._k0_mask_0 / self._prod_n_sqrt / np.sqrt(2 - 2 / self.dim)
        self._noise_normal_factor = self._k0_mask_0 / self._prod_n_sqrt / np.sqrt(2)

        self.linear_operator = -params.viscosity * self.k2
        self.correlation_func = sf.Function(self.S)
        self.filter_kernel = sf.Function(self.S)
        self.filter_kernel[:] = 1.0

    def _define_variables(self):
        self.u = sf.Array(self.V)
        self.u_hat = sf.Function(self.V)
        self.u_dealias = sf.Array(self.Vp)

        if self._filter_velocity:
            self.u_bar = sf.Array(self.V)
            self.u_bar_hat = sf.Function(self.V)

        self._uw = self.cached_array[(self.u_dealias, 0, True)]

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

    def _initialize(self, u0: np.ndarray, space: str = 'physical', mask_zero_mode: bool = True):
        if space.casefold() == 'physical':
            if u0.shape != self.u.shape:
                try:
                    raise ValueError(
                        f"Invalid shape for the input velocity. Expected {self.u.shape}, got {u0.shape}."
                    )
                except ValueError as e:
                    if self.mpi_rank == 0:
                        logger.critical(str(e))
                    self.mpi_comm.Abort(1)

            self.u[:] = u0
            if self._filter_velocity:
                self.u_bar[:] = u0
            self.forward()

        elif space.casefold() == 'fourier':
            if u0.shape != self.u_hat.shape:
                try:
                    raise ValueError(
                        f"Invalid shape for the input velocity. Expected {self.u_hat.shape}, got {u0.shape}."
                    )
                except ValueError as e:
                    if self.mpi_rank == 0:
                        logger.critical(str(e))
                    self.mpi_comm.Abort(1)

            self.u_hat[:] = u0
            if self._filter_velocity:
                self.u_bar_hat[:] = u0
            self.backward()
        else:
            raise ValueError("SpaceSolver.initialize_velocity: Invalid space type. Options are 'physical' or 'fourier'.")

        if mask_zero_mode:
            self.u_hat *= self._k0_mask_0

    def forward(self):
        """
        Call the forward transform from the vector space in space_solver
        """
        self.u_hat = self.V.forward(self.u, self.u_hat)

        if self._filter_velocity:
            self.u_bar_hat = self.V.forward(self.u_bar, self.u_bar_hat)

    def backward(self):
        """
        Call the backward transform from the vector space in space_solver
        """
        if self._filter_velocity:
            self.u_bar = self.V.backward(self.u_bar_hat, self.u_bar)
        else:
            self.u = self.V.backward(self.u_hat, self.u)

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
        """Generate spatially correlated noise field."""
        apply_linear_operator(self.noise, self.correlation_func, self.w_hat)

    def add_forcing(self):
        """Generate external forcing field."""
        pass

    def filter_velocity(self):
        """Filter the velocity field."""
        apply_linear_operator(self.u_bar_hat, self.filter_kernel, self.u_hat)

    def compute_rhs_linear(self):
        """Compute the linear part of the right-hand side."""
        apply_linear_operator(self.rhs_linear, self.linear_operator, self.u_hat)

    def compute_vorticity(self):
        """Compute the vorticity field in Fourier space."""
        curl(self.vort_hat, self.k, self.u_hat)
        self.vort_dealias = self.Wp.backward(self.vort_hat, self.vort_dealias)

    def compute_rhs_nonlinear(self):
        """
        Compute the nonlinear part of the right-hand side.
        """
        self.compute_vorticity()
        self.u_dealias = self.Vp.backward(self.u_hat, self.u_dealias)

        cross(self._uw, self.u_dealias, self.vort_dealias)

        self.rhs_nonlinear = self.Vp.forward(self._uw, self.rhs_nonlinear)
        leray_projection(self.rhs_nonlinear, self.k, self.k_over_k2, self.p_hat)

        if self.nyquist_mask is not None:
            self.rhs_nonlinear.mask_nyquist(self.nyquist_mask)

    def compute_rhs(self, a: float = 0.0):
        """
        Assemble the overall right-hand side.
        """
        self.compute_rhs_linear()
        if self._enable_nonlinear:
            self.compute_rhs_nonlinear()
        self.add_noise(a)
        self.add_forcing()
