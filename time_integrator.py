import numpy as np
import numba as nb
from abc import ABC
from .math import apply_linear_operator


DICT_TIME_INTEGRATORS = {}


def register_integrator(name):
    def decorator(cls):
        DICT_TIME_INTEGRATORS[name] = cls
        return cls
    return decorator


class TimeIntegrator(ABC):
    def __init__(
            self,
            dt: float,
            linear_operator: np.ndarray = None,
            optimization: bool = False
    ):
        self.dt = dt
        self._L = linear_operator
        self._optimization = optimization

        if linear_operator is not None:
            self._Linv = 1 / (1 - self.dt / 2 * linear_operator)

    def update_linear_operator(self, L: np.ndarray):
        self._L[:] = L
        self._Linv[:] = 1 / (1 - self.dt / 2 * L)


@register_integrator('explicit_pc')
class ExplicitPredictorCorrector(TimeIntegrator):
    """
    This is a 2-stage explicit predictor-corrector time integrator proposed by Delong et al. (2013, PRE)
    """

    def __init__(self, dt: float, linear_operator: np.ndarray, optimization=False):
        super().__init__(dt, linear_operator, optimization)

        self.n_stages = 2
        self._increment_factor = (0.5, 1.0)
        self.noise_factor = (0.0, 1.0)

        self.stepping = self._stepping_optimized if optimization else self._stepping_regular

    def _stepping_regular(self, unew, u0, stage, rhs_nonlinear, rhs_linear, forcing, noise):
        a = self._increment_factor[stage]
        dt = self.dt
        dt_sqrt = np.sqrt(0.5 * dt)

        unew[:] = u0 + a * dt * (rhs_nonlinear + rhs_linear + forcing) + dt_sqrt * noise

    def _stepping_optimized(self, unew, u0, stage, rhs_nonlinear, rhs_linear, forcing, noise):
        a = self._increment_factor[stage]
        dt = self.dt
        dt_sqrt = np.sqrt(0.5 * dt)

        self._loops_numba(unew, a, dt, dt_sqrt, u0, rhs_nonlinear, rhs_linear, forcing, noise)

    @staticmethod
    @nb.njit
    def _loops_numba(unew, a, dt, dt_sqrt, u0, rhs_nonlinear, rhs_linear, forcing, noise):
        ndim = u0.ndim
        shape = u0.shape

        if ndim == 3:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        unew[i, j, k] = (
                            u0[i, j, k]
                            + a * dt * (rhs_nonlinear[i, j, k] + rhs_linear[i, j, k] + forcing[i, j, k])
                            + dt_sqrt * noise[i, j, k]
                        )

        elif ndim == 4:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        for l in range(shape[3]):
                            unew[i, j, k, l] = (
                                u0[i, j, k, l]
                                + a * dt * (rhs_nonlinear[i, j, k, l] + rhs_linear[i, j, k, l] + forcing[i, j, k, l])
                                + dt_sqrt * noise[i, j, k, l]
                            )


@register_integrator('implicit_pc')
class ImplicitPredictorCorrector(TimeIntegrator):
    """
    This is a 2-stage implicit predictor-corrector time integrator proposed by Delong et al. (2013, PRE)
    """

    def __init__(self, dt: float, linear_operator: np.ndarray, optimization=False):
        super().__init__(dt, linear_operator, optimization)

        self.n_stages = 2
        self._increment_factor = [
            (0.5, 0.0),
            (1.0, 0.5)
        ]
        self.noise_factor = (0.0, 1.0)

        self.stepping = self._stepping_optimized if optimization else self._stepping_regular

    def _stepping_regular(self, unew, u0, stage, rhs_nonlinear, rhs_linear, forcing, noise):
        a, b = self._increment_factor[stage]
        dt = self.dt
        dt_sqrt = np.sqrt(0.5 * dt)

        if stage == self.n_stages - 1:
            apply_linear_operator(rhs_linear, self._L, u0)

        unew[:] = self._Linv * (u0 + a * dt * (rhs_nonlinear + forcing) + b * dt * rhs_linear + dt_sqrt * noise)

    def _stepping_optimized(self, unew, u0, stage, rhs_nonlinear, rhs_linear, forcing, noise):
        a, b = self._increment_factor[stage]
        dt = self.dt
        dt_sqrt = np.sqrt(0.5 * dt)

        if stage == self.n_stages - 1:
            apply_linear_operator(rhs_linear, self._L, u0)

        self._loops_numba(unew, self._Linv, dt, dt_sqrt, a, b, u0, rhs_nonlinear, rhs_linear, forcing, noise)

    @staticmethod
    @nb.njit
    def _loops_numba(unew, Linv, dt, dt_sqrt, a, b, u0, rhs_nonlinear, rhs_linear, forcing, noise):
        ndim = u0.ndim
        shape = u0.shape

        if ndim == 3:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        unew[i, j, k] = Linv[j, k] * (
                            u0[i, j, k]
                            + a * dt * (rhs_nonlinear[i, j, k] + forcing[i, j, k])
                            + b * dt * rhs_linear[i, j, k]
                            + dt_sqrt * noise[i, j, k]
                        )

        elif ndim == 4:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        for l in range(shape[3]):
                            unew[i, j, k, l] = Linv[j, k, l] * (
                                u0[i, j, k, l]
                                + a * dt * (rhs_nonlinear[i, j, k, l] + forcing[i, j, k, l])
                                + b * dt * rhs_linear[i, j, k, l]
                                + dt_sqrt * noise[i, j, k, l]
                            )


@register_integrator('rk2')
class RungeKuttaTVD2(TimeIntegrator):
    """
    2nd-order Runge-Kutta TVD time integrator proposed by Shu & Osher (1998).
    """

    def __init__(self, dt: float, linear_operator: np.ndarray, optimization=False):
        super().__init__(dt, linear_operator, optimization)

        self.n_stages = 2
        self._increment_factors = [
            (0.0, 1.0),
            (0.5, 0.5)
        ]

        self.stepping = self._stepping_optimized if optimization else self._stepping_regular

    def _stepping_regular(self, unew, u0, stage, rhs_nonlinear, rhs_linear, forcing, noise=None):
        a, b = self._increment_factors[stage]
        unew[:] = a * u0 + b * (unew + self.dt * (rhs_nonlinear + rhs_linear + forcing))

    def _stepping_optimized(self, unew, u0, stage, rhs_nonlinear, rhs_linear, forcing, noise=None):
        dt = self.dt
        a, b = self._increment_factors[stage]

        self._loops_numba(unew, a, b, dt, u0, rhs_nonlinear, rhs_linear, forcing, noise)

    @staticmethod
    @nb.njit
    def _loops_numba(unew, a, b, dt, u0, rhs_nonlinear, rhs_linear, forcing, noise=None):
        ndim = u0.ndim
        shape = u0.shape

        if ndim == 3:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        unew[i, j, k] = a * u0[i, j, k] + b * (
                            unew[i, j, k]
                            + dt * (rhs_nonlinear[i, j, k] + rhs_linear[i, j, k] + forcing[i, j, k])
                        )

        elif ndim == 4:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        for l in range(shape[3]):
                            unew[i, j, k, l] = a * u0[i, j, k, l] + b * (
                                unew[i, j, k, l] + dt * (rhs_nonlinear[i, j, k, l] + rhs_linear[i, j, k, l] + forcing[i, j, k, l])
                            )


@register_integrator('rk3')
class RungeKuttaTVD3(TimeIntegrator):
    """
    This is an stochastic extension, made by Delong et al. (2013, PRE), of the TVD RK3 proposed by Shu & Osher (1998).
    """

    def __init__(self, dt: float, linear_operator: np.ndarray, optimization=False):
        super().__init__(dt, linear_operator, optimization)

        self.n_stages = 3
        self._increment_factor = [
            (0.0, 1.0),
            (3.0 / 4.0, 1.0 / 4.0),
            (1.0 / 3.0, 2.0 / 3.0)
        ]
        self.noise_factor = [
            (2 * np.sqrt(2) + np.sqrt(3)) / 5,
            (-4 * np.sqrt(2) + 3 * np.sqrt(3)) / 5,
            (np.sqrt(2) - 2 * np.sqrt(3)) / 10
        ]

        self.stepping = self._stepping_optimized if optimization else self._stepping_regular

    def _stepping_regular(self, unew, u0, stage, rhs_nonlinear, rhs_linear, forcing, noise):
        a, b = self._increment_factor[stage]
        dt = self.dt
        dt_sqrt = np.sqrt(dt)

        unew[:] = a * u0 + b * (unew + self.dt * (rhs_nonlinear + rhs_linear + forcing) + dt_sqrt * noise)

    def _stepping_optimized(self, unew, u0, stage, rhs_nonlinear, rhs_linear, forcing, noise):
        a, b = self._increment_factor[stage]
        dt = self.dt
        dt_sqrt = np.sqrt(dt)

        self._loops_numba(unew, a, b, dt, dt_sqrt, u0, rhs_nonlinear, rhs_linear, forcing, noise)

    @staticmethod
    @nb.njit
    def _loops_numba(unew, a, b, dt, dt_sqrt, u0, rhs_nonlinear, rhs_linear, forcing, noise=None):
        ndim = u0.ndim
        shape = u0.shape

        if ndim == 3:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        unew[i, j, k] = a * u0[i, j, k] + b * (
                            unew[i, j, k]
                            + dt * (rhs_nonlinear[i, j, k] + rhs_linear[i, j, k] + forcing[i, j, k])
                            + dt_sqrt * noise[i, j, k]
                        )

        elif ndim == 4:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        for l in range(shape[3]):
                            unew[i, j, k, l] = a * u0[i, j, k, l] + b * (
                                unew[i, j, k, l]
                                + dt * (rhs_nonlinear[i, j, k, l] + rhs_linear[i, j, k, l] + forcing[i, j, k, l])
                                + dt_sqrt * noise[i, j, k, l]
                            )


@register_integrator('rk4')
class RungeKutta4(TimeIntegrator):
    """
    Standard low-storage 4th-order Runge-Kutta time integrator following Bogey & Bailly (2004).
    """

    def __init__(self, dt: float, linear_operator: np.ndarray, optimization=False):
        super().__init__(dt, linear_operator, optimization)

        self.n_stages = 4
        self._increment_factor = [1. / 4., 1. / 3., 1. / 2., 1.]

        self.stepping = self._stepping_optimized if optimization else self._stepping_regular

    def _stepping_regular(self, unew, u0, stage, rhs_nonlinear, rhs_linear, forcing, noise):
        unew[:] = u0 + self._increment_factor[stage] * self.dt * (rhs_nonlinear + rhs_linear + forcing)

    def _stepping_optimized(self, unew, u0, stage, rhs_nonlinear, rhs_linear, forcing, noise):
        dt = self.dt
        a = self._increment_factor[stage]

        self._loops_numba(unew, a, dt, u0, rhs_nonlinear, rhs_linear, forcing, noise)

    @staticmethod
    @nb.njit
    def _loops_numba(unew, a, dt, u0, rhs_nonlinear, rhs_linear, forcing, noise):
        ndim = u0.ndim
        shape = u0.shape

        if ndim == 3:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        unew[i, j, k] = u0[i, j, k] + a * dt * (rhs_nonlinear[i, j, k] + rhs_linear[i, j, k] + forcing[i, j, k])

        elif ndim == 4:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        for l in range(shape[3]):
                            unew[i, j, k, l] = u0[i, j, k, l] + a * dt * (rhs_nonlinear[i, j, k, l] + rhs_linear[i, j, k, l] + forcing[i, j, k, l])
