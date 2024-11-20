import numpy as np
from periodicflow import simulation as sim


def initial_condition(shape, n, kBT):
    dim = shape[0]
    factor = n**(dim / 2) / kBT**0.5 * np.sqrt(1 - 1 / dim)

    u0 = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    for i in range(dim):
        u0[i] /= np.std(u0[i]) * factor
    return u0


if __name__ == '__main__':
    """
    This script solves for the fluctuating Stokes equation subject to the regular thermal noise.
    """

    n = 256
    dim = 2
    nu, kBT = 818.72, 4.2052

    sol = sim.Solver(
        [n] * dim,
        1e-3,
        10,
        viscosity=nu,
        is_nonlinear=False,
        time_integrator='implicit_pc',
        write_solution=True,
        file_name='thermal',
        write_interval=2,
        noise_type='thermal',
        noise_mag=(2 * nu * kBT)**0.5
    )

    shape = sol.shape_local_fourier
    u0 = initial_condition(shape, n, kBT)
    sol.initialize(u0, 'fourier')

    sol.solve()
    sol.solution_writer.reconfigure_dataset()
