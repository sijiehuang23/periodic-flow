import numpy as np
from periodicflow import simulation as sim


if __name__ == '__main__':
    """
    This script solves for the fluctuating Stokes equation subject to the regular thermal noise.
    """

    n = 128
    dim = 2

    nu = 1
    kBT = 4

    sol = sim.Solver(
        [n] * dim,
        0.01,
        10,
        viscosity=nu,
        optimization=True,
        time_integrator='implicit_pc',
        write_solution=True,
        file_name='thermal',
        write_interval=5,
        noise_mag=(2 * nu * kBT)**0.5
    )

    if dim == 2:
        shape = (dim, n, n // 2 + 1)
    else:
        shape = (dim, n, n, n // 2 + 1)

    u0 = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    for i in range(dim):
        u0[i] /= np.std(u0[i]) * n**(dim / 2) / kBT**0.5

    sol.initialize(u0, 'fourier')

    sol.solve()
    sol.solution_writer.reconfigure_dataset()
