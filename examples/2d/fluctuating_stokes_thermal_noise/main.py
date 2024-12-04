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
    sol = sim.Solver()

    shape = sol.local_shape_fourier
    u0 = initial_condition(shape, sol.params.N[0], 4.2052)
    sol.initialize(u0, 'fourier')

    sol.solve()
    sol.data_writer.reconfigure_dataset()
