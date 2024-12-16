import numpy as np
from periodicflow.io import Params
from periodicflow import simulation as sim


def initial_condition(shape, n, kBT):
    dim = shape[0]
    factor = kBT**0.5 / n**(dim / 2)

    u0 = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    for i in range(dim):
        u0[i] *= factor / np.std(u0[i])
    return u0


if __name__ == '__main__':
    """
    This script solves for the fluctuating Stokes equation subject to the regular thermal noise.
    """
    params = Params()
    sol = sim.Solver(params)

    shape = sol.local_shape_fourier
    u0 = initial_condition(shape, sol.params.N[0], 4.2052)
    sol.initialize(u0, 'fourier')

    sol.solve()
    sol.data_writer.reconfigure_dataset()
