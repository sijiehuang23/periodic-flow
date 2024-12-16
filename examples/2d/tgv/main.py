import numpy as np
from periodicflow.io import Params
from periodicflow import simulation as sim


def taylor_green(x, local_shape):
    u0 = np.empty(local_shape)

    u0[0] = -np.cos(x[0]) * np.sin(x[1])
    u0[1] = np.sin(x[0]) * np.cos(x[1])

    return u0


if __name__ == '__main__':
    params = Params()
    sol = sim.Solver(params)

    x = sol.x
    local_shape = sol.local_shape_physical
    u0 = taylor_green(x, local_shape)
    sol.initialize(u0)

    sol.solve()
    sol.data_writer.reconfigure_dataset()
