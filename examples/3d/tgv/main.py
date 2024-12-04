import numpy as np
from periodicflow import simulation as sim


def taylor_green(x, local_shape):
    u0 = np.zeros(local_shape)

    u0[0] = np.sin(x[0]) * np.cos(x[1]) * np.cos(x[2])
    u0[1] = -np.cos(x[0]) * np.sin(x[1]) * np.cos(x[2])
    u0[2] = 0.0

    return u0


if __name__ == '__main__':
    sol = sim.Solver()

    x = sol.x
    local_shape = sol.local_shape_physical
    u0 = taylor_green(x, local_shape)
    sol.initialize(u0)

    sol.solve()
    sol.data_writer.reconfigure_dataset()   # reconfigure the dataset structure
