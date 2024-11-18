import numpy as np
from periodicflow import simulation as sim


def taylor_green(x, local_shape):
    u0 = np.zeros(local_shape)

    u0[0] = np.sin(x[0]) * np.cos(x[1]) * np.cos(x[2])
    u0[1] = -np.cos(x[0]) * np.sin(x[1]) * np.cos(x[2])
    u0[2] = 0.0

    return u0


if __name__ == '__main__':
    n = 256
    dim = 3
    nu = 1 / 1600

    sol = sim.Solver(
        [n] * dim,
        1e-3,
        10,
        viscosity=1600,
        nonlinear=False,
        optimization=True,
        time_integrator='implicit_pc',
        verbose=True,
        write_solution=True,
        file_name='TG',
        write_interval=50,
        check_interval=1000
    )

    x = sol.x
    local_shape = sol.shape_local_physical
    u0 = taylor_green(x, local_shape)
    sol.initialize(u0)

    sol.solve()
    sol.solution_writer.reconfigure_dataset()   # reconfigure the dataset structure
