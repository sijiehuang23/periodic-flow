import numpy as np
import argparse
from periodicflow import simulation as sim
from periodicflow import logger


def parser_args(comm, rank):
    args = None
    try:
        if rank == 0:
            logger.info(f"Parsing arguments.")
            parser = argparse.ArgumentParser()
            parser.add_argument("L", default=0.0, help="Correlation length", type=float)
            args = parser.parse_args()
    except Exception as e:
        logger.error(f"[Rank {rank}] Error parsing arguments: {e}")
        comm.Abort(1)

    return comm.bcast(args, root=0)


def initial_condition(shape, n, kBT):
    dim = shape[0]
    factor = n**(dim / 2) / kBT**0.5 / (2 - 2 / dim)

    u0 = np.random.randn(*shape) + 1j * np.random.randn(*shape)
    for i in range(dim):
        u0[i] /= np.std(u0[i]) * factor

    return u0


def correlator(l, k, k2, dx):
    dim = k.shape[0]
    prefactor = (np.sqrt(2 * np.pi * l**2) / dx)**dim

    if l > 1e-10:
        if dim == 2:
            kx, ky = k
            C_l = np.exp(-0.5 * ((l * kx)**2 + (l * ky)**2)) * prefactor
        else:
            kx, ky, kz = k
            C_l = np.exp(-0.5 * ((l * kx)**2 + (l * ky)**2 + (l * kz**2))) * prefactor

    else:
        C_l = np.ones_like(k2)

    C_l *= np.where(k2 == 0, 0, 1)

    return C_l


if __name__ == '__main__':
    """
    This script solves for the fluctuating Stokes equation subject to spatially correlated noise.

    To run the case, do 
        ```bash
        mpirun -np <nprocs> python fluctuating_stokes_correlated_noise.py <corr_length>
        ```
    """

    n = 64
    dim = 2
    nu, kBT = 818.72, 4.2052

    sol = sim.Solver(
        [n] * dim,
        0.01,
        10,
        viscosity=nu,
        is_nonlinear=False,
        time_integrator='implicit_pc',
        write_solution=True,
        file_name='correlated',
        write_interval=5,
        noise_type='correlated',
        noise_mag=(2 * nu * kBT)**0.5
    )

    shape = sol.shape_local_fourier
    u0 = initial_condition(shape, n, kBT)
    sol.initialize(u0, 'fourier')

    args = parser_args(sol.comm, sol.mpi_rank)
    dx = sol.x[0][1] - sol.x[0][0]
    C_l = correlator(args.L, sol.k, sol.k2, dx)

    sol.set_linear_operator(-nu * C_l)
    sol.set_correlation_function(np.sqrt(C_l))

    sol.solve()
    sol.solution_writer.reconfigure_dataset()
