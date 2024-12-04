import numpy as np
import time
import mpi4py.MPI as mpi
import shenfun as sf
from periodicflow import logger


def periodic_bc(u: np.ndarray) -> np.ndarray:
    """
    Enforce periodic boundary conditions on the input data.
    """
    for axis in range(u.ndim):
        first_slice = np.take(u, indices=0, axis=axis)
        u = np.concatenate((u, np.expand_dims(first_slice, axis=axis)), axis=axis)
    return u


class Timer:
    """Class to measure the time taken for a simulation to run.

    Parameters
    ----------
    comm (MPI.Comm)
        The MPI communicator.
    verbose (bool)
        If True, prints timing information.
    """

    def __init__(self, comm: mpi.Comm, verbose: bool = False):
        self.comm = comm
        self.mpi_rank = comm.Get_rank()
        self.mpi_size = comm.Get_size()
        self.verbose = verbose

        self.start_time = time.time()
        self.t0 = self.start_time

    def __call__(self, simulation_time: float, step: int):
        """
        Measure the time since the last call and print if verbose.

        Parameters:
            simulation_time (float): The current simulation time.
            step (int): The current simulation step.
        """
        t1 = time.time()
        dt = t1 - self.t0
        self.t0 = t1

        dt_sum = self.comm.allreduce(dt, op=mpi.SUM)
        dt_avg = dt_sum / self.mpi_size

        if self.verbose and self.mpi_rank == 0:
            logger.info(f"Step = {step:08d}, time = {simulation_time:.2e}, runtime since last check = {self._format_time(dt_avg, 'mm:ss')}")

        self.comm.Barrier()

    def start(self):
        """Print the start time of the simulation."""
        if self.mpi_rank == 0:
            logger.info(f"Simulation started")

    def final(self):
        """Print the final timing information."""
        runtime = time.time() - self.start_time

        runtime_sum = self.comm.allreduce(runtime, op=mpi.SUM)
        runtime_avg = runtime_sum / self.mpi_size

        if self.mpi_rank == 0:
            logger.info(f"Simulation completed. Total run time: {self._format_time(runtime_avg, 'hh:mm:ss')}")

    @staticmethod
    def _format_time(seconds: float, format='dd-hh:mm:ss') -> str:
        """
        Format the time into various formats: 'dd-hh:mm:ss', 'hh:mm:ss', or 'mm:ss'.

        Parameters:
            seconds (float): Time in seconds.
            format (str): Desired format ('dd-hh:mm:ss', 'hh:mm:ss', 'mm:ss').

        Returns:
            str: Formatted time string.
        """
        days, remainder = divmod(seconds, 86400)  # Total seconds in a day
        hours, remainder = divmod(remainder, 3600)
        minutes, secs = divmod(remainder, 60)

        format = format.casefold()
        if format == 'mm:ss':
            return f"{int(minutes):02d}:{int(secs):02d}"
        elif format == 'hh:mm:ss':
            return f"{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
        elif format == 'dd-hh:mm:ss':
            return f"{int(days):02d}-{int(hours):02d}:{int(minutes):02d}:{int(secs):02d}"
        else:
            raise ValueError("Invalid format. Choose 'dd-hh:mm:ss', 'hh:mm:ss', or 'mm:ss'.")
