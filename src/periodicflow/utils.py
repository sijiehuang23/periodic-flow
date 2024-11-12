import numpy as np
import time
from datetime import datetime
import mpi4py.MPI as mpi
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

        # self.start_wall_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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

        dt_sum = self.comm.reduce(dt, op=mpi.SUM, root=0)
        dt_avg = dt_sum / self.mpi_size if self.mpi_size > 1 else dt

        if self.verbose and self.mpi_rank == 0:
            logger.info(f"Step = {step:08d}, time = {simulation_time:.2e}, runtime since last check = {self._format_time(dt_avg)}")

        self.comm.Barrier()

    def start(self):
        """Print the start time of the simulation."""
        if self.mpi_rank == 0:
            logger.info(f"Simulation started")

    def final(self):
        """Print the final timing information."""
        # end_wall_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        runtime = time.time() - self.start_time

        runtime_sum = self.comm.reduce(runtime, op=mpi.SUM, root=0)
        runtime_avg = runtime_sum / self.mpi_size if self.mpi_size > 1 else runtime

        if self.mpi_rank == 0:
            logger.info(f"Simulation completed")
            logger.info(f"  Total run time: {self._format_time(runtime_avg)}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        """
        Format the time in DD-HH:MM:SS format.

        Parameters:
            seconds (float): Time in seconds.

        Returns:
            str: Formatted time string.
        """
        days = int(seconds // 86400)  # Total seconds in a day
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{days:02d}-{hours:02d}:{minutes:02d}:{secs:02d}"
