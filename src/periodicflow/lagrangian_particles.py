"""
This module contains the LagrangianParticles class (and its helper classes) for tracking the trajectory of passive particles in a fluid flow field.

The module is designed somewhat like an independent package, not integrated with the rest of the periodicflow package for the sake of simplicity.

Example:
```python
from periodicflow.lagrangian_particles import LagrangianParticles
```
"""

from pathlib import Path
import numpy as np
import scipy as sp
import numba as nb
import h5py
import time
from pathlib import Path
from datetime import datetime
from .utils import format_time


__all__ = ['LagrangianParticles']


class _Timer:
    def __init__(self, check_interval: int = 10, verbose: bool = False):
        self.check_interval = check_interval
        self.verbose = verbose
        self.tic = self.t_check = 0.0
        self.start_time = ""

    def __call__(self, step, t):
        if step % self.check_interval == 0 and self.verbose:
            t_curr = time.time()
            dt = t_curr - self.t_check
            self.t_check = t_curr
            print(f"\tStep = {step:06d}, time = {t:.2e}, runtime since last check = {format_time(dt, 'mm:ss')}")

    def start(self):
        self.tic = self.t_check = time.time()
        self.start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if self.verbose:
            print(f"LPT started at {self.start_time}")

    def stop(self):
        runtime = time.time() - self.tic
        end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if self.verbose:
            print(f"LPT finished at {end_time}\n  Total runtime: {format_time(runtime, 'hh:mm:ss')}")


class _H5Reader:
    def __init__(self, filename: str, velocity_path: list[str]):
        self.filename = filename
        self._velocity_path = velocity_path
        self._ndim = len(velocity_path)

        self._get_info()

        self.velocity = np.empty((self._ndim, *self._shape), dtype=np.float64)

    def _get_info(self):
        if not Path(self.filename).is_file():
            raise FileNotFoundError(f"File '{self.filename}' not found.")

        self.open()
        for vel_path in self._velocity_path:
            if vel_path not in self.f:
                raise KeyError(f"Group '{vel_path}' not found in file '{self.filename}'.")

            steps = sorted(list(self.f[vel_path].keys()), key=int)
            if not all(s.isdigit() for s in steps):
                raise ValueError(f"Potentially wrong path to velocity: '{vel_path}'.")
        self.steps = steps
        self._shape = self.f[self._velocity_path[0]][self.steps[0]].shape
        self.close()

    def open(self):
        self.f = h5py.File(self.filename, 'r')

    def close(self):
        self.f.close()

    def read_velocity(self, step: str = None):
        if step is None:
            step = self.steps[0]

        if self.f:
            for i, vel_path in enumerate(self._velocity_path):
                self.f[vel_path][step].read_direct(self.velocity[i])


class _H5Writer:
    def __init__(self, filename: str, mode: str = 'w'):
        self.filename = filename
        self.mode = mode

        self._create_file()

    def _create_file(self):
        if not self.filename.endswith('.h5'):
            self.filename += '.h5'

        with h5py.File(self.filename, self.mode) as f:
            f.require_group('position')
            f.require_group('trajectory')
            f.require_group('velocity')

            if 't' not in f.attrs:
                f.attrs['t'] = 0.0
            if 'step' not in f.attrs:
                f.attrs['step'] = 0

    def _read_data_info(self):
        if self.f:
            self._steps = sorted(list(self.f['position'].keys()), key=int)
            self._shape = self.f['position'][self._steps[0]].shape

    def open(self, mode: str = 'r+'):
        self.f = h5py.File(self.filename, mode=mode)

    def close(self):
        self.f.close()

    def write(self, t: float, step: int, position: np.ndarray, trajectory: np.ndarray, velocity: np.ndarray):
        self.open()
        self.f['position'].create_dataset(str(step), data=position)
        self.f['trajectory'].create_dataset(str(step), data=trajectory)
        self.f['velocity'].create_dataset(str(step), data=velocity)
        self.f.attrs['t'] = t
        self.f.attrs['step'] = step
        self.close()

    def restart(self):
        try:
            self.open('r')

            self._read_data_info()
            if len(self._steps) < 2:
                raise ValueError("Not enough steps to restart. At least two steps are required.")

            required_attrs = ['t', 'step']
            for attr in required_attrs:
                if attr not in self.f.attrs:
                    raise KeyError(f"Missing required attribute: {attr}")

            required_groups = ['position', 'trajectory', 'velocity']
            for group in required_groups:
                if group not in self.f:
                    raise KeyError(f"restart(): Missing required group: {group}")

            restart_data = {
                't': self.f.attrs['t'],
                'step': self.f.attrs['step'],
                'position': self.f['position'][self._steps[-1]],
                'trajectory': self.f['trajectory'][self._steps[-1]],
                'velocity': np.stack(
                    [
                        self.f['velocity'][self._steps[-1]],
                        self.f['velocity'][self._steps[-2]]
                    ], axis=-1)
            }
            return restart_data
        finally:
            self.close()


class LagrangianParticles:
    """
    This class is used to track the trajectory of passive particles in a fluid flow field. Note that
    this class is designed to perform Lagrangian Particle Tracking (LPT) ONLY in the post-processing stage.
    The velocity field is assumed to be in a periodic domain.

    The class uses a `RegularGridInterpolator` from `SciPy` to interpolate the velocity field data at the particle positions. The equation of motion is integrated using the 2nd-order Adams-Bashforth scheme.

    Parameters
    ----------
    - input_file: str
        The name of the input file containing the velocity field data.
    - grid: list
        The grid coordinates for each dimension of the velocity field.
    - grp_name: str, optional
        The name of the group containing `'fields'`. Default is ''.
    - n_particles: int, optional
        The number of particles to track. Default is 1.
    - t_end: float, optional
        The end time for the simulation. Default is 1.0.
    - dt: float, optional
        The time step size. Default is 0.01.
    - init_position: ndarray, optional
        The initial position of the particles. If None, the particles are randomly initialized.
    - check_interval: int, optional
        The period to check the runtime. Default is 10.
    - optimization: bool, optional
        Flag to enable optimization. Default is False.
    - interp_method: str, optional
        The interpolation method to use. Options include 'linear', 'cubic', 'quintic', 'pchip'. Default is 'cubic'.
    - output_file: str, optional
        The name of the output file to store the results. Default is None.
    - verbose: bool, optional
        Flag to enable verbose output. Default is False.
    """

    def __init__(
        self,
        input_file: str,
        data_path_velocity: list,
        grid: list[np.ndarray],
        output_file: str = 'particles',
        restart_file: str = 'particles',
        n_particles: int = 100,
        t_end: np.float64 = 1.0,
        dt: np.float64 = 0.01,
        interp_method: str = "cubic",
        write_first_step: bool = True,
        verbose: bool = False,
        check_interval: int = 10
    ):
        self._n_particles = n_particles
        self._t_end = t_end
        self._dt = dt
        self._interp_method = interp_method.casefold()
        self._verbose = verbose
        self._write_first_step = write_first_step

        self.t = 0.0
        self.step = 0

        self._n_stages = 2              # number of stages for the midpoint predictor-corrector scheme
        self._mpc_factor = [0.5, 1.0]   # factors for the midpoint predictor-corrector scheme

        self._set_coordinates(grid)

        self._init_hdf5_reader(input_file, data_path_velocity)
        self._init_hdf5_writer(output_file)

        self._initialize()
        self._set_interpolator()
        self._init_velocity()

        self.timer = _Timer(check_interval, verbose)

    def _set_coordinates(self, grid: list[np.ndarray]):
        self._grid = grid
        self._ndim = len(grid)
        self._domain = np.array([[g.min(), g.max()] for g in grid])
        self._L = np.array([g.max() - g.min() for g in grid])

    def _initialize(self):
        # the prefix 'p_' denotes particle-related variables
        self._p_pos_prev = np.zeros((self._ndim, self._n_particles), dtype=np.float64)
        self.p_position = np.zeros((self._ndim, self._n_particles), dtype=np.float64)
        self.p_trajectory = np.zeros((self._ndim, self._n_particles), dtype=np.float64)
        self.p_velocity = np.zeros((self._ndim, self._n_particles), dtype=np.float64)

        for i in range(self._ndim):
            self.p_position[i] = np.random.uniform(self._domain[i, 0], self._domain[i, 1], self._n_particles)
        self._p_pos_prev[:] = self.p_position

    def _init_hdf5_reader(self, filename: str, velocity_path: list[str]):
        if len(velocity_path) != self._ndim:
            raise ValueError(
                f"Number of velocity paths ({len(velocity_path)}) must match the number of dimensions ({self._ndim})."
            )
        self.h5reader = _H5Reader(filename, velocity_path)

    def _init_hdf5_writer(self, filename: str, mode: str = 'w'):
        self.h5writer = _H5Writer(filename, mode)

    def _set_interpolator(self):
        if self._interp_method not in sp.interpolate.RegularGridInterpolator._ALL_METHODS:
            raise ValueError(f"Invalid interpolation method: {self._interp_method}. Available methods include: {sp.interpolate.RegularGridInterpolator._ALL_METHODS}")

        shape = [len(d) for d in self._grid]
        dummy_velocity = [np.zeros(shape) for _ in range(self._ndim)]

        self.interpolator = [
            sp.interpolate.RegularGridInterpolator(
                self._grid,
                velocity,
                method=self._interp_method
            ) for velocity in dummy_velocity
        ]

    def _init_velocity(self):
        self.h5reader.open()
        self.h5reader.read_velocity()
        self.h5reader.close()
        self._update_flow_velocity(self.h5reader.velocity)
        self._update_particle_velocity()

    def _update_flow_velocity(self, flow_velocity: np.ndarray):
        for i in range(self._ndim):
            self.interpolator[i].values[:] = flow_velocity[i]

    def _update_particle_velocity(self):
        self.p_velocity[:] = np.array([interp(self.p_position.T) for interp in self.interpolator])

    def _update_previous_position(self):
        self._p_pos_prev[:] = self.p_position

    def init_position(self, position: np.ndarray):
        if position.shape != (self._ndim, self._n_particles):
            raise ValueError(f"Position array shape {position.shape} must match ({self._ndim}, {self._n_particles}).")
        self.p_position[:] = position
        self._init_velocity()

    def restart(self):
        restart_data = self.h5writer.restart()
        if np.close(restart_data['t'], self._t_end, atol=1e-8):
            raise ValueError("Restart time coincides with end time.")

        self.t = restart_data['t']
        self.step = restart_data['step']
        self.p_position[:] = restart_data['position']
        self.p_trajectory[:] = restart_data['trajectory']
        self.p_velocity[:] = restart_data['velocity']

    def write(self):
        self.h5writer.write(self.t, self.step, self.p_position, self.p_trajectory, self.p_velocity)

    @staticmethod
    @nb.njit
    def _midpoint_predictor_corrector(
        pos_curr: np.ndarray,
        pos_prev: np.ndarray,
        vel: np.ndarray,
        traj: np.ndarray,
        a: float, dt: float, stage: int
    ):
        shape = pos_curr.shape

        for i in range(shape[0]):
            for j in range(shape[1]):
                dx = a * dt * vel[i, j]
                pos_curr[i, j] = pos_prev[i, j] + dx

                if stage == 1:
                    traj[i, j] += dx

    @staticmethod
    @nb.njit
    def _periodic_bc(pos: np.ndarray, domain: list[list], L: list):
        shape = pos.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                pos[i, j] = (pos[i, j] - domain[i][0]) % L[i] + domain[i][0]

    def stepping(self):
        """
        Time stepping loop for the Lagrangian Particle Tracking (LPT). In each step, the motion of the particles is integrated using a midpoint predictor-corrector scheme.
        """

        self.timer.start()

        if self._write_first_step:
            self.write()

        self.h5reader.open()
        for tstep in self.h5reader.steps:
            self.step += 1
            self.t += self._dt

            self.h5reader.read_velocity(tstep)
            self._update_flow_velocity(self.h5reader.velocity)
            self._update_previous_position()
            for stage in range(self._n_stages):
                a = self._mpc_factor[stage]
                self._update_particle_velocity()
                self._midpoint_predictor_corrector(
                    self.p_position,
                    self._p_pos_prev,
                    self.p_velocity,
                    self.p_trajectory,
                    a, self._dt, stage
                )
                self._periodic_bc(self.p_position, self._domain, self._L)

            self.write()
            self.timer(self.step, self.t)
        self.h5reader.close()

        self.timer.stop()


def compute_msd(self):
    """
    Compute the Mean Squared Displacement (MSD) using either time lag or absolute time.

    Parameters
    ----------
    - trajectory: ndarray
        The trajectory array with shape (n_dims, n_particles, n_samples).
    - time: ndarray
        The time array with shape (n_samples,).

    Returns
    ---------
    - msd (ndarray):
        The computed MSD values.
    - time_lags (ndarray):
        The corresponding time lags (or absolute times) for each MSD value.
    """

    self.h5writer.open()
    steps = sorted(list(self.h5writer.f['trajectory'].keys()), key=int)
    n_steps = len(steps)

    trajectory = np.zeros((self._ndim, self._n_particles, n_steps))
    for i, step in enumerate(steps):
        trajectory[..., i] = self.h5writer.f['trajectory'][step]

    msd = np.zeros(n_steps)
    self._calculate_displacement_sum(msd, trajectory)

    self.h5writer.f.create_dataset('msd', data=msd)
    self.h5writer.close()


@nb.njit
def _calculate_displacement_sum(msd, trajectory):
    """
    Compute the Mean Squared Displacement (MSD) using time lags.

    Parameters
    ----------
    - msd: ndarray
        Pre-allocated array to store the MSD values.
    - trajectory: ndarray
        The trajectory array with shape (n_dims, n_particles, n_samples).
    """

    n_dims, n_particles, n_samples = trajectory.shape

    for lag in range(1, n_samples):
        sum_squared_displacements = 0.0
        time_lag = n_samples - lag

        for dim in range(n_dims):
            for particle in range(n_particles):
                for t in range(time_lag):
                    displacement = (
                        trajectory[dim, particle, t + lag] - trajectory[dim, particle, t]
                    )
                    sum_squared_displacements += displacement**2

        msd[lag] = sum_squared_displacements / (n_particles * time_lag)
