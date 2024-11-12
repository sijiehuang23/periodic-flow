from pathlib import Path
import numpy as np
import h5py
from .utils import periodic_bc

try:
    import shenfun as sf
except ImportError:
    raise ImportError('shenfun is required for this module')


class HDF5Writer:
    def __init__(
            self,
            file_name: str,
            solution: dict = {},
            time_points: list = [0.0, 1.0]
    ):
        self.file_name = file_name
        self.solution = solution
        self.time_points = time_points

        self.solution_file = sf.ShenfunFile(
            self.file_name,
            self.solution['space'],
            mode='w'
        )

    def write(self, step: int):
        self.solution_file.write(step, self.solution['data'], as_scalar=True)

    def close(self):
        if self.solution_file.f:
            self.solution_file.close()

    def sort_dateset(self, periodic: bool = True, x_end: float = 2 * np.pi) -> None:
        """
        Reads and processes data from `spectralDNS` and stores it in an HDF5 file.

        Parameters
        ----------
        input_file : str
            Name of the input HDF5 file.
        output_file : str
            Name of the output HDF5 file.
        periodic : bool, optional
            Whether to enforce periodic boundary conditions.
        x_end : float, optional
            The end value for periodicity, default is 2π.
        """

        input_path = Path(self.file_name).with_suffix(".h5")
        temp_path = Path(str(input_path).replace(".h5", "_temp.h5"))

        if not input_path.exists():
            raise FileNotFoundError(f"Solution file '{input_path}' does not exist.")

        try:
            with h5py.File(input_path, 'r') as fr, h5py.File(temp_path, 'w') as fw:
                var_names = [vn for vn in fr]

                dimension = next(iter(fr[var_names[0]]))
                ndims = {"3d": 3, "2d": 2}.get(dimension.lower(), 0)
                steps = sorted(fr[var_names[0]][dimension], key=int)

                for key, label in zip(["x0", "x1", "x2"][:ndims], ["x", "y", "z"][:ndims]):
                    coord = fr[f"{var_names[0]}/mesh/{key}"][:]
                    if periodic:
                        coord = np.append(coord, x_end)
                    fw.create_dataset(label, data=coord)

                fw.create_dataset("t", data=np.linspace(self.time_points[0], self.time_points[-1], len(steps)))

                n_digits = len(str(max(map(int, steps))))

                for var in var_names:
                    for step in steps:
                        data = fr[f"{var}/{dimension}/{step}"][:]
                        if periodic:
                            data = periodic_bc(data)
                        step_name = str(step).zfill(n_digits)
                        fw.create_dataset(f"{var}/{step_name}", data=data)

            input_path.unlink()
            temp_path.rename(input_path)

        except Exception as e:
            raise RuntimeError(f"Error processing {input_path}: {e}")
