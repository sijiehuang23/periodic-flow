import numpy as np
import mpi4py.MPI as mpi

try:
    import shenfun as sf
except ImportError:
    raise ImportError("This module requires shenfun to be installed.")


class FourierSpace:
    def __init__(
        self,
        comm: mpi.Comm,
        mpi_rank: int,
        N: list | tuple,
        domain: float | list | tuple = 2 * np.pi,
        dealias: str = '3/2',
        fft_plan: str = 'FFTW_MEASURE',
        decomposition: str = 'slab'
    ):
        self.comm = comm
        self.mpi_rank = mpi_rank
        self.N = N
        self.dim = len(self.N)

        if isinstance(domain, float):
            self.domain = [(0, domain) for _ in range(self.dim)]
        elif isinstance(domain, (list, tuple)):
            if all(isinstance(i, (list, tuple)) for i in domain):
                self.domain = domain
            elif all(isinstance(i, float) for i in domain):
                self.domain = [domain for _ in range(self.dim)]
            else:
                raise ValueError("Invalid domain.")
        else:
            raise ValueError("Input domain must be either float, list or tuple.")

        dtype = [np.float64 if i == self.dim - 1 else np.complex128 for i in range(self.dim)]
        F = [sf.FunctionSpace(n, 'F', domain=self.domain[i], dtype=dtype[i]) for i, n in enumerate(self.N)]
        S = sf.TensorProductSpace(self.comm, F, dtype=np.float64, slab=(decomposition == 'slab'), planner_effort=fft_plan)
        k = S.local_wavenumbers(scaled=True, broadcast=True)

        self.S = S
        self.V = sf.VectorSpace(self.S)
        self.T = sf.CompositeSpace([self.S] * self.dim**2)
        self.W = self.S if self.dim == 2 else self.V

        self.x = S.local_mesh()
        self.k = np.array(k)
        self.k2 = sum(ki**2 for ki in k)
        self.k_over_k2 = np.array([ki / np.where(self.k2 == 0, 1, self.k2) for ki in k])

        dealias_options = {
            '3/2': {"padding_factor": 1.5, "dealias_direct": False},
            'pad': {"padding_factor": 1.5, "dealias_direct": False},
            '2/3': {"padding_factor": 1.0, "dealias_direct": True},
            'truncate': {"padding_factor": 1.0, "dealias_direct": True},
        }
        dealias_option = dealias_options.get(dealias)
        if dealias_option is None:
            raise ValueError(f"Invalid dealiasing rule: {dealias}.")

        self.Sp = self.S.get_dealiased(**dealias_option)
        self.Vp = sf.VectorSpace(self.Sp)
        self.Tp = sf.CompositeSpace([self.Sp] * self.dim**2)
        self.Wp = self.Sp if self.dim == 2 else self.Vp
