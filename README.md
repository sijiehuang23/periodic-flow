# <tt>periodicflow</tt>: A simple Fourier-Galerkin pseudo-spectral Navier-Stokes solver

## Introduction

<tt>periodicflow</tt> is a simple code designed to perform __direct numerical simulation__ (__DNS__) in periodic domains, which indicates that flow being statistically homogeneous in all spatial directions (such as isotropic turbulence). It solves for the incompressible Navier-Stokes equations using _Fourier Galerkin pseudo-spectral_ method [[1]](#ref1) in both two and three dimensions. The backbone of the code builds upon the <tt>Python</tt> package [<tt>shenfun</tt>](https://github.com/spectralDNS/shenfun) [[2]](#ref2), which takes care most of the heavy-lifting jobs (such as forward and inverse Fourier transform, MPI parallelization, dealiasing, I/O, etc). The code uses <tt>numba</tt>-optimized functions to speedup simulations. 


## Governing equations and numerical methods 

The code solves the incompressible fluctuating Navier-Stokes equations in both two and three dimensions. A noise term is included to model mesoscopic fluctuations, such as thermal fluctuations. Additionally, an external forcing term is added to sustain the flow in scenarios such as statistically steady turbulent flows.

The following time integration schemes are provided for now:
1. An explicit and an implicit predictor-corrector schemes designed for Langevin-type of equations [[3]](#ref3)
2. Second and third-order TVD RK schemes by Gottlieb & Shu (1998) [[4]](#ref4). The 3rd-order version is modified in [[3]](#ref3) to account for stochastic terms.
3. A low-storage 4th-order RK scheme provided by Bogey & Bailly (2004) [[5]](#ref5)


## Installation and usage 

1. Create a `mamba environment
    ```bash
    mamba create -n <env_name>
    ```
2. The only direct dependency of the code is the [shenfun](https://github.com/spectralDNS/shenfun) package. One can install via `mamba` by 
    ```bash
    mamba activate <env_name>
    mamba install -c conda-forge shenfun # can also install using conda
    ```
    Dependencies of `shenfun` should be installed during the processes by `mamba` automatically. 
3. To install the code, 
    ```bash
    cd /path/to/periodic-flow/
    pip install -e . # Editable installation; remove -e for non-editable one
    ```
4. To run a simulation in parallel, do the following 
    ```bash
    mpirun -np <number_of_processes> python <script_name>.py
    ```

### Uninstall

To uninstall the package, simply do 
```bash
mamba activate <env_name>
pip uninstall periodicflow
```

**Note**: If the package was installed in editable mode, one might have to remove a `periodicflow.egg-info/` folder manually under `src` as well, since `pip uninstall` does not automatically remove this folder. 



## References

1. <a id="ref1"></a> Canuto, C., Quarteroni, A., Hussaini, M. Y., Zang, T. A. 2007 Spectral Methods: Evolution to Complex Geometries and Applications to Fluid Dynamics. [_Springer_](https://link.springer.com/book/10.1007/978-3-540-30728-0).
2. <a id="ref2"></a> Mortensen, M. 2018 Shenfun: High performance spectral Galerkin computing platform. [_J. Open Source Softw._ **3**(31), 1071.](https://doi.org/10.21105/joss.01071)
3. <a id="ref3"></a> Delong, S., Griffith, B. E., Vanden-Eijnden, E., & Donev, A. 2013 Temporal integrators for fluctuating hydrodynamics. [_Phys. Rev. E_ **87**(3), 033302.](https://doi.org/10.1103/PhysRevE.87.033302)
4. <a id="ref4"></a> Gottlieb, S., & Shu, C.-W. 1998 Total variation diminishing Runge-Kutta schemes. [*Math. Comput.* **67**(221), 73-85.](https://doi.org/10.1090/S0025-5718-98-00913-2)
5. <a id="ref5"></a> Bogey, C. & Bailly, C. 2004 A family of low dispersive and low dissipative explicit schemes for flow and noise computations. [_J. Comput. Phys._ **194**, 194-214.](https://doi.org/10.1016/j.jcp.2003.09.003)
