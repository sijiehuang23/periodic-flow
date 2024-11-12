# A simple Fourier-Galerkin pseudo-spectral Navier-Stokes solver

## Introduction

This code is designed to perform __direct numerical simulation__ (__DNS__) in periodic domains (homogeneous in all spatial directions). It solves for the incompressible Navier-Stokes equations using _Fourier Galerkin pseudo-spectral_ method [[1]](#ref1). The backbone of the code builds upon the <tt>Python</tt> package [shenfun](https://github.com/spectralDNS/shenfun), which takes care most of the heavy-lifting jobs (such as forward and inverse Fourier transform, parallelization, dealiasing, etc). The code uses `numba`-optimized functions to speedup simulation. 

The code solves the incompressible fluctuating Navier-Stokes equations in both two and three dimensions. A noise term is included to model mesoscopic fluctuations, such as thermal fluctuations. Additionally, an external forcing term is added to sustain the flow in scenarios such as statistically steady turbulent flows.

The following time integration schemes are provided for now:
1. An explicit and an implicit predictor-corrector schemes designed for Langevin-type of equations [[2]](#ref2)
2. Second and third-order TVD RK schemes by Gottlieb & Shu (1998) [[3]](#ref3). The 3rd-order version is modified in [[2]](#ref2) to account for stochastic terms.
3. A low-storage 4th-order RK scheme provided by Bogey & Bailly (2004) [[4]](#ref4)


## Installation 

The only direct dependency of the code is the [shenfun](https://github.com/spectralDNS/shenfun) package. One can install via `conda` by 
```bash
conda install -c conda-forge shenfun
```
or `mamba`
```bash
mamba install -c conda-forge shenfun 
```
Dependencies of `shenfun` will be installed during the processes. 


## References

1. <a id="ref1"></a> Canuto, C., Quarteroni, A., Hussaini, M. Y., Zang, T. A. (2007) Spectral Methods: Evolution to Complex Geometries and Applications to Fluid Dynamics. [*Springer Berlin*](https://link.springer.com/book/10.1007/978-3-540-30728-0).
2. <a id="ref2"></a> Delong, S., Griffith, B. E., Vanden-Eijnden, E., & Donev, A. (2013). **Temporal integrators for fluctuating hydrodynamics**. [*Physical Review E*, **87**(3), 033302.](https://doi.org/10.1103/PhysRevE.87.033302)
3. <a id="ref3"></a> Gottlieb, S., & Shu, C.-W. 1998 **Total variation diminishing Runge-Kutta schemes**. [*Math. Comput.*, **67**(221), 73--85.](https://doi.org/10.1090/S0025-5718-98-00913-2)
4. <a id="ref4"></a> Bogey, C. & Bailly, C. 2004 **A family of low dispersive and low dissipative explicit schemes for flow and noise computations**. [_J. Comput. Phys._, **194**, 194--214.](https://www.sciencedirect.com/science/article/pii/S0021999103004662)
