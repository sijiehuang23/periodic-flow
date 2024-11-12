# A simple Fourier-Galerkin pseudo-spectral Navier-Stokes solver


This code is designed to perform __direct numerical simulation__ (__DNS__) in periodic domains (homogeneous in all spatial directions). It solves for the incompressible Navier-Stokes equations using _Fourier Galerkin pseudo-spectral_ method [[1]](#ref1). The backbone of the code builds upon the <tt>Python</tt> package [shenfun](https://github.com/spectralDNS/shenfun), which takes care most of the heavy-lifting jobs (such as forward and inverse Fourier transform, parallelization, dealiasing, etc). The code uses `numba`-optimized functions to speedup simulation. 


## Governing equations and numerical methods

The incompressible Navier-Stokes equations under consideration are 
$$
\partial_t\bm{u} = \bm{u}\times\bm{\omega} -\bm{\nabla}P + \nu\Delta\bm{u} + \bm{\eta} + \bm{f},\tag{1}
$$
where $\bm{u}(\bm{x})$ is the velocity field subject to the incompressibility condition ($\bm{\nabla}\cdot\bm{u}\equiv0$), $\bm{\omega} = \bm{\nabla}\times\bm{u}$ is the vorticity field, $P = p + |\bm{u}|^2/2$ is the modified pressure field, $\nu$ is the kinematic viscosity assumed to be constant, $\bm{\eta}$ is a stochastic field modeling mesoscopic random fluctuations (such as thermal fluctuation in fluctuating hydrodynamics), and $\bm{f}$ is an external forcing to sustain the flow if needed (for example, sustaining turbulent flows). 

### Pseudo-spectral Fourier-Galerkin method

The equations are solved in Fourier space, which are written as 
$$
\left(d_t + \nu k^2\right)\hat{\bm{u}}(\bm{k}) = \bm{\mathcal{P}}\left(\widehat{\bm{u}\times\bm{\omega}} + \hat{\bm{\eta}} + \hat{\bm{f}}\right),\tag{2}
$$
where $\mathcal{P}\equiv\bm{I} - \bm{k}\otimes\bm{k}/k^2$ is the __Leray projection operator__ projecting vectors to the solenoidal subspace, and the nonlinear term $\bm{u}\times\bm{\omega}$ is evaluated in physical space and then transformed to the Fourier space (hence, pseudo-spectral method). This step requires dealiasing. [shenfun](https://github.com/spectralDNS/shenfun) provides both zero-padding (3/2 method) and truncating (2/3 method) methods. 


### Time integration

The following time integration schemes are provided for now:

1. An implicit predictor-corrector scheme designed for equations involving stochastic terms [[2]](#ref2)
2. Second and third-order TVD RK schemes by Gottlieb & Shu (1998) [[3]](#ref3). The 3rd-order version is modified in [[2]](#ref2) to account for stochastic term.
3. A low-storage 4th-order RK scheme provided by Bogey & Bailly (2004) [[4]](#ref4)


## Fluctuations




## References

1. <a id="ref4"></a> Canuto, C., Quarteroni, A., Hussaini, M. Y., Zang, T. A. (2007) Spectral Methods: Evolution to Complex Geometries and Applications to Fluid Dynamics. [*Springer Berlin*](https://link.springer.com/book/10.1007/978-3-540-30728-0).
2. <a id="ref1"></a> Delong, S., Griffith, B. E., Vanden-Eijnden, E., & Donev, A. (2013). **Temporal integrators for fluctuating hydrodynamics**. [*Physical Review E*, **87**(3), 033302.](https://doi.org/10.1103/PhysRevE.87.033302)
3. <a id="ref2"></a> Gottlieb, S., & Shu, C.-W. 1998 **Total variation diminishing Runge-Kutta schemes**. [*Math. Comput.*, **67**(221), 73--85.](https://doi.org/10.1090/S0025-5718-98-00913-2)
4. <a id="ref3"></a> Bogey, C. & Bailly, C. 2004 **A family of low dispersive and low dissipative explicit schemes for flow and noise computations**. [_J. Comput. Phys._, **194**, 194--214.](https://www.sciencedirect.com/science/article/pii/S0021999103004662)
# periodic-flow
