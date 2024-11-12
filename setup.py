from setuptools import setup, find_packages


setup(
    name='periodicflow',
    version='0.1',
    description="A simple pseudo-spectral Fourier-Galerkin solver for incompressible Navier-Stokes equations",
    package_dir={'': 'src'},
    packages=find_packages(where='src')
)
