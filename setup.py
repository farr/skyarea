from setuptools import setup
from sky_area import __version__ as version

setup(
    name='skyarea',
    packages=['sky_area'],
    scripts=['bin/run_sky_area'],
    version=version,
    description='Compute credible regions on the sky from RA-DEC MCMC samples',
    author='Will M. Farr',
    author_email='will.farr@ligo.org',
    url='http://farr.github.io/skyarea/',
    license='MIT',
    keywords='MCMC credible regions skymap LIGO',
    install_requires=['astropy', 'numpy', 'scipy', 'healpy', 'six'],
    classifiers=['Development Status :: 5 - Production/Stable',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 2.7',
                 'Topic :: Scientific/Engineering :: Astronomy',
                 'Topic :: Scientific/Engineering :: Visualization']
)
