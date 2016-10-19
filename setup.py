from setuptools import setup

setup(
    name='skyarea',
    packages=['sky_area'],
    scripts=['bin/make_search_map', 'bin/process_areas', 'bin/run_sky_area'],
    version='0.2.1',
    description='Compute credible regions on the sky from RA-DEC MCMC samples',
    author='Will M. Farr',
    author_email='will.farr@ligo.org',
    url='http://farr.github.io/skyarea/',
    license='MIT',
    keywords='MCMC credible regions skymap LIGO',
    install_requires=['numpy', 'matplotlib', 'scipy', 'healpy', 'glue'],
    classifiers=['Development Status :: 5 - Production/Stable',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Programming Language :: Python :: 2.7',
                 'Topic :: Scientific/Engineering :: Astronomy',
                 'Topic :: Scientific/Engineering :: Visualization']
)
