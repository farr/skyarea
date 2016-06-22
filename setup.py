from setuptools import setup

setup(
    name='skyarea',
    packages=['sky_area'],
    scripts=['bin/make_search_map.py', 'bin/process_areas.py', 'bin/run_sky_area.py'],
    version='0.1',
    description='Code for computing credible regions on the sky from RA-DEC MCMC samples.',
    author='Will M. Farr',
    author_email='will.farr@ligo.org',
    url='http://farr.github.io/skyarea/',
    license='MIT',
    keywords='MCMC credible regions skymap LIGO',
    classifiers=['Development Status :: 5 - Production/Stable',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Topic :: Scientific/Engineering :: Astronomy',
                 'Topic :: Scientific/Engineering :: Visualization']
)
