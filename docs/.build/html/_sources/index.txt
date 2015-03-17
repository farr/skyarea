.. Sky Area documentation master file, created by
   sphinx-quickstart on Thu Feb  6 11:26:14 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Sky Area's documentation!
====================================

The ``sky_area`` package provides utilities to turn samples from
probability distributions on the sky (i.e. samples of RA and DEC) into
sky maps, computing credible areas of the distribution, calculating
the minimum searched area following a greedy algorithm to find an
object, and producing `Healpix-pixellated
<http://healpix.jpl.nasa.gov>`_ maps that can be used to optimise a
search with a known telescope beam.

There are also executable codes, that rely on the `LALInference
<https://www.lsc-group.phys.uwm.edu/daswg/projects/lalsuite.html>`_
libraries from the LIGO Scientific Collaboration, for producing
various skymaps and credible regions in FITS format.  

The algorithm used to turn discrete samples into a probability
distribution on the sky is an improved version of the clustering
algorithm `X-means <http://www.cs.cmu.edu/~dpelleg/kmeans.html>`_ that
provides more flexibility in the shape of each cluster.  The code
works hard to ensure that the quoted credible areas are `unbiased
<https://dcc.ligo.org/LIGO-P1400054>`_, so the X% credible area will,
on average, enclose X% of the probability mass.  

You may want to:

* Compute the credible areas or the area searched under a greedy
  algorithm for a distribution on the sky represented by discrete
  samples.  Use the
  :class:`sky_area.sky_area_clustering.ClusteredSkyKDEPosterior` class.

* Automatically produce the above from the output of a LALInference
  run.  Use the executable program ``run_sky_area.py``

* Produce a Healpix map that ranks pixels on the sky for a search
  following the posterior denisty with a telescope having a known beam
  size.  Use the :func:`sky_area.search.search_map` function or, from
  the command-line, the ``make_search_map.py`` executable.

* Collate a bunch of sky maps, searched areas, and credible areas to
  produce a cumulative distribution of searched/credible areas from a
  combined data set of posteriors, as in `Singer, et al
  <http://arxiv.org/abs/1404.5623>`_.  Use the ``process_areas.py``
  executable.

* Compute, as a function of position on the sky, the constraints on
  the distance of the source.  Use the :class:`sky_area.sky_area_clustering.Clustered3DKDEPosterior`.

The :mod:`sky_area.sky_area_clustering` Module
----------------------------------------------

.. automodule:: sky_area.sky_area_clustering
   :members:
   :undoc-members:
   :show-inheritance:

The :mod:`sky_area.search` Module
---------------------------------

.. automodule:: sky_area.search
   :members:
   :undoc-members:
   :show-inheritance:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

