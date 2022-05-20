Callback reference
==================

As described in the custumization tutorial, :code:`py4dgeo` uses a callback software architecture
to allow to flexibly change components of the core algorithm while maintaining its performance.
Callbacks can be implemented in Python (for rapid prototyping) or C++ (for performance). In this
section, we summarize the available types of callbacks and their available implementations.

Distance + Uncertainty Calculation
----------------------------------

This callback is responsible for calculating the distance measure and its uncertainty at one core point.
The C++ signature for this callback is the following:

.. doxygentypedef:: py4dgeo::DistanceUncertaintyCalculationCallback

The default implementation calculates the mean and standard deviation:

.. doxygenfunction:: py4dgeo::mean_stddev_distance
.. autofunction:: py4dgeo.fallback.mean_stddev_distance

Alternative, a distance measure based on the median and interquartile range is available:

.. doxygenfunction:: py4dgeo::median_iqr_distance
.. autofunction:: py4dgeo.fallback.median_iqr_distance

Working Set Finder
------------------

This callback determines which points from a given epoch that are located around a given corepoint
should be taken into consideration by the M3C2 algorithm.
The C++ signature is the following:

.. doxygentypedef:: py4dgeo::WorkingSetFinderCallback

The available implementations perform a radius and a cylinder search:

.. doxygenfunction:: py4dgeo::radius_workingset_finder
.. autofunction:: py4dgeo.fallback.radius_workingset_finder

.. doxygenfunction:: py4dgeo::cylinder_workingset_finder
.. autofunction:: py4dgeo.fallback.cylinder_workingset_finder
