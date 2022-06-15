Python API reference
====================

User API reference
------------------

This is the complete reference of the Python API for the :code:`py4dgeo` package.
It focuses on those aspects relevant to end users that are not interested in algorithm development.

.. autoclass:: py4dgeo.Epoch
    :members:

.. autofunction:: py4dgeo.read_from_las

.. autofunction:: py4dgeo.read_from_xyz

.. autofunction:: py4dgeo.save_epoch

.. autofunction:: py4dgeo.load_epoch

.. autoclass:: py4dgeo.M3C2
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: py4dgeo.CloudCompareM3C2
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: py4dgeo.SpatiotemporalAnalysis
    :members:

.. autoclass:: py4dgeo.RegionGrowingAlgorithm
    :members:
    :inherited-members:
    :show-inheritance:

.. autofunction:: py4dgeo.regular_corepoint_grid

.. autofunction:: py4dgeo.set_py4dgeo_logfile

.. autofunction:: py4dgeo.set_memory_policy

.. autoclass:: py4dgeo.MemoryPolicy

.. autofunction:: py4dgeo.get_num_threads

.. autofunction:: py4dgeo.set_num_threads

Developer API reference
-----------------------

.. autofunction:: py4dgeo.epoch.as_epoch

.. autofunction:: py4dgeo.epoch.normalize_timestamp

.. autoclass:: py4dgeo.m3c2.M3C2LikeAlgorithm
    :members:

.. autoclass:: py4dgeo.fallback.PythonFallbackM3C2
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: py4dgeo.segmentation.RegionGrowingAlgorithmBase
    :members:

.. autoclass:: py4dgeo.segmentation.RegionGrowingSeed
    :members:

.. autoclass:: py4dgeo.segmentation.ObjectByChange
    :members:

.. autofunction:: py4dgeo.segmentation.check_epoch_timestamp

.. autoclass:: py4dgeo.util.Py4DGeoError

.. autofunction:: py4dgeo.find_file

.. autofunction:: py4dgeo.util.as_double_precision

.. autofunction:: py4dgeo.util.as_single_precision

.. autofunction:: py4dgeo.util.make_contiguous

.. autofunction:: py4dgeo.util.memory_policy_is_minimum

.. autofunction:: py4dgeo.util.append_file_extension

.. autofunction:: py4dgeo.util.is_iterable

.. autoclass:: py4dgeo.zipfile.UpdateableZipFile

.. autofunction:: py4dgeo.logger.create_default_logger

.. autofunction:: py4dgeo.logger.logger_context
