Python API reference
====================

User API reference
------------------

This is the complete reference of the Python API for the :code:`py4dgeo` package.
It focuses on those aspects relevant to end users that are not interested in algorithm development.

.. autoclass:: py4dgeo.M3C2
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: py4dgeo.Epoch
    :members:

.. autofunction:: py4dgeo.set_memory_policy

.. autoclass:: py4dgeo.MemoryPolicy

.. autofunction:: py4dgeo.find_file


Developer API reference
-----------------------

.. autoclass:: py4dgeo.m3c2.M3C2LikeAlgorithm
    :members:

.. autoclass:: py4dgeo.fallback.PythonFallbackM3C2
    :members:
    :inherited-members:
    :show-inheritance:

.. autofunction:: py4dgeo.util.as_double_precision

.. autofunction:: py4dgeo.util.make_contiguous
