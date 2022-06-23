Frequently Asked Questions
==========================

py4dgeo requires more RAM than e.g. CloudCompare/I run out of memory processing a pointcloud that CloudCompare can process on the same computer
-----------------------------------------------------------------------------------------------------------------------------------------------

This might be related to a deliberate choice in the design of :code:`py4dgeo`:
Point clouds are stored as double precision values (:code:`np.float64` or C++'s :code:`double`),
while many other software packages that process point clouds (e.g. CloudCompare) go
for a mixed precision approach, where the point cloud is stored in single precision
and computational results (e.g. M3C2 distances) are represented in double precision.
In order to store point clouds in single precision without significant loss of precision,
they need to be correctly shifted close to the origin of the space and the resulting
offset needs to be taken into account in a lot of computations. With :code:`py4dgeo`
being an open system for users to develop their own algorithms, we felt that the risk
of caveats related to this transformation process is quite high and therefore decided
to choose the very robust (but memory-inefficient) method of storing point clouds in
global coordinates as double precision values. Other than what was just explained,
the design of the :code:`py4dgeo` library aims for maximum RAM efficiency.
