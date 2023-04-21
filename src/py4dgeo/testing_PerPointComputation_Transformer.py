import py4dgeo
import random

import numpy as np

py4dgeo.util.ensure_test_data_availability()

random.seed(10)
np.random.seed(10)

epoch0, epoch1 = py4dgeo.read_from_xyz(
    "plane_horizontal_t1.xyz", "plane_horizontal_t2.xyz"
)

# apply the pipeline
X0 = np.hstack((epoch0.cloud[:, :], np.zeros((epoch0.cloud.shape[0], 1))))
X1 = np.hstack((epoch1.cloud[:, :], np.ones((epoch1.cloud.shape[0], 1))))
X = np.vstack((X0, X1))
# X = X0


per_point_computation = py4dgeo.PerPointComputation()
per_point_computation.fit(X)
out = per_point_computation.transform(X)
print(out)
