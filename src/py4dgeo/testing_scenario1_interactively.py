import py4dgeo
import random

import numpy as np


py4dgeo.util.ensure_test_data_availability()

random.seed(10)
np.random.seed(10)

epoch0, epoch1 = py4dgeo.read_from_xyz(
    "plane_horizontal_t1.xyz", "plane_horizontal_t2.xyz"
)

# ***************
# Scenario 1

random.seed(10)
np.random.seed(10)

Alg = py4dgeo.PB_M3C2()


# Alg.generate_extended_labels_interactively(
#     epoch0=epoch0, epoch1=epoch1,
#     **{"Transform Segmentation__output_file_name": "seg_test_out_alex", "get_pipeline_options": True})

# Alg.generate_extended_labels_interactively(epoch0=epoch0, epoch1=epoch1, bla=False)

# Alg.generate_extended_labels_interactively(get_pipeline_options=True)

# ----------------------------

segments, extended_y = Alg.generate_extended_labels_interactively(
    epoch0=epoch0, epoch1=epoch1
)
Alg.training(segments=segments, extended_y=extended_y)

# ----------------------------

print(Alg.predict(epoch0=epoch0, epoch1=epoch1))
print(Alg.compute_distances(epoch0=epoch0, epoch1=epoch1))
