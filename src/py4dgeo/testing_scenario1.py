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

alg = py4dgeo.PB_M3C2()

# alg.export_segmented_point_cloud_and_segments(
#     epoch0=epoch0, epoch1=epoch1,
#     **{"Transform Segmentation__output_file_name": "seg_test_out_alex", "c": True})

(
    x_y_z_id_epoch0,
    x_y_z_id_epoch1,
    extracted_segments,
) = alg.export_segmented_point_cloud_and_segments(epoch0=epoch0, epoch1=epoch1)

extended_y = py4dgeo.generate_random_extended_y(
    extracted_segments, extended_y_file_name="extended_y.csv"
)

# alg.training(extracted_segments, extended_y)
alg.training(
    extracted_segments_file_name="extracted_segments.seg",
    extended_y_file_name="extended_y.csv",
)


print(alg.predict(epoch0=epoch0, epoch1=epoch1))
print(alg.compute_distances(epoch0=epoch0, epoch1=epoch1))
