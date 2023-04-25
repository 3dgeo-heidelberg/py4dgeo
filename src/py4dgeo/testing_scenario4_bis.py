import py4dgeo
import numpy as np
import pickle

py4dgeo.set_interactive_backend("vtk")

py4dgeo.ensure_test_data_availability()

epoch0, epoch1 = py4dgeo.read_from_xyz(
    "plane_horizontal_t1.xyz", "plane_horizontal_t2.xyz"
)


alg4_no_reconstruction = py4dgeo.PB_M3C2_time_series_no_reconstruction()

_0, _1, epoch0_segments = super(
    py4dgeo.PB_M3C2_time_series_no_reconstruction, alg4_no_reconstruction
).export_segmented_point_cloud_and_segments(
    epoch0=epoch0,
    epoch1=None,
    x_y_z_id_epoch0_file_name=None,
    x_y_z_id_epoch1_file_name=None,
    extracted_segments_file_name=None,
)


_0, _1, extracted_segments = super(
    py4dgeo.PB_M3C2_time_series_no_reconstruction, alg4_no_reconstruction
).export_segmented_point_cloud_and_segments(
    epoch0=epoch0,
    epoch1=epoch1,
    x_y_z_id_epoch0_file_name=None,
    x_y_z_id_epoch1_file_name=None,
    extracted_segments_file_name=None,
)

# (
#     _0,
#     _1,
#     extracted_segments,
# ) = alg4_no_reconstruction.export_segmented_point_cloud_and_segments(
#     epoch0_segments=epoch0_segments,
#     epoch1_xyz=epoch1,
#     x_y_z_id_epoch1_file_name=None,
#     extracted_segments_file_name=None,
# )
#
# py4dgeo.Viewer.segments_visualizer(X=extracted_segments)

extended_y = py4dgeo.generate_random_extended_y(
    extracted_segments,
    # extended_y_file_name="extended_y.csv"
)

alg4_no_reconstruction.training(segments=extracted_segments, extended_y=extended_y)

with open("alg4.pickle", "wb") as outfile:
    pickle.dump(alg4_no_reconstruction, outfile)

with open("alg4.pickle", "rb") as infile:
    alg4_no_reconstruction = pickle.load(infile)

# alg4_no_reconstruction.predict(
#     epoch0_segments=epoch0_segments, epoch1_xyz=epoch1)

distances, uncertainties = alg4_no_reconstruction.compute_distances(
    epoch0_segments=epoch0_segments, epoch1_xyz=epoch1
)

print(distances)
print(uncertainties)
