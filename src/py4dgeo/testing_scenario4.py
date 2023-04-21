import py4dgeo
import numpy as np

py4dgeo.set_interactive_backend("vtk")

py4dgeo.ensure_test_data_availability()

epoch0, epoch1 = py4dgeo.read_from_xyz(
    "plane_horizontal_t1.xyz", "plane_horizontal_t2.xyz"
)

# new_epoch0, new_epoch1 = py4dgeo.build_input_scenario2_without_normals(
#     epoch0=epoch0, epoch1=epoch1
# )

# you can also use data that have precomputed normals.
new_epoch0, new_epoch1 = py4dgeo.build_input_scenario2_with_normals(
    epoch0=epoch0, epoch1=epoch1
)


np.savetxt(
    "epoch0_segmented.xyz",
    new_epoch0,
    delimiter=",",
)

np.savetxt(
    "epoch1_segmented.xyz",
    new_epoch1,
    delimiter=",",
)

# epoch0_segmented = py4dgeo.read_from_xyz(
#     "epoch0_segmented.xyz",
#     additional_dimensions={3: "segment_id"},
#     **{"delimiter": ","}
# )

epoch0_segmented = py4dgeo.read_from_xyz(
    "epoch0_segmented.xyz",
    additional_dimensions={3: "N_x", 4: "N_y", 5: "N_z", 6: "segment_id"},
    **{"delimiter": ","}
)


# epoch1_segmented = py4dgeo.read_from_xyz(
#     "epoch1_segmented.xyz",
#     additional_dimensions={3: "segment_id"},
#     **{"delimiter": ","}
# )

epoch1_segmented = py4dgeo.read_from_xyz(
    "epoch1_segmented.xyz",
    additional_dimensions={3: "N_x", 4: "N_y", 5: "N_z", 6: "segment_id"},
    **{"delimiter": ","}
)

alg = py4dgeo.PB_M3C2_time_series()

(xyz_epoch0, xyz_epoch1, segments) = alg.export_segmented_point_cloud_and_segments(
    epoch0_xyz_id_normal=epoch0_segmented,
    epoch1_xyz=epoch1,
    #    **{"Transform Post Segmentation__output_file_name": "seg_test_post_seg.out", "get_pipeline_options": True}
)

# py4dgeo.Viewer.segments_visualizer(X=segments)

extended_y = py4dgeo.generate_random_extended_y(
    segments, extended_y_file_name="extended_y.csv"
)

alg.training(segments=segments, extended_y=extended_y)

alg.predict(epoch0=epoch0_segmented, epoch1=epoch1)


distances, uncertainties = alg.compute_distances(epoch0=epoch0_segmented, epoch1=epoch1)

print(distances)
print(uncertainties)
