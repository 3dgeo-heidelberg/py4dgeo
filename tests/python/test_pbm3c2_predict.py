import py4dgeo

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def test_predict(epochs):
    epoch0, epoch1 = epochs

    alg = py4dgeo.PBM3C2(
        classifier=py4dgeo.ClassifierWrapper(
            classifier=RandomForestClassifier(random_state=42)
        )
    )

    (
        x_y_z_id_epoch0,
        x_y_z_id_epoch1,
        extracted_segments,
    ) = alg.export_segmented_point_cloud_and_segments(
        epoch0=epoch0,
        epoch1=epoch1,
        **{
            # "c": True, # used for testing
            # "get_pipeline_options": True,
            # "Transform_Segmentation__output_file_name": "segmented_point_cloud.out"
        },
    )
    # segmented_point_cloud = py4dgeo.Viewer.read_np_ndarray_from_xyz(
    #     input_file_name="segmented_point_cloud.out"
    # )
    # py4dgeo.Viewer.segmented_point_cloud_visualizer(X=segmented_point_cloud)

    extended_y = py4dgeo.generate_random_extended_y(
        extracted_segments, extended_y_file_name="extended_y.csv"
    )

    alg.training(extracted_segments, extended_y)
    # alg.training(
    #     extracted_segments_file_name="extracted_segments.seg",
    #     extended_y_file_name="extended_y.csv",
    # )

    rez0 = alg.predict(epoch0=epoch0, epoch1=epoch1)
    # print(alg.predict(epoch0=epoch0, epoch1=epoch1, get_pipeline_option=True))

    (
        _0,
        _1,
        extracted_segments_epoch0,
    ) = alg.export_segmented_point_cloud_and_segments(
        epoch0=epoch0,
        # epoch1=None,
        x_y_z_id_epoch0_file_name=None,
        x_y_z_id_epoch1_file_name=None,
        extracted_segments_file_name=None,
    )

    rez1 = alg.predict(
        epoch0=extracted_segments_epoch0,
        epoch1=epoch1,
        get_pipeline_options=True,
        epoch0_Transform_PerPointComputation__skip=True,
        epoch0_Transform_Segmentation__skip=True,
        epoch0_Transform_Second_Segmentation__skip=True,
        epoch0_Transform_ExtractSegments__skip=True,
    )

    config_epoch0_as_segments = {
        "get_pipeline_options": True,
        "epoch0_Transform_PerPointComputation__skip": True,
        "epoch0_Transform_Segmentation__skip": True,
        "epoch0_Transform_Second_Segmentation__skip": True,
        "epoch0_Transform_ExtractSegments__skip": True,
    }

    rez2 = alg.predict(
        epoch0=extracted_segments_epoch0, epoch1=epoch1, **config_epoch0_as_segments
    )

    assert np.array_equal(rez0, rez1), "unequal anymore"
    assert np.array_equal(rez0, rez1), "unequal anymore"
    assert np.array_equal(rez1, rez2), "unequal anymore"
