import py4dgeo

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def test_consistency(epochs):
    epoch0, epoch1 = epochs

    # ===============

    alg_cpy = py4dgeo.PB_M3C2(
        classifier=py4dgeo.ClassifierWrapper(
            classifier=RandomForestClassifier(random_state=42)
        )
    )

    (
        x_y_z_id_epoch0,
        x_y_z_id_epoch1,
        extracted_segments,
    ) = alg_cpy.export_segmented_point_cloud_and_segments(
        epoch0=epoch0,
        epoch1=epoch1,
        x_y_z_id_epoch0_file_name=None,
        x_y_z_id_epoch1_file_name=None,
        extracted_segments_file_name=None,
    )
    extended_y = py4dgeo.generate_random_extended_y(
        extracted_segments, extended_y_file_name="extended_y.csv"
    )
    alg_cpy.training(extracted_segments, extended_y)

    rez_cpy = alg_cpy.predict(epoch0=epoch0, epoch1=epoch1)

    # ===============

    alg_original = py4dgeo.PB_M3C2(
        classifier=py4dgeo.ClassifierWrapper(
            classifier=RandomForestClassifier(random_state=42)
        )
    )

    # (
    #     x_y_z_id_epoch0,
    #     x_y_z_id_epoch1,
    #     extracted_segments,
    # ) = \
    alg_original.export_segmented_point_cloud_and_segments(
        epoch0=epoch0,
        epoch1=epoch1,
        x_y_z_id_epoch0_file_name=None,
        x_y_z_id_epoch1_file_name=None,
        extracted_segments_file_name=None,
    )
    # extended_y = py4dgeo.generate_random_extended_y(extracted_segments, extended_y_file_name="extended_y.csv")

    alg_original.training(extracted_segments, extended_y)

    rez_original = alg_original.predict(epoch0=epoch0, epoch1=epoch1)

    assert np.array_equal(rez_original, rez_cpy), "unequal anymore"
