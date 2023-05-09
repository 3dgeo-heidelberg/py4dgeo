import py4dgeo
from py4dgeo.util import Py4DGeoError

import numpy as np
import os
import pickle
import pytest
import tempfile

import random

from sklearn.pipeline import Pipeline


def test_consistency(epochs):

    epoch0, epoch1 = epochs

    # ===============

    random.seed(10)
    np.random.seed(10)

    alg_cpy = py4dgeo.PB_M3C2()

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

    random.seed(10)
    np.random.seed(10)

    alg_original = py4dgeo.PB_M3C2()

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

    # assert np.array_equal(rez_original, rez_cpy), "unequal anymore"

    rez_original_second_call = alg_original.predict(epoch0=epoch0, epoch1=epoch1)

    assert np.array_equal(rez_original, rez_original_second_call), "unequal anymore"

    assert hash(alg_original._per_point_computation) == hash(
        alg_cpy._per_point_computation
    )
    assert hash(alg_original._segmentation) == hash(alg_cpy._segmentation)
    assert hash(alg_original._second_segmentation) == hash(alg_cpy._second_segmentation)
    assert hash(alg_original._extract_segments) == hash(alg_cpy._extract_segments)
    assert hash(alg_original._classifier) == hash(alg_cpy._classifier)

    original = Pipeline(
        [
            ("Transform_PerPointComputation", alg_original._per_point_computation),
            ("Transform_Segmentation", alg_original._segmentation),
            ("Transform_Second_Segmentation", alg_original._second_segmentation),
            ("Transform_ExtractSegments", alg_original._extract_segments),
            ("Classifier", alg_original._classifier),
        ]
    )

    original2 = Pipeline(
        [
            ("Transform_PerPointComputation", alg_original._per_point_computation),
            ("Transform_Segmentation", alg_original._segmentation),
            ("Transform_Second_Segmentation", alg_original._second_segmentation),
            ("Transform_ExtractSegments", alg_original._extract_segments),
            ("Classifier", alg_original._classifier),
        ]
    )

    cpy = Pipeline(
        [
            ("Transform_PerPointComputation", alg_cpy._per_point_computation),
            ("Transform_Segmentation", alg_cpy._segmentation),
            ("Transform_Second_Segmentation", alg_cpy._second_segmentation),
            ("Transform_ExtractSegments", alg_cpy._extract_segments),
            ("Classifier", alg_cpy._classifier),
        ]
    )

    assert hash(str(original.get_params())) == hash(str(cpy.get_params()))
    assert hash(str(original)) == hash(str(original2)), "not the same"
    assert hash(original) == hash(cpy), "not the same..."
