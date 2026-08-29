import py4dgeo
import numpy as np
import os
import pytest

from py4dgeo.util import Py4DGeoError


def test_read_segmented_epochs(epochs_segmented, pbm3c2_correspondences_file):
    epoch0, epoch1 = epochs_segmented
    correspondences_file = pbm3c2_correspondences_file

    assert epoch0 is not None
    assert epoch1 is not None
    assert os.path.exists(
        correspondences_file
    ), f"Correspondences file should exist: {correspondences_file}"

    assert epoch0.cloud.shape[0] > 0, "epoch0 should have points"
    assert epoch1.cloud.shape[0] > 0, "epoch1 should have points"

    assert (
        epoch0.additional_dimensions is not None
    ), "epoch0 additional_dimensions should not be None"
    assert (
        epoch1.additional_dimensions is not None
    ), "epoch1 additional_dimensions should not be None"


def test_preprocess(epochs_segmented, pbm3c2_correspondences_file):
    epoch0, epoch1 = epochs_segmented
    correspondences_file = pbm3c2_correspondences_file

    alg = py4dgeo.PBM3C2(registration_error=0.01)

    (
        epoch0_preprocessed,
        epoch1_preprocessed,
        correspondences_arr,
        epoch0_id_mapping,
        epoch1_id_mapping,
        epoch0_reverse_mapping,
        epoch1_reverse_mapping,
    ) = alg.preprocess_epochs(epoch0, epoch1, correspondences_file)

    assert epoch0_preprocessed is not None
    assert epoch1_preprocessed is not None
    assert correspondences_arr is not None
    assert (
        correspondences_arr.shape[1] >= 2
    ), "Correspondences should have at least 2 columns"

    assert (
        epoch0_preprocessed.cloud.shape[0] > 0
    ), "Preprocessed epoch0 should have points"
    assert (
        epoch1_preprocessed.cloud.shape[0] > 0
    ), "Preprocessed epoch1 should have points"


def test_compute_distances(epochs_segmented, pbm3c2_correspondences_file):
    epoch0, epoch1 = epochs_segmented
    correspondences_file = pbm3c2_correspondences_file
    apply_ids = np.arange(1, 31)

    alg = py4dgeo.PBM3C2(registration_error=0.01)

    rez = alg.run(
        epoch0=epoch0,
        epoch1=epoch1,
        correspondences_file=correspondences_file,
        apply_ids=apply_ids,
        search_radius=5.0,
    )

    assert rez is not None


def test_compute_distances_nearest_neighbor_without_training(epochs_segmented):
    epoch0, epoch1 = epochs_segmented
    apply_ids = np.arange(1, 31)

    alg = py4dgeo.PBM3C2(registration_error=0.01)

    rez = alg.run(
        epoch0=epoch0,
        epoch1=epoch1,
        correspondences_file=None,
        apply_ids=apply_ids,
        search_radius=5.0,
        correspondence_method="nearest_neighbor",
    )

    assert rez is not None
    if not rez.empty:
        for col in (
            "epoch0_original_id",
            "epoch1_original_id",
            "epoch0_segment_id",
            "epoch1_segment_id",
            "distance",
            "uncertainty",
        ):
            assert col in rez.columns


def test_pbm3c2_invalid_correspondence_method_raises(epochs_segmented):
    epoch0, epoch1 = epochs_segmented

    with pytest.raises(Py4DGeoError):
        py4dgeo.PBM3C2(registration_error=0.01).run(
            epoch0=epoch0,
            epoch1=epoch1,
            correspondences_file=None,
            apply_ids=np.array([1]),
            correspondence_method="invalid",
        )
