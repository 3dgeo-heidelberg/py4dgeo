import py4dgeo
import numpy as np
import pytest
import os
import logging

def test_read_segmented_epochs(epochs_segmented, pbm3c2_correspondences_file):
    epoch0, epoch1 = epochs_segmented
    correspondences_file = pbm3c2_correspondences_file

    assert epoch0 is not None
    assert epoch1 is not None
    assert os.path.exists(correspondences_file), f"Correspondences file should exist: {correspondences_file}"
    
    assert epoch0.cloud.shape[0] > 0, "epoch0 should have points"
    assert epoch1.cloud.shape[0] > 0, "epoch1 should have points"

    assert epoch0.additional_dimensions is not None, "epoch0 additional_dimensions should not be None"
    assert epoch1.additional_dimensions is not None, "epoch1 additional_dimensions should not be None"


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup logging handlers after test to prevent Windows file lock issues"""
    yield
    for logger_name in ['py4dgeo', 'root']:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            try:
                handler.close()
                logger.removeHandler(handler)
            except:
                pass


# def test_preprocess(epochs_segmented,pbm3c2_correspondences_file):
#     epoch0, epoch1 = epochs_segmented
#     correspondences_file = pbm3c2_correspondences_file

#     alg = py4dgeo.PBM3C2(registration_error=0.01)
#     epoch0_preprocessed, epoch1_preprocessed, correspondences_df =alg.preprocess_epochs(epoch0, epoch1, correspondences_file)

#     assert epoch0_preprocessed is not None
#     assert epoch1_preprocessed is not None
#     assert correspondences_df is not None

#     assert epoch0_preprocessed.cloud.shape[0] > 0, "Preprocessed epoch0 should have points"
#     assert epoch1_preprocessed.cloud.shape[0] > 0, "Preprocessed epoch1 should have points"


# def test_compute_distances(epochs_segmented, pbm3c2_correspondences_file):
#     epoch0, epoch1 = epochs_segmented
#     correspondences_file = pbm3c2_correspondences_file
#     apply_ids = np.arange(1, 31)

#     alg = py4dgeo.PBM3C2(registration_error=0.01)

#     rez = alg.run(
#         epoch0=epoch0,
#         epoch1=epoch1,
#         correspondences_file=correspondences_file,
#         apply_ids=apply_ids,
#         search_radius=5.0,
#     )

#     assert rez is not None