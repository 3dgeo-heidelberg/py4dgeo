import py4dgeo
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def test_compute_distances(epochs_segmented, pbm3c2_labels):
    epoch0, epoch1 = epochs_segmented
    labels = pbm3c2_labels
    apply_ids = np.arange(1, 31)

    alg = py4dgeo.PBM3C2(registration_error=0.01)

    rez = alg.run(
        epoch0=epoch0,
        epoch1=epoch1,
        correspondences_file=labels,
        apply_ids=apply_ids,
        search_radius=5.0,
    )

    assert rez is not None
