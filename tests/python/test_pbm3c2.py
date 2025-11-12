import py4dgeo
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def test_compute_distances(epochs_segmented, pbm3c2_correspondences_file):
    epoch0, epoch1 = epochs_segmented
    correspondences_file = pbm3c2_correspondences_file
    apply_ids = np.arange(1,31)

    alg = py4dgeo.PBM3C2(registration_error=0.01)

    rez = alg.run(
        epoch0=epoch0, 
        epoch1=epoch1,
        correspondences_file=correspondences_file,
        apply_ids=apply_ids,
        search_radius=5.0,
    )
    
    assert rez is not None


