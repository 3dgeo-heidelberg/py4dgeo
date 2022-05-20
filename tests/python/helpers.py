import numpy as np


def compare_uncertainties(uncertainties1, uncertainties2):
    assert np.allclose(uncertainties1["lodetection"], uncertainties2["lodetection"])
    assert np.allclose(uncertainties1["spread1"], uncertainties2["spread1"])
    assert np.allclose(uncertainties1["spread2"], uncertainties2["spread2"])
    assert np.allclose(uncertainties1["num_samples1"], uncertainties2["num_samples1"])
    assert np.allclose(uncertainties1["num_samples2"], uncertainties2["num_samples2"])


def compare_algorithms(alg1, alg2):
    """A helper to compare the output of two algorithms that should be equivalent"""
    # Run the two algorithms
    distances1, uncertainties1 = alg1.run()
    distances2, uncertainties2 = alg2.run()

    assert np.allclose(distances1, distances2)
    compare_uncertainties(uncertainties1, uncertainties2)


def compare_segmentations(seg1, seg2):
    assert np.allclose(seg1.distances, seg2.distances)
    compare_uncertainties(seg2.uncertainties, seg2.uncertainties)

    for td1, td2 in zip(seg1.timedeltas, seg2.timedeltas):
        assert td1 == td2
