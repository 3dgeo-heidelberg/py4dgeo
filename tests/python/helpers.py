import numpy as np


def compare_algorithms(alg1, alg2):
    """A helper to compare the output of two algorithms that should be equivalent"""
    # Run the two algorithms
    distances1, uncertainties1 = alg1.run()
    distances2, uncertainties2 = alg2.run()

    assert np.allclose(distances1, distances2)
    assert np.allclose(uncertainties1["lodetection"], uncertainties2["lodetection"])
    assert np.allclose(uncertainties1["spread1"], uncertainties2["spread1"])
    assert np.allclose(uncertainties1["spread2"], uncertainties2["spread2"])
    assert np.allclose(uncertainties1["num_samples1"], uncertainties2["num_samples1"])
    assert np.allclose(uncertainties1["num_samples2"], uncertainties2["num_samples2"])
