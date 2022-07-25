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


def simple_jump():
    # A simple time series with a jump
    ts = np.linspace(0, 0.1, 100)
    ts[50:] += 1

    return ts


def complex_timeseries():
    # A non-trivial time series
    return np.array(
        [
            -9.30567119,
            -6.50542506,
            -5.25334064,
            -3.95708071,
            -10.62295044,
            -6.00679331,
            -7.10972198,
            -4.75825001,
            -6.7100845,
            -8.11943878,
            -9.56607421,
            -4.94672353,
            -6.67989247,
            -4.65803801,
            -7.37845623,
            -7.48818285,
            -5.78868842,
            -5.63853894,
            -6.82157223,
            -4.36439707,
            -10.23610407,
            -7.17848293,
            -10.35211983,
            -11.3509054,
            -11.61486292,
            -9.35727084,
            -10.34027666,
            -10.86800044,
            -9.93429977,
            -8.48897875,
            -15.3695928,
            -14.26784041,
            -9.31700749,
            -10.71438333,
            -6.79820964,
            -9.38362261,
            -10.55322992,
            -10.752245,
            -13.40571741,
            -14.18765576,
            -9.54996564,
            -19.35102448,
            -20.7635553,
            -19.26623954,
            -20.00548551,
            -17.99023991,
            -18.37537823,
            -17.53357472,
            -23.12740104,
            -17.55591524,
            -20.86915052,
            -17.75074536,
            -20.28542954,
            -20.5310157,
            -18.73412243,
            -19.07024164,
            -20.43017388,
            -21.11249791,
            -18.45269281,
            -18.07089436,
            -19.96608851,
            -26.63058919,
            -28.97962105,
            -25.47049917,
            -26.02732842,
            -25.42334059,
            -24.96684487,
            -22.69612178,
            -24.85381947,
            -25.34017963,
            -27.46753899,
            -24.05766122,
            -29.416168,
            -19.5100961,
            -25.98309514,
            -26.24838995,
            -25.40961916,
            -23.21292358,
            -26.63478854,
            -29.04883697,
            -16.35986185,
            -12.87413666,
            -16.2863866,
            -15.55762212,
            -21.88964465,
            -17.68327092,
            -16.60648862,
            -13.49961018,
            -15.97483962,
            -17.23104101,
            -18.17963806,
            -16.19978101,
            -17.62239823,
            -17.14176606,
            -14.73581351,
            -14.62187256,
            -20.06316631,
            -14.96743604,
            -19.01623592,
            -15.53150285,
        ]
    )
