import numpy as np
import pandas as pd
import pytest

from py4dgeo.pbm3c2 import PBM3C2
from py4dgeo.util import Py4DGeoError


class DummyEpoch:
    """
    Minimal stand-in for an Epoch needed by the new PBM3C2:
      - .cloud : numpy.ndarray (n_points, 3)
      - .additional_dimensions : dict-like with "segment_id" -> numpy array (n_points,)
    """

    def __init__(self, cloud: np.ndarray, segment_ids: np.ndarray):
        self.cloud = np.array(cloud, dtype=np.float64, copy=True)
        self.additional_dimensions = {
            "segment_id": np.array(segment_ids, dtype=np.int32, copy=True)
        }


def _make_cluster(center, n=5, spread=0.01, rng=None):
    if rng is None:
        rng = np.random
    return center + rng.normal(scale=spread, size=(n, 3))


def test_preprocess_detects_header(tmp_path):
    """
    If the correspondence CSV contains a textual header row, preprocess_epochs should raise Py4DGeoError.
    """
    epoch0 = DummyEpoch(np.zeros((3, 3)), [0, 1, 2])
    epoch1 = DummyEpoch(np.zeros((3, 3)), [0, 1, 2])

    corr_file = tmp_path / "corr_with_header.csv"
    with open(corr_file, "w") as fh:
        fh.write("id0,id1,label\n")
        fh.write("0,0,1\n")
        fh.write("1,1,0\n")

    with pytest.raises(Py4DGeoError):
        PBM3C2.preprocess_epochs(epoch0, epoch1, str(corr_file))


def test_preprocess_non_numeric_first_columns(tmp_path):
    """
    If the first two columns contain non-numeric values, preprocess_epochs should raise Py4DGeoError.
    """
    epoch0 = DummyEpoch(np.zeros((2, 3)), [0, 0])
    epoch1 = DummyEpoch(np.zeros((2, 3)), [0, 1])

    corr_file = tmp_path / "corr_bad.csv"
    with open(corr_file, "w") as fh:
        fh.write("foo,bar,baz\n")
        fh.write("0,0,1\n")

    with pytest.raises(Py4DGeoError):
        PBM3C2.preprocess_epochs(epoch0, epoch1, str(corr_file))


def test_preprocess_offsets_overlapping_segment_ids(tmp_path):
    """
    When epoch0 and epoch1 have overlapping segment ids, preprocess_epochs should offset epoch1 ids
    and update the correspondences second column by the same offset.
    """
    # epoch0 has segment ids 0 and 1
    epoch0 = DummyEpoch(np.zeros((4, 3)), [0, 0, 1, 1])
    # epoch1 also has segment id 0 and 2 -> overlap at id 0
    epoch1 = DummyEpoch(np.zeros((4, 3)), [0, 0, 2, 2])

    # correspondences: pairs (0,0,1) and (1,2,0) (no header)
    corr_rows = np.array([[0, 0, 1], [1, 2, 0]])
    corr_file = tmp_path / "corr.csv"
    np.savetxt(corr_file, corr_rows, delimiter=",", fmt="%d")

    e0_out, e1_out, corr_df = PBM3C2.preprocess_epochs(epoch0, epoch1, str(corr_file))

    ids0 = np.unique(e0_out.additional_dimensions["segment_id"])
    ids1 = np.unique(e1_out.additional_dimensions["segment_id"])

    # offset should be max(ids0) + 1
    expected_offset = ids0.max() + 1
    # original epoch1 first id 0 should become 0 + offset
    assert (e1_out.additional_dimensions["segment_id"][:2] == 0 + expected_offset).all()

    # correspondences_df second column should have been incremented by offset for relevant rows
    # corr_df's second column originally [0, 2] => now should be [0+offset, 2+offset]
    assert list(corr_df.iloc[:, 1].astype(int)) == [
        int(0 + expected_offset),
        int(2 + expected_offset),
    ]


def test_run_end_to_end(tmp_path):
    """
    Build two small synthetic epochs (2 segments each), a correspondences CSV (no header),
    run PBM3C2.run and assert returned correspondences contain distance & uncertainty
    and match PBM3C2._calculate_m3c2 computations. If classifier finds no correspondences, skip.
    """
    rng = np.random.RandomState(0)

    # epoch0: two segments centered at (0,0,0) and (10,0,0)
    pts0_seg0 = _make_cluster(np.array([0.0, 0.0, 0.0]), n=6, spread=0.01, rng=rng)
    pts0_seg1 = _make_cluster(np.array([10.0, 0.0, 0.0]), n=6, spread=0.01, rng=rng)
    cloud0 = np.vstack((pts0_seg0, pts0_seg1))
    segids0 = np.array([0] * pts0_seg0.shape[0] + [1] * pts0_seg1.shape[0])

    # epoch1: same two segments, shifted along x by +1.0 and -0.5 respectively
    pts1_seg0 = _make_cluster(np.array([1.0, 0.0, 0.0]), n=6, spread=0.01, rng=rng)
    pts1_seg1 = _make_cluster(np.array([9.5, 0.0, 0.0]), n=6, spread=0.01, rng=rng)
    cloud1 = np.vstack((pts1_seg0, pts1_seg1))
    segids1 = np.array([0] * pts1_seg0.shape[0] + [1] * pts1_seg1.shape[0])

    epoch0 = DummyEpoch(cloud0, segids0)
    epoch1 = DummyEpoch(cloud1, segids1)  # overlapping IDs intentionally

    # correspondences CSV: two rows: (0,0,1) positive, (1,1,0) negative -- no header
    corr_rows = np.array([[0, 0, 1], [1, 1, 0]])
    corr_file = tmp_path / "corr2.csv"
    np.savetxt(corr_file, corr_rows, delimiter=",", fmt="%d")

    pbm = PBM3C2(registration_error=0.05)

    result = pbm.run(
        epoch0=epoch0,
        epoch1=epoch1,
        correspondences_file=str(corr_file),
        apply_ids=[0, 1],
        search_radius=5.0,
    )

    assert result is not None
    assert isinstance(result, pd.DataFrame)

    if result.empty:
        pytest.skip(
            "No correspondences were found by the classifier in this synthetic test (acceptable)."
        )

    # Expect certain columns
    assert "epoch0_segment_id" in result.columns
    assert "epoch1_segment_id" in result.columns
    assert "distance" in result.columns
    assert "uncertainty" in result.columns

    # validate numeric consistency against internal _calculate_m3c2
    for _, row in result.iterrows():
        id0 = int(row["epoch0_segment_id"])
        id1 = int(row["epoch1_segment_id"])
        dist_expected, lod_expected = pbm._calculate_m3c2(id0, id1)
        assert pytest.approx(dist_expected, rel=1e-6, abs=1e-8) == float(
            row["distance"]
        )
        assert pytest.approx(lod_expected, rel=1e-6, abs=1e-8) == float(
            row["uncertainty"]
        )
