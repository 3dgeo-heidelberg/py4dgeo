import numpy as np
import py4dgeo
import math
import time
import scipy.stats as sstats
import multiprocessing as mp
import laspy

from py4dgeo.epoch import Epoch
from py4dgeo.util import (
    as_double_precision,
    Py4DGeoError,
)

from py4dgeo import M3C2

import warnings

warnings.filterwarnings("ignore")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

default_tfM = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])


class M3C2EP(M3C2):
    def __init__(
        self,
        tfM: np.ndarray = default_tfM,
        Cxx: np.ndarray = np.zeros((12, 12)),
        refPointMov: np.ndarray = np.array([0, 0, 0]),
        perform_trans: bool = True,
        **kwargs,
    ):
        """An M3C2-EP implementation
        that push the limits of 3D topographic point cloud change detection by error propagation.
        The algorithm needs an alignment covariance matrix of shape 12 x 12, an affine transformation matrix
        of shape 3 x 4 and a reduction point  (x0,y0,z0) (rotation origin, 3 parameters) obtained from
        aligning the two point clouds. The formula of the transformation see in user docs.
        The transformation can be set by a boolean flag 'perform_trans' and is performed by default.
        """
        assert tfM.shape == (3, 4)
        assert refPointMov.shape == (3,)
        assert Cxx.shape == (12, 12)

        self.tfM = tfM
        self.Cxx = Cxx
        self.refPointMov = refPointMov
        self.perform_trans = perform_trans
        super().__init__(**kwargs)

    def calculate_distances(self, epoch1, epoch2):
        print(self.name + " running")
        """Calculate the distances between two epochs"""

        if not isinstance(self.cyl_radius, float):
            raise Py4DGeoError(
                f"{self.name} requires exactly one cylinder radius to be given"
            )

        # Ensure appropriate trees are built
        epoch1._validate_search_tree()
        epoch2._validate_search_tree()

        p1_coords = epoch1.cloud
        p1_positions = epoch1.scanpos_id
        p2_coords = epoch2.cloud
        p2_positions = epoch2.scanpos_id

        # set default M3C2Meta
        M3C2Meta = {"searchrad": 0.5, "maxdist": 3, "minneigh": 5, "maxneigh": 100000}
        M3C2Meta["searchrad"] = self.cyl_radius
        M3C2Meta["maxdist"] = self.max_distance

        M3C2Meta["spInfos"] = [epoch1.scanpos_info, epoch2.scanpos_info]
        M3C2Meta["tfM"] = self.tfM
        M3C2Meta["Cxx"] = self.Cxx
        M3C2Meta["redPoint"] = self.refPointMov

        refPointMov = self.refPointMov
        tfM = self.tfM

        # transform p2
        if self.perform_trans:
            p2_coords = p2_coords - refPointMov
            p2_coords = np.dot(tfM[:3, :3], p2_coords.T).T + tfM[:, 3] + refPointMov

        # load query points
        query_coords = self.corepoints
        query_norms = self.directions()

        # Repeat normals to shape of corepoints when the user explicitly provided one direction
        if query_norms.shape[0] == 1:
            query_norms = query_norms.repeat(self.corepoints.shape[0], axis=0)

        if query_norms is None:
            raise Py4DGeoError("Core point point cloud needs normals set. Exiting.")
            exit(-1)
        subsample = False
        if subsample:
            sub_idx = np.random.choice(np.arange(0, query_coords.shape[0]), 2000)
            query_coords = query_coords[sub_idx]
            query_norms = query_norms[sub_idx]

        NUM_THREADS = 4
        NUM_BLOCKS = 16

        query_coords_subs = np.array_split(query_coords, NUM_BLOCKS)
        query_norms_subs = np.array_split(query_norms, NUM_BLOCKS)

        # start mp
        manager = mp.Manager()
        return_dict = manager.dict()

        # prepare shared memory
        p1_coords_shm = mp.shared_memory.SharedMemory(
            create=True, size=p1_coords.nbytes
        )
        p1_coords_sha = np.ndarray(
            p1_coords.shape, dtype=p1_coords.dtype, buffer=p1_coords_shm.buf
        )
        p1_coords_sha[:] = p1_coords[:]
        p2_coords_shm = mp.shared_memory.SharedMemory(
            create=True, size=p2_coords.nbytes
        )
        p2_coords_sha = np.ndarray(
            p2_coords.shape, dtype=p2_coords.dtype, buffer=p2_coords_shm.buf
        )
        p2_coords_sha[:] = p2_coords[:]

        max_dist = M3C2Meta["maxdist"]
        search_radius = M3C2Meta["searchrad"]
        effective_search_radius = math.hypot(max_dist, search_radius)

        # Querying neighbours
        pbarQueue = mp.Queue()
        pbarProc = mp.Process(
            target=updatePbar, args=(query_coords.shape[0], pbarQueue, NUM_THREADS)
        )
        pbarProc.start()
        procs = []

        last_started_idx = -1
        running_ps = []
        while True:
            if len(running_ps) < NUM_THREADS:
                last_started_idx += 1
                if last_started_idx < len(query_coords_subs):
                    curr_subs = query_coords_subs[last_started_idx]
                    p1_idx = radius_search(epoch1, curr_subs, effective_search_radius)
                    p2_idx = radius_search(epoch2, curr_subs, effective_search_radius)

                    p = mp.Process(
                        target=process_corepoint_list,
                        args=(
                            curr_subs,
                            query_norms_subs[last_started_idx],
                            p1_idx,
                            p1_coords_shm.name,
                            p1_coords.shape,
                            p1_positions,
                            p2_idx,
                            p2_coords_shm.name,
                            p2_coords.shape,
                            p2_positions,
                            M3C2Meta,
                            last_started_idx,
                            return_dict,
                            pbarQueue,
                        ),
                    )
                    procs.append(p)

                    procs[last_started_idx].start()
                    running_ps.append(last_started_idx)
                else:
                    break
            for running_p in running_ps:
                if not procs[running_p].is_alive():
                    running_ps.remove(running_p)
            time.sleep(1)

        for p in procs:
            p.join()

        pbarQueue.put((0, 0))
        pbarProc.terminate()

        p1_coords_shm.close()
        p1_coords_shm.unlink()
        p2_coords_shm.close()
        p2_coords_shm.unlink()

        out_attrs = {
            key: (
                np.empty((query_coords.shape[0], 3, 3), dtype=val.dtype)
                if key == "m3c2_cov1" or key == "m3c2_cov2"
                else np.empty(query_coords.shape[0], dtype=val.dtype)
            )
            for key, val in return_dict[0].items()
        }
        for key in out_attrs:
            curr_start = 0
            for i in range(NUM_BLOCKS):
                curr_len = return_dict[i][key].shape[0]
                out_attrs[key][curr_start : curr_start + curr_len] = return_dict[i][key]
                curr_start += curr_len

        distances = out_attrs["val"]
        cov1 = out_attrs["m3c2_cov1"]
        cov2 = out_attrs["m3c2_cov2"]
        unc = {
            "lodetection": out_attrs["lod_new"],
            "spread1": out_attrs["m3c2_spread1"],
            "num_samples1": out_attrs["m3c2_n1"],
            "spread2": out_attrs["m3c2_spread2"],
            "num_samples2": out_attrs["m3c2_n2"],
        }

        unc_list = []
        cov_list = []
        for i in range(unc["lodetection"].shape[0]):
            unc_item = (
                unc["lodetection"][i],
                unc["spread1"][i],
                unc["num_samples1"][i],
                unc["spread2"][i],
                unc["num_samples2"][i],
            )
            unc_list.append(unc_item)
            cov_item = (cov1[i], cov2[i])
            cov_list.append(cov_item)

        uncertainties = np.array(
            unc_list,
            dtype=[
                ("lodetection", "f8"),
                ("spread1", "f8"),
                ("num_samples1", "i8"),
                ("spread2", "f8"),
                ("num_samples2", "i8"),
            ],
        )
        covariance = np.array(
            cov_list, dtype=[("cov1", "f8", (3, 3)), ("cov2", "f8", (3, 3))]
        )
        print(self.name + " end")
        return distances, uncertainties, covariance

    @property
    def name(self):
        return "M3C2EP"


def updatePbar(total, queue, maxProc):
    desc = "Processing core points"
    pCount = 0
    if tqdm is None:
        pbar = None
    else:
        pbar = tqdm(
            total=total,
            ncols=100,
            desc=desc + " (%02d/%02d Process(es))" % (pCount, maxProc),
        )

    while True:
        inc, process = queue.get()
        if pbar is not None:
            pbar.update(inc)
            if process != 0:
                pCount += process
                pbar.set_description(
                    desc + " (%02d/%02d Process(es))" % (pCount, maxProc)
                )


eijk = np.zeros((3, 3, 3))
eijk[0, 1, 2] = eijk[1, 2, 0] = eijk[2, 0, 1] = 1
eijk[0, 2, 1] = eijk[2, 1, 0] = eijk[1, 0, 2] = -1

dij = np.zeros((3, 3))
dij[0, 0] = dij[1, 1] = dij[2, 2] = 1

n = np.zeros((3,))
poa_pts = np.zeros((3, 100))
path_opt = np.einsum_path(
    "mi, ijk, j, kn -> mn", dij, eijk, n, poa_pts, optimize="optimal"
)


def getAlongAcrossSqBatch(pts, poa, n):
    pts_poa = pts - poa[:, np.newaxis]
    alongs = n.dot(pts_poa)
    poa_pts = poa[:, np.newaxis] - pts
    crosses = np.einsum(
        "mi, ijk, j, kn -> mn", dij, eijk, n, poa_pts, optimize=path_opt[0]
    )
    across2 = np.einsum("ij, ij -> j", crosses, crosses)
    return (alongs, across2)


def get_local_mean_and_Cxx_nocorr(
    Cxx, tfM, origins, redPoint, sigmas, curr_pts, curr_pos, epoch, tf=True
):
    nPts = curr_pts.shape[0]
    A = np.tile(np.eye(3), (nPts, 1))
    ATP = np.zeros((3, 3 * nPts))
    tfM = tfM if tf else np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    dx = np.zeros((nPts,), dtype=np.float64)
    dy = np.zeros((nPts,), dtype=np.float64)
    dz = np.zeros((nPts,), dtype=np.float64)
    rrange = np.zeros((nPts,), dtype=np.float64)
    sinscan = np.zeros((nPts,), dtype=np.float64)
    cosscan = np.zeros((nPts,), dtype=np.float64)
    cosyaw = np.zeros((nPts,), dtype=np.float64)
    sinyaw = np.zeros((nPts,), dtype=np.float64)
    sigmaRange = np.zeros((nPts,), dtype=np.float64)
    sigmaYaw = np.zeros((nPts,), dtype=np.float64)
    sigmaScan = np.zeros((nPts,), dtype=np.float64)

    for scanPosId in np.unique(curr_pos):
        scanPos = np.array(origins[scanPosId - 1, :])
        scanPosPtsIdx = curr_pos == scanPosId

        dd = curr_pts[scanPosPtsIdx, :] - scanPos[np.newaxis, :]
        dlx, dly, dlz = dd[:, 0], dd[:, 1], dd[:, 2]
        yaw = np.arctan2(dly, dlx)
        planar_dist = np.hypot(dlx, dly)
        scan = np.pi / 2 - np.arctan(dlz / planar_dist)
        rrange[scanPosPtsIdx] = np.hypot(planar_dist, dlz)
        sinscan[scanPosPtsIdx] = np.sin(scan)
        cosscan[scanPosPtsIdx] = np.cos(scan)
        sinyaw[scanPosPtsIdx] = np.sin(yaw)
        cosyaw[scanPosPtsIdx] = np.cos(yaw)

        dr = curr_pts[scanPosPtsIdx, :] - redPoint
        dx[scanPosPtsIdx] = dr[:, 0]
        dy[scanPosPtsIdx] = dr[:, 1]
        dz[scanPosPtsIdx] = dr[:, 2]

        sigmaRange[scanPosPtsIdx] = np.array(
            np.sqrt(
                sigmas[scanPosId - 1][0] ** 2
                + sigmas[scanPosId - 1][1] * 1e-6 * rrange[scanPosPtsIdx] ** 2
            )
        )  # a + b*d
        sigmaYaw[scanPosPtsIdx] = np.array(sigmas[scanPosId - 1][2])
        sigmaScan[scanPosPtsIdx] = np.array(sigmas[scanPosId - 1][3])

    if tf:
        SigmaXiXj = (
            dx**2 * Cxx[0, 0]
            + 2 * dx * dy * Cxx[0, 1]  # a11a11
            + dy**2 * Cxx[1, 1]  # a11a12
            + 2 * dy * dz * Cxx[1, 2]  # a12a12
            + dz**2 * Cxx[2, 2]  # a12a13
            + 2 * dz * dx * Cxx[0, 2]  # a13a13
            + 2  # a11a13
            * (dx * Cxx[0, 9] + dy * Cxx[1, 9] + dz * Cxx[2, 9])  # a11tx  # a12tx
            + Cxx[9, 9]  # a13tx
        )  # txtx

        SigmaYiYj = (
            dx**2 * Cxx[3, 3]
            + 2 * dx * dy * Cxx[3, 4]  # a21a21
            + dy**2 * Cxx[4, 4]  # a21a22
            + 2 * dy * dz * Cxx[4, 5]  # a22a22
            + dz**2 * Cxx[5, 5]  # a22a23
            + 2 * dz * dx * Cxx[3, 5]  # a23a23
            + 2  # a21a23
            * (dx * Cxx[3, 10] + dy * Cxx[4, 10] + dz * Cxx[5, 10])  # a21ty  # a22ty
            + Cxx[10, 10]  # a23ty
        )  # tyty

        SigmaZiZj = (
            dx**2 * Cxx[6, 6]
            + 2 * dx * dy * Cxx[6, 7]  # a31a31
            + dy**2 * Cxx[7, 7]  # a31a32
            + 2 * dy * dz * Cxx[7, 8]  # a32a32
            + dz**2 * Cxx[8, 8]  # a32a33
            + 2 * dz * dx * Cxx[6, 8]  # a33a33
            + 2  # a31a33
            * (dx * Cxx[6, 11] + dy * Cxx[7, 11] + dz * Cxx[8, 11])  # a31tz  # a32tz
            + Cxx[11, 11]  # a33tz
        )  # tztz

        SigmaXiYj = (
            Cxx[9, 10]
            + dx * Cxx[0, 10]  # txty
            + dy * Cxx[1, 10]  # a11ty
            + dz * Cxx[2, 10]  # a12ty
            + dx  # a13ty
            * (Cxx[3, 9] + Cxx[0, 3] * dx + Cxx[1, 3] * dy + Cxx[2, 3] * dz)
            + dy * (Cxx[4, 9] + Cxx[0, 4] * dx + Cxx[1, 4] * dy + Cxx[2, 4] * dz)
            + dz * (Cxx[5, 9] + Cxx[0, 5] * dx + Cxx[1, 5] * dy + Cxx[2, 5] * dz)
        )

        SigmaXiZj = (
            Cxx[9, 11]
            + dx * Cxx[0, 11]  # txtz
            + dy * Cxx[1, 11]  # a11tz
            + dz * Cxx[2, 11]  # a12tz
            + dx  # a13tz
            * (Cxx[6, 9] + Cxx[0, 6] * dx + Cxx[1, 6] * dy + Cxx[2, 6] * dz)
            + dy * (Cxx[7, 9] + Cxx[0, 7] * dx + Cxx[1, 7] * dy + Cxx[2, 7] * dz)
            + dz * (Cxx[8, 9] + Cxx[0, 8] * dx + Cxx[1, 8] * dy + Cxx[2, 8] * dz)
        )

        SigmaYiZj = (
            Cxx[10, 11]
            + dx * Cxx[6, 10]  # tytz
            + dy * Cxx[7, 10]  # a21tx
            + dz * Cxx[8, 10]  # a22tx
            + dx  # a23tx
            * (Cxx[3, 11] + Cxx[3, 6] * dx + Cxx[3, 7] * dy + Cxx[3, 8] * dz)
            + dy * (Cxx[4, 11] + Cxx[4, 6] * dx + Cxx[4, 7] * dy + Cxx[4, 8] * dz)
            + dz * (Cxx[5, 11] + Cxx[5, 6] * dx + Cxx[5, 7] * dy + Cxx[5, 8] * dz)
        )
        C11 = np.sum(SigmaXiXj)  # sum over all j
        C12 = np.sum(SigmaXiYj)  # sum over all j
        C13 = np.sum(SigmaXiZj)  # sum over all j
        C22 = np.sum(SigmaYiYj)  # sum over all j
        C23 = np.sum(SigmaYiZj)  # sum over all j
        C33 = np.sum(SigmaZiZj)  # sum over all j
        local_Cxx = np.array([[C11, C12, C13], [C12, C22, C23], [C13, C23, C33]])
    else:
        local_Cxx = np.zeros((3, 3))

    C11p = (
        (
            tfM[0, 0] * cosyaw * sinscan
            + tfM[0, 1] * sinyaw * sinscan  # dX/dRange - measurements
            + tfM[0, 2] * cosscan
        )
        ** 2
        * sigmaRange**2
        + (
            -1 * tfM[0, 0] * rrange * sinyaw * sinscan
            + tfM[0, 1] * rrange * cosyaw * sinscan  # dX/dYaw
        )
        ** 2
        * sigmaYaw**2
        + (
            tfM[0, 0] * rrange * cosyaw * cosscan
            + tfM[0, 1] * rrange * sinyaw * cosscan  # dX/dScan
            + -1 * tfM[0, 2] * rrange * sinscan
        )
        ** 2
        * sigmaScan**2
    )

    C12p = (
        (
            tfM[1, 0] * cosyaw * sinscan
            + tfM[1, 1] * sinyaw * sinscan  # dY/dRange - measurements
            + tfM[1, 2] * cosscan
        )
        * (
            tfM[0, 0] * cosyaw * sinscan
            + tfM[0, 1] * sinyaw * sinscan  # dX/dRange - measurements
            + tfM[0, 2] * cosscan
        )
        * sigmaRange**2
        + (
            -1 * tfM[1, 0] * rrange * sinyaw * sinscan
            + tfM[1, 1] * rrange * cosyaw * sinscan  # dY/dYaw
        )
        * (
            -1 * tfM[0, 0] * rrange * sinyaw * sinscan
            + tfM[0, 1] * rrange * cosyaw * sinscan  # dX/dYaw
        )
        * sigmaYaw**2
        + (
            tfM[0, 0] * rrange * cosyaw * cosscan
            + tfM[0, 1] * rrange * sinyaw * cosscan  # dX/dScan
            + -1 * tfM[0, 2] * rrange * sinscan
        )
        * (
            tfM[1, 0] * rrange * cosyaw * cosscan
            + tfM[1, 1] * rrange * sinyaw * cosscan  # dY/dScan
            + -1 * tfM[1, 2] * rrange * sinscan
        )
        * sigmaScan**2
    )

    C22p = (
        (
            tfM[1, 0] * cosyaw * sinscan
            + tfM[1, 1] * sinyaw * sinscan  # dY/dRange - measurements
            + tfM[1, 2] * cosscan
        )
        ** 2
        * sigmaRange**2
        + (
            -1 * tfM[1, 0] * rrange * sinyaw * sinscan
            + tfM[1, 1] * rrange * cosyaw * sinscan  # dY/dYaw
        )
        ** 2
        * sigmaYaw**2
        + (
            tfM[1, 0] * rrange * cosyaw * cosscan
            + tfM[1, 1] * rrange * sinyaw * cosscan  # dY/dScan
            + -1 * tfM[1, 2] * rrange * sinscan
        )
        ** 2
        * sigmaScan**2
    )

    C23p = (
        (
            tfM[1, 0] * cosyaw * sinscan
            + tfM[1, 1] * sinyaw * sinscan  # dY/dRange - measurements
            + tfM[1, 2] * cosscan
        )
        * (
            tfM[2, 0] * cosyaw * sinscan
            + tfM[2, 1] * sinyaw * sinscan  # dZ/dRange - measurements
            + tfM[2, 2] * cosscan
        )
        * sigmaRange**2
        + (
            -1 * tfM[1, 0] * rrange * sinyaw * sinscan
            + tfM[1, 1] * rrange * cosyaw * sinscan  # dY/dYaw
        )
        * (
            -1 * tfM[2, 0] * rrange * sinyaw * sinscan
            + tfM[2, 1] * rrange * cosyaw * sinscan  # dZ/dYaw
        )
        * sigmaYaw**2
        + (
            tfM[2, 0] * rrange * cosyaw * cosscan
            + tfM[2, 1] * rrange * sinyaw * cosscan  # dZ/dScan
            + -1 * tfM[2, 2] * rrange * sinscan
        )
        * (
            tfM[1, 0] * rrange * cosyaw * cosscan
            + tfM[1, 1] * rrange * sinyaw * cosscan  # dY/dScan
            + -1 * tfM[1, 2] * rrange * sinscan
        )
        * sigmaScan**2
    )

    C33p = (
        (
            tfM[2, 0] * cosyaw * sinscan
            + tfM[2, 1] * sinyaw * sinscan  # dZ/dRange - measurements
            + tfM[2, 2] * cosscan
        )
        ** 2
        * sigmaRange**2
        + (
            -1 * tfM[2, 0] * rrange * sinyaw * sinscan
            + tfM[2, 1] * rrange * cosyaw * sinscan  # dZ/dYaw
        )
        ** 2
        * sigmaYaw**2
        + (
            tfM[2, 0] * rrange * cosyaw * cosscan
            + tfM[2, 1] * rrange * sinyaw * cosscan  # dZ/dScan
            + -1 * tfM[2, 2] * rrange * sinscan
        )
        ** 2
        * sigmaScan**2
    )

    C13p = (
        (
            tfM[2, 0] * cosyaw * sinscan
            + tfM[2, 1] * sinyaw * sinscan  # dZ/dRange - measurements
            + tfM[2, 2] * cosscan
        )
        * (
            tfM[0, 0] * cosyaw * sinscan
            + tfM[0, 1] * sinyaw * sinscan  # dX/dRange - measurements
            + tfM[0, 2] * cosscan
        )
        * sigmaRange**2
        + (
            -1 * tfM[2, 0] * rrange * sinyaw * sinscan
            + tfM[2, 1] * rrange * cosyaw * sinscan  # dZ/dYaw
        )
        * (
            -1 * tfM[0, 0] * rrange * sinyaw * sinscan
            + tfM[0, 1] * rrange * cosyaw * sinscan  # dX/dYaw
        )
        * sigmaYaw**2
        + (
            tfM[2, 0] * rrange * cosyaw * cosscan
            + tfM[2, 1] * rrange * sinyaw * cosscan  # dZ/dScan
            + -1 * tfM[2, 2] * rrange * sinscan
        )
        * (
            tfM[0, 0] * rrange * cosyaw * cosscan
            + tfM[0, 1] * rrange * sinyaw * cosscan  # dX/dScan
            + -1 * tfM[0, 2] * rrange * sinscan
        )
        * sigmaScan**2
    )
    local_Cxx[0, 0] += np.sum(C11p)
    local_Cxx[0, 1] += np.sum(C12p)
    local_Cxx[0, 2] += np.sum(C13p)
    local_Cxx[1, 0] += np.sum(C12p)
    local_Cxx[1, 1] += np.sum(C22p)
    local_Cxx[1, 2] += np.sum(C23p)
    local_Cxx[2, 1] += np.sum(C23p)
    local_Cxx[2, 0] += np.sum(C13p)
    local_Cxx[2, 2] += np.sum(C33p)

    # Get mean without correlation (averages out anyway, or something...)
    for pii in range(nPts):
        Cxx = np.array(
            [
                [C11p[pii], C12p[pii], C13p[pii]],
                [C12p[pii], C22p[pii], C23p[pii]],
                [C13p[pii], C23p[pii], C33p[pii]],
            ]
        )
        if np.linalg.det(Cxx) == 0:
            Cxx = np.eye(3)
        Cix = np.linalg.inv(Cxx)
        ATP[:, pii * 3 : (pii + 1) * 3] = Cix
    N = np.dot(ATP, A)
    Qxx = np.linalg.inv(N)  # can only have > 0 in main diagonal!
    pts_m = curr_pts.mean(axis=0)
    l = (curr_pts - pts_m).flatten()
    mean = np.dot(Qxx, np.dot(ATP, l)) + pts_m

    return mean, local_Cxx / nPts


def process_corepoint_list(
    corepoints,
    corepoint_normals,
    p1_idx,
    p1_shm_name,
    p1_size,
    p1_positions,
    p2_idx,
    p2_shm_name,
    p2_size,
    p2_positions,
    M3C2Meta,
    idx,
    return_dict,
    pbarQueue,
):
    pbarQueue.put((0, 1))
    p1_shm = mp.shared_memory.SharedMemory(name=p1_shm_name)
    p2_shm = mp.shared_memory.SharedMemory(name=p2_shm_name)
    p1_coords = np.ndarray(p1_size, dtype=np.float64, buffer=p1_shm.buf)
    p2_coords = np.ndarray(p2_size, dtype=np.float64, buffer=p2_shm.buf)

    max_dist = M3C2Meta["maxdist"]
    search_radius = M3C2Meta["searchrad"]

    M3C2_vals = np.full((corepoints.shape[0]), np.nan, dtype=np.float64)
    M3C2_LoD = np.full((corepoints.shape[0]), np.nan, dtype=np.float64)
    M3C2_N1 = np.full((corepoints.shape[0]), np.nan, dtype=np.int32)
    M3C2_N2 = np.full((corepoints.shape[0]), np.nan, dtype=np.int32)

    M3C2_spread1 = np.full((corepoints.shape[0]), np.nan, dtype=np.float64)
    M3C2_spread2 = np.full((corepoints.shape[0]), np.nan, dtype=np.float64)
    M3C2_cov1 = np.full((corepoints.shape[0], 3, 3), np.nan, dtype=np.float64)
    M3C2_cov2 = np.full((corepoints.shape[0], 3, 3), np.nan, dtype=np.float64)

    for cp_idx, p1_neighbours in enumerate(p1_idx):
        n = corepoint_normals[cp_idx]
        p1_curr_pts = p1_coords[p1_neighbours, :]
        along1, acrossSq1 = getAlongAcrossSqBatch(p1_curr_pts.T, corepoints[cp_idx], n)
        p1_curr_pts = p1_curr_pts[
            np.logical_and(np.abs(along1) <= max_dist, acrossSq1 <= search_radius**2),
            :,
        ]
        p1_scanPos = p1_positions[p1_neighbours]
        p1_scanPos = p1_scanPos[
            np.logical_and(np.abs(along1) <= max_dist, acrossSq1 <= search_radius**2)
        ]
        if p1_curr_pts.shape[0] < M3C2Meta["minneigh"]:
            pbarQueue.put((1, 0))  # point processed
            M3C2_N1[cp_idx] = p1_curr_pts.shape[0]
            continue
        elif p1_curr_pts.shape[0] > M3C2Meta["maxneigh"]:
            p1_curr_pts = p1_curr_pts[np.argsort(acrossSq1[: M3C2Meta["maxneigh"]])]
            p1_scanPos = p1_scanPos[np.argsort(acrossSq1[: M3C2Meta["maxneigh"]])]

        Cxx = M3C2Meta["Cxx"]
        tfM = M3C2Meta["tfM"]
        origins = np.array([SP["origin"] for SP in M3C2Meta["spInfos"][0]])
        redPoint = M3C2Meta["redPoint"]
        sigmas = np.array(
            [
                [
                    SP["sigma_range"],
                    SP["sigma_range"],
                    SP["sigma_scan"],
                    SP["sigma_yaw"],
                ]
                for SP in M3C2Meta["spInfos"][0]
            ]
        )

        p1_weighted_CoG, p1_local_Cxx = get_local_mean_and_Cxx_nocorr(
            Cxx,
            tfM,
            origins,
            redPoint,
            sigmas,
            p1_curr_pts,
            p1_scanPos,
            epoch=0,
            tf=False,
        )  # only one dataset has been transformed
        along1_var = np.var(
            along1[
                np.logical_and(
                    np.abs(along1) <= max_dist, acrossSq1 <= search_radius**2
                )
            ]
        )

        p2_neighbours = p2_idx[cp_idx]
        p2_curr_pts = p2_coords[p2_neighbours, :]
        along2, acrossSq2 = getAlongAcrossSqBatch(p2_curr_pts.T, corepoints[cp_idx], n)
        p2_curr_pts = p2_curr_pts[
            np.logical_and(np.abs(along2) <= max_dist, acrossSq2 <= search_radius**2),
            :,
        ]
        p2_scanPos = p2_positions[p2_neighbours]
        p2_scanPos = p2_scanPos[
            np.logical_and(np.abs(along2) <= max_dist, acrossSq2 <= search_radius**2)
        ]
        if p2_curr_pts.shape[0] < M3C2Meta["minneigh"]:
            pbarQueue.put((1, 0))  # point processed
            M3C2_N2[cp_idx] = p2_curr_pts.shape[0]
            continue
        elif p2_curr_pts.shape[0] > M3C2Meta["maxneigh"]:
            p2_curr_pts = p2_curr_pts[np.argsort(acrossSq2[: M3C2Meta["maxneigh"]])]
            p2_scanPos = p2_scanPos[np.argsort(acrossSq2[: M3C2Meta["maxneigh"]])]

        origins = np.array([SP["origin"] for SP in M3C2Meta["spInfos"][1]])
        sigmas = np.array(
            [
                [
                    SP["sigma_range"],
                    SP["sigma_range"],
                    SP["sigma_scan"],
                    SP["sigma_yaw"],
                ]
                for SP in M3C2Meta["spInfos"][1]
            ]
        )
        p2_weighted_CoG, p2_local_Cxx = get_local_mean_and_Cxx_nocorr(
            Cxx,
            tfM,
            origins,
            redPoint,
            sigmas,
            p2_curr_pts,
            p2_scanPos,
            epoch=1,
            tf=True,
        )
        along2_var = np.var(
            along2[
                np.logical_and(
                    np.abs(along2) <= max_dist, acrossSq2 <= search_radius**2
                )
            ]
        )

        p1_CoG = p1_weighted_CoG
        p2_CoG = p2_weighted_CoG

        p1_CoG_Cxx = p1_local_Cxx
        p2_CoG_Cxx = p2_local_Cxx

        p1_p2_CoG_Cxx = np.zeros((6, 6))
        p1_p2_CoG_Cxx[0:3, 0:3] = p1_CoG_Cxx
        p1_p2_CoG_Cxx[3:6, 3:6] = p2_CoG_Cxx

        M3C2_dist = n.dot(p1_CoG - p2_CoG)
        F = np.hstack([-n, n])

        M3C2_vals[cp_idx] = M3C2_dist

        N1 = p1_curr_pts.shape[0]
        N2 = p2_curr_pts.shape[0]

        sigmaD = p1_CoG_Cxx + p2_CoG_Cxx

        p = 3  # three dimensional
        Tsqalt = n.T.dot(np.linalg.inv(sigmaD)).dot(n)

        M3C2_LoD[cp_idx] = np.sqrt(sstats.chi2.ppf(0.95, p) / Tsqalt)
        M3C2_N1[cp_idx] = N1
        M3C2_N2[cp_idx] = N2

        # add M3C2 spreads
        normal = n[np.newaxis, :]
        M3C2_spread1[cp_idx] = np.sqrt(
            np.matmul(np.matmul(normal, p1_CoG_Cxx), normal.T)
        )
        M3C2_spread2[cp_idx] = np.sqrt(
            np.matmul(np.matmul(normal, p2_CoG_Cxx), normal.T)
        )
        M3C2_cov1[cp_idx] = p1_CoG_Cxx
        M3C2_cov2[cp_idx] = p2_CoG_Cxx

        pbarQueue.put((1, 0))  # point processed

    return_dict[idx] = {
        "lod_new": M3C2_LoD,
        "val": M3C2_vals,
        "m3c2_n1": M3C2_N1,
        "m3c2_n2": M3C2_N2,
        "m3c2_spread1": M3C2_spread1,
        "m3c2_spread2": M3C2_spread2,
        "m3c2_cov1": M3C2_cov1,
        "m3c2_cov2": M3C2_cov2,
    }
    pbarQueue.put((0, -1))
    p1_shm.close()
    p2_shm.close()


def radius_search(epoch: Epoch, query: np.ndarray, radius: float):
    """Query the tree for neighbors within a radius r
    :param query:
        An array of points to query.
        Array-like of shape (n_samples, 3) or query 1 sample point of shape (3,)
    :type query: array
    :param radius:
        Rebuild the search tree even if it was already built before.
    :type radius: float
    """
    if len(query.shape) == 1 and query.shape[0] == 3:
        return [epoch._radius_search(query, radius)]

    if len(query.shape) == 2 and query.shape[1] == 3:
        neighbors = []
        for i in range(query.shape[0]):
            q = query[i]
            result = epoch._radius_search(q, radius)
            neighbors.append(result)
        return neighbors

    raise Py4DGeoError(
        "Please ensure queries are array-like of shape (n_samples, 3)"
        " or of shape (3,) to query 1 sample point!"
    )

    return None
