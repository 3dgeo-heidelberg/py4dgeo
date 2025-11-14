# This is a version of VAPC (https://github.com/3dgeo-heidelberg/vapc.git) that has been developed 
# specifically for Py4DGeo. Functions relevant # to change analysis, such as the hierarchical 
# analysis and subsampling of point clouds, have # been transferred. Other functions, such as 
# the calculation of geometric features, have been partially transferred, but are currently 
# not tested as they are not the focus of this implementation. Also, the Py4DGeo implementation 
# is more efficient than the original implementation. Further functionalities can be incorporated 
# if there is demand. The Octree implementation of Py4DGeo is only partially implemented here. 
# Due to the slower computation times observed in some initial tests not every function has been 
# adapted to use it yet (if implemented in a later stage, we may need to change the max possible 
# octree depth).

import numpy as np
import laspy
from py4dgeo.epoch import Epoch
from functools import wraps
import time
import psutil

DECORATOR_CONFIG = {"trace": True, "timeit": True}

def enable_trace(enable=True):
    """
    Enables or disables the trace decorator.

    Parameters
    ----------
    enable : bool, optional
        If True, tracing is enabled. If False, tracing is disabled.
    """
    DECORATOR_CONFIG["trace"] = enable

def enable_timeit(enable=True):
    """
    Enables or disables the timeit decorator.

    Parameters
    ----------
    enable : bool, optional
        If True, timing is enabled. If False, timing is disabled.
    """
    DECORATOR_CONFIG["timeit"] = enable

def trace(func):
    """
    A decorator that prints the name of the function before it is called. Controlled by the DECORATOR_CONFIG["trace"] flag.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if DECORATOR_CONFIG["trace"]:
            print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

def timeit(func):
    """
    A decorator that measures and prints the execution time of a function.Controlled by the DECORATOR_CONFIG["timeit"] flag.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if DECORATOR_CONFIG["timeit"]:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
            return result
        else:
            return func(*args, **kwargs)
    return wrapper

try:
    import numba
    NUMBA_AVAILABLE = True
    prange = numba.prange  # safe alias
except Exception:
    print("Numba is not installed.")
    print("Use `pip install numba` to install it to enable faster computations in Vapc.")
    print("You can run py4dgeo.Vapc without Numba, but it may be slower.")
    NUMBA_AVAILABLE = False

if NUMBA_AVAILABLE:
    @numba.njit(parallel=True, cache=True)
    def _argmin_per_group(inverse, d2):
        inverse = inverse.astype(np.int64)
        M = inverse.max() + 1
        N = d2.size

        # counts per group
        counts = np.bincount(inverse, minlength=M).astype(np.int64)

        # prefix sum → group offsets
        offsets = np.empty(M + 1, np.int64)
        offsets[0] = 0
        for g in range(M):
            offsets[g + 1] = offsets[g] + counts[g]

        # scatter indices into per-group buckets
        perm = np.empty(N, np.int64)
        write_pos = offsets[:-1].copy()  # per-group cursor (serial to avoid races)
        for i in range(N):
            g = inverse[i]
            j = write_pos[g]
            perm[j] = i
            write_pos[g] = j + 1

        # parallel per-group scan for the argmin
        best_idx = np.full(M, -1, np.int64)
        for g in numba.prange(M):
            start = offsets[g]
            end   = offsets[g + 1]
            if start < end:
                b = perm[start]
                bv = d2[b]
                for k in range(start + 1, end):
                    i = perm[k]
                    v = d2[i]
                    if v < bv:
                        b = i
                        bv = v
                best_idx[g] = b
        return best_idx
else:
    def _argmin_per_group(inverse, d2):
        """
        Pure-NumPy sort by (group, value) and take the first index per group.
        """
        inverse = np.asarray(inverse, dtype=np.int64)
        d2 = np.asarray(d2)
        # sort by group, then by value
        order = np.lexsort((d2, inverse))
        inv_sorted = inverse[order]
        # first row for each new group
        first = np.empty(inv_sorted.size, dtype=bool)
        first[0] = True
        first[1:] = inv_sorted[1:] != inv_sorted[:-1]
        return order[first]

class Vapc:
    def __init__(self, epoch: Epoch, voxel_size: float, origin: list = None):
        """
        epoch      : py4dgeo.Epoch containing your points in epoch._cloud (Nx3 numpy) and optional extra dimensions
        voxel_size : edge length of the  voxel
        origin     : [x0,y0,z0] offset of your voxel grid (default [0,0,0])
        """
        self.epoch = epoch
        self.voxel_size = float(voxel_size)
        self.origin = np.array(origin or [0.0, 0.0, 0.0], dtype=float)

        self.extra_dims = None
        if hasattr(self.epoch, "additional_dimensions"):
            self.extra_dims = self.epoch.additional_dimensions

        # self.use_octree = False  # use octree for grouping, OFF by default 

        self.voxel_indices = None
        self.unique_voxels = None
        self.inverse = None
        self.counts = None
        self.centroids = None
        self.covariance = None
        self.eigenvalues = None
        self.voxel_centers = None
        self.closest_to_voxel_centers = None
        self.closest_to_centroids = None


        self.out   = {}
        self.mapped = False

        self.delta = None

        # map feature‐names to computations
        self.AVAILABLE_COMPUTATIONS = {
            "count":          self._compute_count,
            "density":        self._compute_density,
            "centroid":       self.compute_centroids,
            "voxel_center":   self.compute_voxel_centers,
            "covariance":     self.compute_covariance,
            "closest_to_voxel_centers":  self.compute_closest_to_voxel_centers,
            "closest_to_centroids":      self.compute_closest_to_centroids,

            #Implemented but not relevant for change analysis tasks at the moment:
            "eigenvalues":    self.compute_eigenvalues,
            "linearity":      self._compute_linearity,
            "planarity":      self._compute_planarity,
            "sphericity":     self._compute_sphericity,
            "sum_of_eigenvalues":        self._compute_sum_of_eigenvalues,
            "omnivariance":              self._compute_omnivariance,
            "eigenentropy":              self._compute_eigenentropy,
            "anisotropy":                self._compute_anisotropy,
            "surface_variation":         self._compute_surface_variation,
        }

    ############### Octree methods for until grouping ####################################
    # @timeit
    # @trace
    # def _ensure_octree_with_voxel_size(self,min_corner=None, max_corner=None):
    #     if not hasattr(self, "octree"):
    #         self._max_depth = 10  # default max depth for octree
    #         self._level_of_interest = self._max_depth
    #         # Ensure the octree is built to get a specific cell size (voxel size) at the deepest level
    #         bbox_extent = self.voxel_size * (2 **  self._level_of_interest)

    #         print("Bounding box extent for cell size %s at level %s: %s"%(self.voxel_size, self._level_of_interest, bbox_extent))
    #         if min_corner is None:
    #             min_corner = self.epoch.cloud.min(axis=0)

    #         self._octree_origin = min_corner.astype(float)
    #         max_corner = min_corner + np.array([bbox_extent, bbox_extent, bbox_extent])  # Adjusted to match the octree size
    #         print("\nDifference between min and max corner:", max_corner - min_corner)
    #         self.epoch._octree.build_tree(force_cubic=True, min_corner=min_corner, max_corner=max_corner)
    #         self.octree = self.epoch._octree

    # @timeit
    # @trace
    # def compute_octree_keys(self, level: int = None) -> np.ndarray:
    #     """
    #     Returns a length‐N array of integer “cell keys” for each point.
    #     If level is None, these are full‐depth (Morton) keys.
    #     If level is given, keys are right‐shifted so they index the
    #     level’s 2^level grid.
    #     """
    #     self._ensure_octree_with_voxel_size()
        
    #     # Get the raw (sorted) Z‐order keys + the mapping back to point indices:
    #     sorted_keys    = np.array(self.octree.get_spatial_keys(),    dtype=np.uint32)
    #     sorted_indices = np.array(self.octree.get_point_indices(),   dtype=np.int32)

    #     # Reassemble `keys_full[i]` = Morton‐key of point i
    #     keys_full = np.empty_like(sorted_keys)
    #     keys_full[sorted_indices] = sorted_keys

    #     if level is None:
    #         return keys_full

    #     # Truncate to the desired level:
    #     #   bit_shift = 3*(max_depth − level)
    #     shift = 3 * (self._max_depth - level)
    #     return keys_full >> shift
    
    # @timeit
    # @trace
    # def group_octree_cells(self, level: int = None):
    #     """
    #     Groups points by their Octree cell at the given level.
    #     Populates:
    #       • self.octree_keys      – length‐N array of keys per point  
    #       • self.unique_cells     – sorted array of unique cell‐keys  
    #       • self.inverse          – length‐N array, giving 0…M-1 cell‐ID for each point  
    #       • self.counts           – length‐M array of point counts per cell
    #     """
    #     keys       = self.compute_octree_keys(level)
    #     unique, inv, counts = np.unique(keys,
    #                                     return_inverse=True,
    #                                     return_counts=True)

    #     self.octree_keys  = keys
    #     self.unique_voxels = unique
    #     self.inverse      = inv
    #     self.counts       = counts
    #     return unique, inv, counts

    
    ############### Voxel methods for grouping ####################################
    @timeit
    @trace
    def compute_voxel_indices(self) -> np.ndarray:
        """
        Compute integer (i,j,k) voxel indices for each point in epoch._cloud.
        Returns an (Nx3) int array.
        """
        # grab the raw point‐cloud as an (Nx3) numpy array
        coords = np.asarray(self.epoch.cloud)
        # shift by origin, scale by voxel_size, floor to get integer voxel coords
        self.voxel_indices = np.floor((coords - self.origin) 
                                      / self.voxel_size).astype(int)
        return self.voxel_indices

    @timeit
    @trace
    def group_voxels(self):
        if self.voxel_indices is None:
            self.compute_voxel_indices()

        vox = self.voxel_indices
        i_min, j_min, k_min = vox.min(axis=0)
        i_max, j_max, k_max = vox.max(axis=0)
        I, J, K = i_max - i_min + 1, j_max - j_min + 1, k_max - k_min + 1
        shifted = vox - np.array([i_min, j_min, k_min])
        keys = np.ravel_multi_index(shifted.T, dims=(I, J, K), order="C").astype(np.int64)

        # --- Estimate memory need ---
        max_key = keys.max()
        estimated_memory = (max_key + 1) * np.dtype(np.int64).itemsize
        available_mem = psutil.virtual_memory().available

        if estimated_memory < available_mem * 0.5:
            # Fast path: use bincount
            counts_all = np.bincount(keys, minlength=(max_key + 1))
            unique_keys = np.nonzero(counts_all)[0]
            counts = counts_all[unique_keys]
            lookup = np.empty(max_key + 1, dtype=np.intp)
            lookup[unique_keys] = np.arange(unique_keys.size)
            inverse = lookup[keys]
        else:
            # Memory-safe fallback
            print(f"Not enough memory for np.bincount, falling back to using np.unique...")
            unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)

        i, j, k = np.unravel_index(unique_keys, (I, J, K), order="C")
        i += i_min
        j += j_min
        k += k_min
        unique_arr = np.stack([i, j, k], axis=1)

        self.unique_voxels = unique_arr
        self.inverse = inverse
        self.counts = counts
        return unique_arr, inverse, counts

    def group(self):
        """
        Group points into voxels (or octree cells -> currently disabled).
        """
        # if self.use_octree:
        #     return self.group_octree_cells()
        # else:
        return self.group_voxels()

    ############### Computation methods ####################################
    @timeit
    @trace
    def compute_voxel_centers(self) -> np.ndarray:
        """
        Compute the center of each *unique* voxel (one per voxel, Mx3).
        """
        if self.unique_voxels is None:
            self.group()

        # handle octree vs voxel grouping
        # if self.use_octree:
        #     # decode Morton keys into 3D voxel indices
        #     ijk = decode_morton(self.unique_voxels, self._max_depth)
        #     origin = self._octree_origin
        # else:
        ijk = self.unique_voxels
        origin = self.origin
        # real‐world center: origin + (i,j,k)*size + half‐voxel
        centers = ijk * self.voxel_size + origin + self.voxel_size / 2.0

        self.voxel_centers = centers   # cache for later
        return centers                 # length M

    @timeit
    @trace
    def compute_centroids(self):
        if self.inverse is None or self.counts is None:
            self.group()

        coords = np.asarray(self.epoch._cloud)
        inv    = self.inverse
        M      = self.unique_voxels.shape[0]
        counts = self.counts.astype(float)

        sum_x = np.bincount(inv, weights=coords[:,0], minlength=M)
        sum_y = np.bincount(inv, weights=coords[:,1], minlength=M)
        sum_z = np.bincount(inv, weights=coords[:,2], minlength=M)

        self.centroids = np.stack([sum_x, sum_y, sum_z], axis=1) / counts[:,None]
        return self.centroids
        
    @timeit
    @trace
    def compute_closest_to_voxel_centers(self): #using numba turned out to be the fastest
        """
        Compute the closest point to the center of each voxel using Numba.
        """
        # ensure voxel-centers exist
        if self.voxel_centers is None:
            self.compute_voxel_centers()
        if self.inverse is None or self.unique_voxels is None:
            self.group()

        coords  = np.asarray(self.epoch._cloud)
        inv     = self.inverse
        centers = self.voxel_centers[inv]
        d2      = np.sum((coords - centers)**2, axis=1)

        idx = _argmin_per_group(inv, d2)

        closest = coords[idx]
        self.closest_to_voxel_centers = closest
        return closest

    @timeit
    @trace
    def compute_closest_to_centroids(self):
        """
        Compute the closest point to the centroid of each voxel
        using the existing _argmin_per_group implementation.
        """
        if self.centroids is None:
            self.compute_centroids()

        coords  = np.asarray(self.epoch._cloud)
        inv     = self.inverse
        # build per-point centroid lookup
        centers = self.centroids[inv]

        # squared distances of each point to its voxel centroid
        d2      = np.sum((coords - centers)**2, axis=1)

        idx = _argmin_per_group(inv, d2)

        # grab the actual points
        self.closest_to_centroids = coords[idx]
        return self.closest_to_centroids
    
    # -- helper methods ----------------------------------------------
    @timeit
    @trace
    def _compute_count(self):
        if self.counts is None:
            self.group()
        return self.counts

    @timeit
    @trace
    def _compute_density(self):
        # points per voxel volume
        return self._compute_count() / (self.voxel_size ** 3)

    @timeit
    @trace
    def compute_covariance(self):
        """
        Covariance per voxel using np.bincount.
        """
        if self.inverse is None:
            self.group()

        coords = np.asarray(self.epoch._cloud)
        inv    = self.inverse
        M      = self.unique_voxels.shape[0]
        counts = self.counts.astype(float)

        # Sums of each coordinate
        sum_x = np.bincount(inv, weights=coords[:, 0], minlength=M)
        sum_y = np.bincount(inv, weights=coords[:, 1], minlength=M)
        sum_z = np.bincount(inv, weights=coords[:, 2], minlength=M)

        # Sums of squares and cross-terms
        sum_xx = np.bincount(inv, weights=coords[:, 0] * coords[:, 0], minlength=M)
        sum_yy = np.bincount(inv, weights=coords[:, 1] * coords[:, 1], minlength=M)
        sum_zz = np.bincount(inv, weights=coords[:, 2] * coords[:, 2], minlength=M)
        sum_xy = np.bincount(inv, weights=coords[:, 0] * coords[:, 1], minlength=M)
        sum_xz = np.bincount(inv, weights=coords[:, 0] * coords[:, 2], minlength=M)
        sum_yz = np.bincount(inv, weights=coords[:, 1] * coords[:, 2], minlength=M)

        # Means
        mean_x = sum_x / counts
        mean_y = sum_y / counts
        mean_z = sum_z / counts

        # Allocate result and fill
        covs = np.empty((M, 3, 3), dtype=float)
        covs[:, 0, 0] = sum_xx / counts - mean_x * mean_x
        covs[:, 1, 1] = sum_yy / counts - mean_y * mean_y
        covs[:, 2, 2] = sum_zz / counts - mean_z * mean_z
        covs[:, 0, 1] = covs[:, 1, 0] = sum_xy / counts - mean_x * mean_y
        covs[:, 0, 2] = covs[:, 2, 0] = sum_xz / counts - mean_x * mean_z
        covs[:, 1, 2] = covs[:, 2, 1] = sum_yz / counts - mean_y * mean_z

        # Clamp any negative variances to exactly zero
        diag = np.diagonal(covs, axis1=1, axis2=2)
        diag_clamped = np.clip(diag, 0.0, None)
        for d in range(3):
            covs[:, d, d] = diag_clamped[:, d]

        self.covariance = covs
        return self.covariance

    @timeit
    @trace
    def compute_eigenvalues(self):
        """
        Compute sorted eigenvalues [λ1 ≥ λ2 ≥ λ3] for each voxel.
        """    
        if self.covariance is None:
                self.compute_covariance()

        covs = self.covariance.astype(np.float64)

        # Compute the eigenvalues of each 3×3 covariance matrix
        eigvals = np.linalg.eigvalsh(covs)

        # Clamp any small negative values (e.g. −1e-12) up to zero
        eigvals[eigvals < 0] = 0.0

        # Sort in descending order so λ₁ ≥ λ₂ ≥ λ₃
        eigvals = np.sort(eigvals, axis=1)[:, ::-1]

        self.eigenvalues = eigvals
        return self.eigenvalues

    @timeit
    @trace
    def _compute_linearity(self):
        """
        Linearity is defined as:
        L = (λ1 - λ2) / λ1
        where λ1 ≥ λ2 ≥ λ3 are the sorted eigenvalues of the covariance matrix.
        http://dx.doi.org/10.1109/CVPR.2016.178
        """
        if self.eigenvalues is None:
            self.compute_eigenvalues()
        λ1 = self.eigenvalues[:, 0]
        λ2 = self.eigenvalues[:, 1]
        λ3 = self.eigenvalues[:, 2]
        # avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            linearity = (λ1 - λ2) / λ1
            linearity[np.isnan(linearity)] = 0
        return linearity

    @timeit
    @trace
    def _compute_planarity(self):
        """
        Planarity is defined as:
        P = (λ2 - λ3) / λ1
        where λ1 ≥ λ2 ≥ λ3 are the sorted eigenvalues of the covariance matrix.
        http://dx.doi.org/10.1109/CVPR.2016.178
        """
        if self.eigenvalues is None:
            self.compute_eigenvalues()
        λ1 = self.eigenvalues[:, 0]
        λ2 = self.eigenvalues[:, 1]
        λ3 = self.eigenvalues[:, 2]
        # avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            planarity = (λ2 - λ3) / λ1
            planarity[np.isnan(planarity)] = 0
        return planarity

    @timeit
    @trace
    def _compute_sphericity(self):
        """
        Sphericity is defined as:
        S = λ3 / λ1
        where λ1 ≥ λ2 ≥ λ3 are the sorted eigenvalues of the covariance matrix.
        http://dx.doi.org/10.1109/CVPR.2016.178
        """
        if self.eigenvalues is None:
            self.compute_eigenvalues()
        λ1 = self.eigenvalues[:, 0]
        λ2 = self.eigenvalues[:, 1]
        λ3 = self.eigenvalues[:, 2]
        # avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            sphericity = λ3 / λ1
            sphericity[np.isnan(sphericity)] = 0
        return sphericity
    
    @trace
    @timeit
    def _compute_sum_of_eigenvalues(self) -> np.ndarray:
        """Σ λᵢ per voxel."""
        if self.eigenvalues is None:
            self.compute_eigenvalues()
        return self.eigenvalues.sum(axis=1)

    @trace
    @timeit
    def _compute_omnivariance(self) -> np.ndarray:
        """(λ1·λ2·λ3)^(1/3) per voxel."""
        if self.eigenvalues is None:
            self.compute_eigenvalues()
        prod = np.prod(self.eigenvalues, axis=1)
        return np.power(prod, 1.0 / 3.0)

    @trace
    @timeit
    def _compute_eigenentropy(self) -> np.ndarray:
        """–∑ pᵢ log pᵢ where pᵢ=λᵢ/∑λ per voxel."""
        if self.eigenvalues is None:
            self.compute_eigenvalues()
        lam = self.eigenvalues
        s   = lam.sum(axis=1, keepdims=True)
        p   = lam / np.where(s > 0, s, 1.0)       
        # safe log: log(0)→0
        logs = np.where(p > 0, np.log(p), 0.0)
        return -np.sum(p * logs, axis=1)

    @trace
    @timeit
    def _compute_anisotropy(self) -> np.ndarray:
        """(λ1 − λ3) / λ1 per voxel."""
        if self.eigenvalues is None:
            self.compute_eigenvalues()
        lam1 = self.eigenvalues[:, 0]
        lam3 = self.eigenvalues[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            a = (lam1 - lam3) / lam1
        # replace NaN (from lam1=0) with zero
        a[np.isnan(a)] = 0.0
        return a

    @trace
    @timeit
    def _compute_surface_variation(self) -> np.ndarray:
        """λ3 / (λ1+λ2+λ3) per voxel."""
        if self.eigenvalues is None:
            self.compute_eigenvalues()
        lam = self.eigenvalues
        s   = lam.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sv = lam[:, 2] / s
        sv[np.isnan(sv)] = 0.0
        return sv

    @timeit
    @trace
    def compute_features(self, feature_names: list):
        """
        Compute any subset of AVAILABLE_COMPUTATIONS.
        Returns a dict: { name → ndarray_for_each_voxel }
        """
        self.out = {}
        for name in feature_names:
            if name not in self.AVAILABLE_COMPUTATIONS:
                raise ValueError(f"Unsupported feature: {name}")
            out = self.AVAILABLE_COMPUTATIONS[name]()
            if out.ndim == 1:
                self.out[name] = out
            elif out.ndim == 2 and out.shape[1] > 1:
                # multi‐dimensional feature, e.g. eigenvalues
                if name == "centroid":
                    keys = ["centroid_x", "centroid_y", "centroid_z"]
                    for i, key in enumerate(keys):
                        self.out[key] = out[:, i]
                elif name == "eigenvalues":
                    keys = ["eigenvalue_1", "eigenvalue_2", "eigenvalue_3"]
                    for i, key in enumerate(keys):
                        self.out[key] = out[:, i]
                elif name == "voxel_center":
                    keys = ["voxel_center_x", "voxel_center_y", "voxel_center_z"]
                    for i, key in enumerate(keys):
                        self.out[key] = out[:, i]
            elif out.ndim == 3:
                if name == "covariance":
                    # flatten each 3x3 to 9 separate arrays
                    keys = [
                        "cov_xx", "cov_xy", "cov_xz",
                        "cov_yx", "cov_yy", "cov_yz",
                        "cov_zx", "cov_zy", "cov_zz",
                    ]
                    M = out.shape[0]
                    for idx, key in enumerate(keys):
                        i, j = divmod(idx, 3)
                        self.out[key] = out[:, i, j]
            else:
                print(f"Warning: feature {name} not handled, skipping.")
                print("Update AVAILABLE_COMPUTATIONS and compute_features() to handle this.")
        return self.out
    
    ##################### 3D selection methods ####################################

    @timeit
    @trace
    def select_by_mask(
        self,
        vapc_mask: "Vapc",
        segment_in_or_out: str = "in",
        overwrite: bool = False
        ) -> "Vapc":
        """
        Return a new Vapc containing only those points of self
        whose voxel is (or is not) occupied in vapc_mask.
        """

        # sanity checks
        if not np.isclose(self.voxel_size, vapc_mask.voxel_size):
            raise ValueError("Voxel sizes differ")
        if not np.allclose(self.origin, vapc_mask.origin):
            raise ValueError("Voxel grid origins differ")

        # ensure both have been grouped
        if self.unique_voxels is None:
            self.group()
        if vapc_mask.unique_voxels is None:
            vapc_mask.group()

        coord2gid = {
            tuple(vox): gid
            for gid, vox in enumerate(self.unique_voxels)
        }

        # map mask’s voxels into my group‐IDs (only those that overlap)
        mask_gids = [
            coord2gid[c]
            for c in map(tuple, vapc_mask.unique_voxels)
            if c in coord2gid
        ]
        if len(mask_gids) == 0:
            # no overlap → empty selection
            sel = np.zeros_like(self.inverse, dtype=bool)
        else:
            mask_gids = np.array(mask_gids, dtype=self.inverse.dtype)
            membership = np.isin(self.inverse, mask_gids)
            sel = membership if segment_in_or_out == "in" else ~membership

        # slice out
        pts = np.asarray(self.epoch._cloud)
        filtered_pts = pts[sel]

        # build new Vapc
        new_epoch = Epoch(cloud=filtered_pts)
        new = Vapc(
            epoch=new_epoch,
            voxel_size=self.voxel_size,
            origin=list(self.origin)
        )

        # new.use_octree = self.use_octree
        if overwrite:
            # overwrite the original Vapc
            self.epoch = new_epoch
            self.out = new.out
            self.mapped = True
            return self
        return new


    #################### Mapping methods ####################################
    
    @timeit
    @trace
    def map_features_to_points(self) -> dict:
        """
        Compute the given voxel‐wise features and map each one
        back onto the original N points.

        Returns:
          mapped: dict[name → ndarray]
            where each ndarray is length N (or NxD for multi‐dim
            features) such that
            mapped[name][i] == feature_value_of_voxel_containing_point_i
        """

        if self.inverse is None:
            self.group()

        # for each feature, index by inverse to get per‐point array
        mapped = {}
        for name, arr in self.out.items():
            mapped[name] = arr[self.inverse]

        # now tack on the original per-point attributes
        if self.extra_dims is not None:
            for dim_name in self.extra_dims.dtype.names:
                mapped[dim_name] = self.extra_dims[dim_name]

        self.out = mapped
        self.mapped = True
        return self.out
    
    ############### Bi-temporal comparison methods ####################
    @timeit
    @trace
    def delta_vapc(self, other: "Vapc"):
        """
        Compare this Vapc to another Vapc (same voxel_size & origin).
        Returns a Vapc whose `out["delta_vapc"]` is a (K,) int array of
        1 = only self, 2 = only other, 3 = both.
        """
        # sanity checks
        if not np.isclose(self.voxel_size, other.voxel_size):
            raise ValueError("Voxel sizes differ")
        if not np.allclose(self.origin, other.origin):
            raise ValueError("Origins differ")

        # ensure both have grouped voxels
        self.group()
        other.group()

        A = self.unique_voxels
        B = other.unique_voxels

        # reinterpret each [i,j,k] as a single struct to unique+label in one go
        dtype = np.dtype([("x", A.dtype), ("y", A.dtype), ("z", A.dtype)])
        A_struct = A.view(dtype).ravel()
        B_struct = B.view(dtype).ravel()

        pts_all = np.concatenate([A_struct, B_struct])
        labels_all = np.concatenate([
            np.ones(len(A_struct), dtype=np.int8),    # label=1 for self
            np.full(len(B_struct), 2, dtype=np.int8)  # label=2 for other
        ])

        unique_pts, inv_idx = np.unique(pts_all, return_inverse=True)
        summed = np.bincount(inv_idx, weights=labels_all)

        labels = summed.astype(np.int8)

        U_pts = unique_pts.view(A.dtype).reshape(-1, 3)

        xyz = U_pts * self.voxel_size + self.origin + self.voxel_size/2

        delta_epoch = Epoch(cloud=xyz)
        dv = Vapc(delta_epoch,
                  voxel_size=self.voxel_size,
                  origin=list(self.origin))

        dv.unique_voxels = U_pts
        dv.out            = {"delta_vapc": labels}
        dv.mapped         = True
        return dv

    @timeit
    @trace
    def compute_bitemporal_mahalanobis(
        self,
        other: "Vapc",
        alpha: float = 0.999,
        min_points: int = 30
    ):
        """
        Compute per-voxel Mahalanobis distance between this Vapc
        and another Vapc (same voxel_size & origin).
        
        Returns a Vapc whose `out` dict contains:
          - "change_type"  : int array (K,) with 1=only self, 2=only other, 3=both (= shared voxels)
          - "mahalanobis"  : float array (K,) Mahalanobis distance on shared voxels where min points are met, NaN else
          - "p_value"      : float array (K,) p-value of Mahalanobis test on shared voxels, NaN else
          - "significance" : int array (K,) 1=significant change on shared voxels, 0 else
          - "changed"      : int array (K,) 1=changed voxel (shared voxel with significant change, or 
          change_type is 1, or change_type is 2, or shared voxel but min points not met ), 0=unchanged 
          voxel (shared voxel with no significant change and min points met)
        where K = total number of unique voxels across both Vapcs.
        """
        from scipy.stats import chi2

        # get union & labels
        delta   = self.delta_vapc(other)
        labels  = delta.out["delta_vapc"]        # (K_total,)
        centers = delta.epoch.cloud              # (K_total,3)
        K       = len(labels)

        # build lookups for shared voxels
        map1 = {tuple(v): i for i, v in enumerate(self.unique_voxels)}
        map2 = {tuple(v): i for i, v in enumerate(other.unique_voxels)}
        shared_mask = (labels == 3)
        shared_idx  = np.nonzero(shared_mask)[0]
        shared_uv   = delta.unique_voxels[shared_mask]
        idx1 = np.array([map1[tuple(v)] for v in shared_uv], dtype=int)
        idx2 = np.array([map2[tuple(v)] for v in shared_uv], dtype=int)

        # get per-shared sample counts
        cnt1 = self.counts[idx1]
        cnt2 = other.counts[idx2]

        # centroids & covariances
        C1 = self.compute_centroids()
        C2 = other.compute_centroids()
        S1 = self.compute_covariance()
        S2 = other.compute_covariance()

        # Mahalanobis on shared
        diff = C1[idx1] - C2[idx2]  # (N_shared,3)

        S1s = 0.5*(S1[idx1] + S1[idx1].transpose((0,2,1)))
        S2s = 0.5*(S2[idx2] + S2[idx2].transpose((0,2,1)))
        tr1 = np.trace(S1s, axis1=1, axis2=2)
        tr2 = np.trace(S2s, axis1=1, axis2=2)
        eps1 = (tr1 + 1e-6)*1e-6
        eps2 = (tr2 + 1e-6)*1e-6
        I3   = np.eye(3)
        S1s += eps1[:,None,None]*I3
        S2s += eps2[:,None,None]*I3

        # invert
        inv1 = np.linalg.pinv(S1s)
        inv2 = np.linalg.pinv(S2s)

        # distances
        d1s = np.einsum("ij,ijk,ik->i", diff, inv2, diff)
        d2s = np.einsum("ij,ijk,ik->i", diff, inv1, diff)
        d1s, d2s = np.clip(d1s, 0, None), np.clip(d2s, 0, None)

        # p-values & significance
        p1s  = 1 - chi2.cdf(d1s, df=3)
        p2s  = 1 - chi2.cdf(d2s, df=3)
        sigs = ((p1s < alpha) | (p2s < alpha)).astype(int)

        # allocate full-length outputs
        mahal = np.full(K, np.nan)
        pval  = np.full(K, np.nan)
        sig   = np.zeros(K, dtype=int)
        chg   = np.zeros(K, dtype=int)

        # fill shared slots
        mahal[shared_idx] = np.maximum(d1s, d2s)
        pval[shared_idx]  = np.maximum(p1s, p2s)
        sig[shared_idx]   = sigs

        # compute "changed" mask
        # occupancy change
        chg[labels != 3] = 1
        # shared but too few points
        underpop = ((cnt1 < min_points) | (cnt2 < min_points))
        chg[shared_idx[underpop]] = 1
        # shared & statistically significant
        chg[shared_idx[sigs == 1]] = 1

        # Create output Vapc
        v = Vapc(Epoch(cloud=centers),
                 voxel_size=self.voxel_size,
                 origin=list(self.origin))
        v.unique_voxels = delta.unique_voxels
        v.out = {
            "change_type":   labels,
            "mahalanobis":   mahal,
            "p_value":       pval,
            "significance":  sig,
            "changed":       chg,
        }
        v.mapped = True
        return v
    
    ############### Feature reduction methods ####################

    @timeit
    @trace
    def reduce_to_feature(self, feature_name: str) -> "Vapc":
        """
        Build a new Vapc with exactly one point per voxel, chosen or synthesized
        according to `feature_name`, and carry over both original extra-dims
        (for “closest_to_…” cases) or their per-voxel means (for synthetic points),
        as well as all previously computed voxel-wise features.

        Supported feature_name’s:
          - "closest_to_centroid"
          - "closest_to_voxel_centers"
          - "centroid"
          - "voxel_center"
        """
        # ensure grouped voxels
        if self.inverse is None or self.counts is None:
            self.group()
        inv = self.inverse
        vs  = self.voxel_size
        org = np.array(self.origin, dtype=float)
        coords_orig = np.asarray(self.epoch._cloud)
        
        idx = None                      # representative indices if we pick real points
        N = coords_orig.shape[0]        # original point count
        M = self.unique_voxels.shape[0] # voxel count

        
        new_extra = None


        # select one point per voxel
        if feature_name == "closest_to_centroid":
            centroids = self.compute_centroids()
            centers_pp = centroids[inv]
            d2 = np.sum((coords_orig - centers_pp)**2, axis=1)
            idx = _argmin_per_group(inv, d2)
            coords = coords_orig[idx]
            if self.extra_dims is not None:
                new_extra = self.extra_dims[idx]

        elif feature_name == "closest_to_voxel_centers":
            ijk     = self.unique_voxels
            centers = ijk * vs + org + vs/2
            centers_pp = centers[inv]
            d2 = np.sum((coords_orig - centers_pp)**2, axis=1)
            idx = _argmin_per_group(inv, d2)
            coords = coords_orig[idx]
            if self.extra_dims is not None:
                new_extra = self.extra_dims[idx]

        elif feature_name == "centroid":
            # synthetic centroid points
            centroids = self.compute_centroids()  # M×3
            coords = centroids
            # average every extra_dim per voxel
            if self.extra_dims is not None:
                dtype = self.extra_dims.dtype
                M     = centroids.shape[0]
                new_extra = np.zeros(M, dtype=dtype)
                for nm in dtype.names:
                    vals = np.asarray(self.extra_dims[nm]).squeeze()  # ensure 1-D
                    sums = np.bincount(inv, weights=vals, minlength=M)
                    new_extra[nm] = sums / self.counts

        elif feature_name == "voxel_center":
            # synthetic grid centers
            ijk     = self.unique_voxels
            coords = ijk * vs + org + vs/2
            if self.extra_dims is not None:
                dtype = self.extra_dims.dtype
                M     = coords.shape[0]
                new_extra = np.zeros(M, dtype=dtype)
                for nm in dtype.names:
                    vals = np.asarray(self.extra_dims[nm]).squeeze()  # ensure 1-D
                    sums = np.bincount(inv, weights=vals, minlength=M)
                    new_extra[nm] = sums / self.counts

        else:
            raise ValueError(f"Unknown reduction feature: {feature_name!r}")

        # assemble the new Epoch and Vapc
        new_epoch = Epoch(cloud=coords, additional_dimensions=new_extra)
        new = Vapc(
            epoch=new_epoch,
            voxel_size=self.voxel_size,
            origin=list(self.origin)
        )
        # new.use_octree = self.use_octree
        
        new.extra_dims = new_extra

        # carry over everything into new.out so save_as_las sees it
        new.out = {}

        # expose extra dims as keys too
        if new_extra is not None:
            for nm in new_extra.dtype.names:
                new.out[nm] = new_extra[nm]

        # transform/copy features from self.out → new.out
        for nm, arr in self.out.items():
            # skip coordinates if present
            if nm in ("X", "Y", "Z", "x", "y", "z"):
                continue

            a = np.asarray(arr)

            if a.shape[0] == M:
                # already per-voxel → copy as-is
                new.out[nm] = a

            elif a.shape[0] == N:
                # per-point → reduce to one per voxel
                if idx is not None:
                    # representative-point modes: index by chosen representatives
                    new.out[nm] = a[idx]
                else:
                    # synthetic modes: mean aggregate per voxel
                    sums = np.bincount(inv, weights=a.astype(float), minlength=M)
                    new.out[nm] = sums / self.counts
            else:
                # length mismatch; don’t break the export, just warn
                print(f"Warning: feature '{nm}' has incompatible length {a.shape[0]} "
                    f"(expected {N} per-point or {M} per-voxel). Skipping.")

        new.mapped = True
        return new


    ############### Data Handler methods ####################
    @timeit
    @trace
    def save_as_las(
        self,
        outfile: str,
        las_point_format=7,
        las_version="1.4",
        las_scales=None,
        las_offset=None,
        skip = dict(),
        ):
        """
        Saves the point-cloud + any per-point attributes in self.df to a LAS/LAZ file.

        Parameters
        ----------
        outfile : str
            Path where the LAS or LAZ file will be stored.
        las_point_format : int, optional
            Point format for the LAS file (default is 7).
        las_version : str, optional
            LAS file version (default is "1.4").
        las_scales : list of float, optional
            Scale factors for X, Y, Z (default is [0.00025, 0.00025, 0.00025]).
        las_offset : list of float, optional
            Offset values for X, Y, Z (default is [X.min(), Y.min(), Z.min()]).
        skip : dict, optional
            Dictionary of column names to skip when writing to the LAS file.
        """
        dtype_map = {
            "return_number": np.uint8,
            "number_of_returns": np.uint8,
            "classification": np.uint8,
            "synthetic": np.uint8,
            "key_point": np.uint8,
            "withheld": np.uint8,
            "scan_direction_flag": np.uint8,
            "edge_of_flight_line": np.uint8,
            "user_data": np.uint8,
            "point_source_id": np.uint32,
            "gps_time": np.float64,  # PF 7 supports GPS time
            "intensity": np.float64,
            "red": np.uint16, "green": np.uint16, "blue": np.uint16,
            # add more if you store them
        }


        if las_scales is None:
            las_scales = [0.00025, 0.00025, 0.00025]
        if las_offset is None:
            las_offset = self.epoch.cloud.min(axis=0).tolist()

        header = laspy.LasHeader(point_format=las_point_format, version=las_version)
        header.scales = las_scales
        header.offsets = las_offset
        
        if self.mapped:
            las = laspy.LasData(header)
            las.xyz = self.epoch.cloud

        else:
            las = laspy.LasData(header)
            las.xyz = self.epoch.cloud
            # Work on voxelisation logic here
            pass 

        # Add all extra dimensions to self.out
        if self.extra_dims is not None:
            for dim_name in self.extra_dims.dtype.names:
                self.out[dim_name] = np.asarray(self.extra_dims[dim_name]).squeeze()  # ensure 1-D


        builtin = set(las.point_format.dimension_names)
        to_write = [c for c in self.out.keys() if c not in skip]

        builtin_cols = [c for c in to_write if c in builtin]
        extra_cols   = [c for c in to_write if c not in builtin]

        for name in builtin_cols:
            arr = self.out[name]
            # cast if we know the on-disk type
            if name in dtype_map:
                arr = arr.astype(dtype_map[name], copy=False)
            las[name] = arr
        
        # Batch add ALL extra dims once, then assign arrays
        if extra_cols:
            extras = []
            for name in extra_cols:
                # store extras by default 
                arr_type = self.out[name].dtype
                if arr_type == bool:
                    arr_type = np.uint8
                print(f"Adding extra dimension '{name}' of type {arr_type}")
                extras.append(laspy.ExtraBytesParams(name=name, type=arr_type, description=name))
            las.add_extra_dims(extras) 
            for name in extra_cols:
                las[name] = self.out[name].reshape(-1)
        las.write(outfile)

    @timeit
    @trace
    def save_as_ply(
        self,
        outfile: str,
        mode:       str | np.ndarray  = "grid",
        features:   list[str]         = None,
        shift_to_center: bool         = False,
    ):
        """
        Export occupied voxels as perfect cubes in a single PLY, with
        optional choice of cube‐centers and extra per‐vertex features.
        This version uses explicit corner‐ordering so faces always connect correctly.
        """
        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            print("plyfile is not installed.")
            print("To enable the save_as_ply functionality in Vapc, use pip to install plyfile.")
        self.mapped = False
        # Ensure voxels are grouped & features are computed
        if self.unique_voxels is None:
            self.group()
        if features:
            if not hasattr(self, "out"):
                self.out = {}
            missing = [f for f in features if f not in self.out]
            if missing:
                self.compute_features(missing)

        # Compute base_positions & corner_offsets
        if isinstance(mode, np.ndarray):
            base_positions = mode.astype(float)
            h = self.voxel_size / 2.0
            # corners around each custom center, in the exact same order as base_faces expects:
            corner_offsets = np.array([
                [-h, -h, -h],
                [ h, -h, -h],
                [ h,  h, -h],
                [-h,  h, -h],
                [-h, -h,  h],
                [ h, -h,  h],
                [ h,  h,  h],
                [-h,  h,  h],
            ], dtype=float)
        else:
            if mode == "grid":
                # if self.use_octree:
                #     keys = self.unique_voxels.astype(np.uint32)
                #     ijk  = decode_morton(keys, self._max_depth)
                #     base_positions = self._octree_origin + ijk * self.voxel_size
                # else:
                base_positions = (
                    self.unique_voxels * self.voxel_size + self.origin
                )
                # cube corners from the min‐corner, ordered to match base_faces:
                corner_offsets = np.array([
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 1],
                    [1, 1, 1],
                    [0, 1, 1],
                ], dtype=np.int32) * self.voxel_size

            elif mode in ("voxel_center", "centroid",
                        "closest_to_centroid", "closest_to_voxel_center"):
                if mode == "voxel_center":
                    centers = (
                        self.compute_voxel_centers()
                        if self.voxel_centers is None else self.voxel_centers
                    )
                elif mode == "centroid":
                    centers = (
                        self.compute_centroids()
                        if self.centroids is None else self.centroids
                    )
                elif mode == "closest_to_centroid":
                    centers = (
                        self.compute_closest_to_centroids()
                        if self.closest_to_centroids is None else
                        self.closest_to_centroids
                    )
                else:  # "closest_to_voxel_center"
                    centers = (
                        self.compute_closest_to_voxel_centers()
                        if self.closest_to_voxel_centers is None else
                        self.closest_to_voxel_centers
                    )
                base_positions = centers.astype(float)
                h = self.voxel_size / 2.0
                # same ordering as above, now centered
                corner_offsets = np.array([
                    [-h, -h, -h],
                    [ h, -h, -h],
                    [ h,  h, -h],
                    [-h,  h, -h],
                    [-h, -h,  h],
                    [ h, -h,  h],
                    [ h,  h,  h],
                    [-h,  h,  h],
                ], dtype=float)

            else:
                raise ValueError(f"Unknown mode: {mode!r}")

        M = base_positions.shape[0]

        # 3) Build the (M*8,3) vertex list
        verts = (base_positions[:, None, :] + corner_offsets[None, :, :]).reshape(-1, 3)

        # 4) Build faces in one broadcasted step
        # Triangles for one cube, indexing corners as above:
        base_faces = np.array([
            [0, 1, 2], [0, 2, 3],   # bottom
            [4, 5, 6], [4, 6, 7],   # top
            [0, 1, 5], [0, 5, 4],   # front
            [2, 3, 7], [2, 7, 6],   # back
            [1, 2, 6], [1, 6, 5],   # right
            [3, 0, 4], [3, 4, 7],   # left
        ], dtype=np.int32)
        offsets = (np.arange(M, dtype=np.int32) * 8)[:, None, None]
        faces   = (base_faces[None, :, :] + offsets).reshape(-1, 3)

        # 5) Optional recenter
        if shift_to_center:
            verts -= verts.mean(axis=0)

        # 6) Assemble PLY vertex dtype
        dtype = [("x","f8"), ("y","f8"), ("z","f8")]
        if features:
            for f in features:
                dtype.append((f, "f8"))

        vertex_array = np.empty(len(verts), dtype=dtype)
        vertex_array["x"], vertex_array["y"], vertex_array["z"] = verts.T

        # 7) Fill feature columns by repeating each voxel's scalar 8×
        if features:
            for f in features:
                arr = self.out[f].astype(float)
                vertex_array[f] = np.repeat(arr, 8)

        # 8) Build face element
        face_array = np.empty(len(faces), dtype=[("vertex_indices","i4",(3,))])
        face_array["vertex_indices"] = faces

        # 9) Write binary PLY
        with open(outfile, "wb") as f:
            PlyData([
                PlyElement.describe(vertex_array, "vertex"),
                PlyElement.describe(face_array,   "face"),
            ], text=False).write(f)

    @timeit
    @trace
    def filter(self, mask, overwrite=False):
        pts = np.asarray(self.epoch.cloud)[mask]
        new = Vapc(Epoch(pts), voxel_size=self.voxel_size, origin=list(self.origin))
        # new.use_octree = self.use_octree
        # Remap per-point outputs if present
        if self.out:
            new.out = {k: v[mask] for k, v in self.out.items() if len(v)==len(mask)}
        # Recompute voxel groups for new cloud
        new.group()
        if overwrite:
            self.__dict__.update(new.__dict__)
            self.mapped = True
            return self
        return new
