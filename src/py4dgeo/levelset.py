import py4dgeo
from py4dgeo.levelset_algorithm_functions import _process

import numpy as np
import os
from multiprocessing import Pool
from pathlib import Path
import alphashape
import re
import tempfile
import tqdm
import geopandas as gpd
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree
from scipy.spatial.qhull import QhullError
from shapely.geometry import Point, MultiPolygon, Polygon
import plotly.graph_objects as go
from plotly.subplots import make_subplots


verbose = False


class LevelSetAlgorithm:
    def __init__(self, working_dir=None, **kwargs):
        """
        Initialize the LevelSetAlgorithm


        Options for the data setup:
        - `first_timestep` (int): The first timestep to process. Default is 0.
        - `last_timestep` (int): The last timestep to process. If set to -1, all timesteps until the end are processed. Default is -1.
        - `timestep_interval` (int): The interval between timesteps to process. Default is 1.

        Example for the interval pairings:
            interval: 24 (every timestep will be compared to the timestep + 24)
            first_timestep: 0 (start with the 0 index)
            last_timestep: 75 (last index to be used will be 75, so the highest index accessed will be 99)
            step0: f1:0 f2:24
            step1: f1:1 f2:25
            step2: f1:2 f2:26
            step3: f3:3 f2:27
            ...
            stepN f1:75 f2:99

        Options for level set algorithm:
        - `reuse_intermediate` (bool): Re-use intermediate calculations
        (neighbors, normals, tangents). Default is True.
        - `active_contour_model` (str): Active contours model, either
        'chan_vese' or 'lmv' (Local Mean and Variance). Default is 'chan_vese'.
        - `num_cycles` (int): Number of cycles, each runs a number of steps
        then stores a result. Default is 12.
        - `num_steps` (int): Number of steps per cycle. Default is 50.
        - `num_smooth` (int): Number of smoothing passes for zeta.
        Default is 1.
        - `stepsize` (int): Explicit Euler step size. Default is 1000.
        - `nu` (float): Controls regularization. Default is 0.0001.
        - `mu` (float): Controls curvature. Default is 0.0025.
        - `lambda1` (float): Controls zeta-in term. Default is 1.0.
        - `lambda2` (float): Controls zeta-out term. Default is 1.0.
        - `epsilon` (float): Heaviside/delta approximation "width",
        is scaled with `h`. Default is 1.0.
        - `h` (float): Approximate neighborhood radius
        (all k neighbors should be within). Default is 2.5.
        - `k` (int): Number of kNN neighbors. Default is 7.
        - `tolerance` (float): Termination tolerance. Default is 5e-5.
        - `cue_clip_pc` (float): Robust cues, clip at X%. Default is 99.9.
        - `vox_size` (int): Initialization voxel size. Default is 10.
        - `init_pc` (int): Initialization cue percentage. Default is 50.
        - `init_method` (str): Initialization method, either 'voxel' or 'cue'.
        Default is 'voxel'.
        - `extraction_threshold` (int): Neighbor threshold for points
        to be extracted
            (must have >= salient neighbors to be extracted).
            Calculated as `k // 2`.
        - `center_data` (bool): Recenter cues by subtracting cue median.
        Default is False.


        Options for the shape analysis:
        - `_filter` (str): Choose between the positive and negative data files.
        Default is 'positive'.
        - `distance_threshold` (float): How far points can be to still be
        considered of the same object. Default is 1.
        - `change_threshold` (float): How high the change value needs to be to
        be considered a valid entry. Default is 0.5.
        - `alpha` (float): Alpha parameter for the alpha shape identification,
        the lower the smoother the shape, but less exact. Default is 1.
        - `area_threshhold` (int): Area threshold for filtering small polygons.
        Default is 100.
        - `iou_threshold` (float): Intersection over Union (IoU) threshold for
        assigning objects IDs in different time steps. Default is 0.5.

        :param working_dir: The directory where the output files will be saved
        if None given this will use a temporary directory, defaults to None
        :type working_dir: str, optional
        """

        self.options = kwargs

        self.working_dir = working_dir
        if self.working_dir is None:
            self.working_dir = tempfile.mkdtemp()

    def run(self, analysis):
        self._analysis = analysis

        self.data = self._get_analysis_data()

        self.run_analysis()

        self.analyse_shapes()

        gdf = self.get_shape_connectivity()
        objects = self.group_objects(gdf)

        analysis.objects = objects

        return objects

    def run_analysis(self):
        """Run the levelset function with the given data."""

        pairs = self.data["pairs"]

        progress_bar = tqdm.tqdm(total=len(pairs) * 2)  # Each pair has two directions

        for field_pair in pairs:
            for changedir in ["positive", "negative"]:
                _process(self.data, field_pair, self.options, changedir)
                progress_bar.update(1)

    def analyse_shapes(self):
        """Read the created data files and extract the object shapes from the labeled arrays

        :param data: the data dict
        :type data: dict
        :param distance_threshold: How far apart points can be to be labeled as the same object, defaults to 1
        :type distance_threshold: int, optional
        :param change_threshold: The minimum value to be considered significant, anything lower will be set to 0, defaults to 0.5
        :type change_threshold: float, optional
        :param alpha: How closly the alpha shape should fit, defaults to 1
        :type alpha: int, optional
        :param area_threshhold: The minimum area for an alpha shape to be considered valid, defaults to 100
        :type area_threshhold: int, optional
        """

        self._collect_result_files()

        shape_dict = {}
        distance_dict = {}

        for file in tqdm.tqdm(self.data_files):
            file = Path(file)
            distance_df = self._extract_change_data_from_file(file)
            distance_df["polygon_id"] = 0

            distance_df = self._label_array(
                distance_df,
            )

            shapes = self._get_shapes_from_lables(distance_df)

            self._find_ids_in_polygon(
                distance_df, shapes, shape_dict, distance_dict, file
            )

        self.shape_dict = shape_dict
        self.distance_dict = distance_dict

    def get_shape_connectivity(self):

        iou_thr = self.options.get("iou_threshold", 0.5)
        # setup the geodataframe
        gdf = gpd.GeoDataFrame(self.shape_dict).T
        gdf[["first_epoch", "second_epoch", "index_in_epoch"]] = gdf.index.str.extract(
            r"(\d+).*?(\d+).*?(\d+)"
        ).values

        gdf["first_epoch"] = pd.to_numeric(gdf["first_epoch"])
        gdf["second_epoch"] = pd.to_numeric(gdf["second_epoch"])
        gdf.sort_values("first_epoch", inplace=True)
        gdf["status"] = "candidate"
        gdf["iou_thr"] = iou_thr

        iou_matrix = self._calc_iou_matrix(gdf, iou_thr)

        self._find_connected_objects(gdf, iou_matrix, iou_thr)

        return gdf

    def group_objects(self, gdf):
        groups_dict = {
            object_id: group for object_id, group in gdf.groupby("object_id")
        }

        objects = []
        _filter = self.options.get("filter", "positive")

        for object_key, obj_df in groups_dict.items():
            new_object = ObjectByLevelset(
                obj_df, self.data, self.distance_dict, _filter, oid=object_key
            )
            objects.append(new_object)
        return objects

    def _get_analysis_data(self):
        """Transform the py4dgeo analysis object into a dictionary that can be used by the levelset function.
        The dictionary contains the following
        - setup: a dictionary all information for the setup process as well as all algorithm parameters
        - xyz: the corepoints of the analysis object
        - origin: the median of the corepoints
        - timedeltas: the timedeltas of the analysis object
        - fields: the fields that will be used for the levelset function
        - change_{t}: the change in the field at time step t
        - pairs: the pairs of fields that will be used for the levelset function




        :return: dictionary containing all necessary data for the levelset function
        :rtype: dict
        """

        first_timestep = self.options.get("first_timestep", 0)
        last_timestep = self.options.get("last_timestep", -1)
        timestep_interval = self.options.get("timestep_interval", 1)

        data = {}
        data_obj = self._analysis

        data["xyz"] = data_obj.corepoints.cloud
        data["origin"] = np.round(np.median(data["xyz"], axis=0), 0)
        data["xyz"] = (
            data["xyz"] - data["origin"]
        )  # similar to read_las func, we also apply this offset 'globally' for the script
        data["timedeltas"] = np.array(
            [int(dt.total_seconds()) for dt in data_obj.timedeltas]
        )  # in seconds
        distances = data_obj.distances_for_compute

        if last_timestep == -1:
            last_timestep = len(distances[0]) - timestep_interval

        if last_timestep + timestep_interval > len(distances[0]):
            raise ValueError(
                "The last timestep plus the interval is larger than the available data"
            )

        # slice the available field data from the first timestep to the last timestep plus the interval
        fields = [
            f"change_{i}"
            for i in range(
                first_timestep, len(distances[0][: last_timestep + timestep_interval])
            )
        ]
        data["fields"] = fields

        for t in range(len(data["timedeltas"])):
            fields_name = f"change_{t}"
            if fields_name in data["fields"]:
                data[f"change_{t}"] = np.array([cp[t] for cp in distances])

        # form the pairs

        data["pairs"] = [
            (data["fields"][i], data["fields"][i + timestep_interval])
            for i in range(len(data["fields"]) - timestep_interval)
        ]

        in_file = self._analysis.filename

        # field to use as cue
        base_dir = os.path.join(
            os.path.dirname(self.working_dir),
            os.path.splitext(os.path.basename(in_file))[0]
            + f"_k{first_timestep}_{last_timestep}_{timestep_interval}",
        )
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        self.options["base_dir"] = base_dir

        return data

    def _calc_iou_matrix(self, gdf, iou_thr):

        # get the intersection over union as a matrix
        iou_matrix = pd.DataFrame(index=gdf.index, columns=gdf.index, dtype=float)

        # Calculate the IoU for each pair of polygons, leveraging symmetry to reduce computations
        for i, poly_i in gdf.iterrows():
            for j, poly_j in gdf.loc[
                i:
            ].iterrows():  # Start from i to avoid recalculating for pairs already computed
                # Calculate the intersection area
                intersection_area = (
                    poly_i["polygon"].intersection(poly_j["polygon"]).area
                )
                # Calculate the union area
                union_area = poly_i["polygon"].union(poly_j["polygon"]).area
                # Calculate IoU and store it in the matrix
                # Avoid division by zero by checking if union_area is not zero
                iou_value = intersection_area / union_area if union_area != 0 else 0
                if iou_value > iou_thr:
                    iou_matrix.at[i, j] = iou_value
                    if i != j:  # Ensure we're not on the diagonal
                        iou_matrix.at[j, i] = (
                            iou_value  # Mirror the value across the diagonal
                        )
                else:
                    iou_matrix.at[i, j] = 0
                    if i != j:  # Ensure we're not on the diagonal
                        iou_matrix.at[j, i] = 0  # Mirror the value across the diagonal
        return iou_matrix

    def _find_connected_objects(self, gdf, iou_matrix, iou_thr):

        # Get the indices of the polygons that are connected
        G = nx.Graph()

        # Add nodes
        for i in gdf.index:
            G.add_node(i)

        # Add edges for IoU values above the threshold
        for i, row in iou_matrix.iterrows():
            for j, iou_value in row.items():  # Use items() instead of iteritems()
                if (
                    i != j and iou_value > iou_thr
                ):  # Avoid self-loops and check threshold
                    G.add_edge(i, j)

        # Find connected components
        connected_components = list(nx.connected_components(G))

        # Create a dictionary to map each polygon to its group
        group_mapping = {}
        for group_id, component in enumerate(connected_components):
            for node in component:
                group_mapping[node] = group_id

        gdf["object_id"] = gdf.index.map(group_mapping)

        # set status to "connected" if the group index occurs more than once
        for i in gdf["object_id"].unique():
            if gdf["object_id"].value_counts()[i] > 1:
                gdf.loc[gdf["object_id"] == i, "status"] = "connected"
            else:
                gdf.loc[gdf["object_id"] == i, "status"] = "isolated"

    def _collect_result_files(self):

        dirs_list = [Path(d) for d in Path(self.options["base_dir"]).glob("change_*")]

        _filter = self.options.get("filter", "positive")

        files_list = []
        for d in dirs_list:
            if not re.search(_filter, d.name):
                continue
            files_dict = {}

            for file in d.glob("*_extract.txt"):
                # find file with highest iteration number
                iter_number = int(re.search(r"(\d+)_extract.txt", file.name).group(1))
                files_dict[iter_number] = file

            files_list.append(files_dict.pop(max(files_dict.keys())))

        self.data_files = files_list

    def _extract_change_data_from_file(self, data_file):
        extracted_data = np.loadtxt(data_file)

        df = pd.DataFrame(
            data=extracted_data,
            columns=["x", "y", "z", "change_x", "change_y", "change_z"],
        )

        total_change = np.sqrt(
            df["change_x"].values ** 2
            + df["change_y"].values ** 2
            + df["change_z"].values ** 2
        )
        df["total_change"] = total_change
        return df

    def _label_array(self, distance_df):

        distance_threshold = self.options.get("distance_threshold", 1)
        change_threshold = self.options.get("change_threshold", 0.5)

        # drop all rows with total_change < change_threshold
        distance_df = distance_df[distance_df["total_change"] >= change_threshold]
        coordinates = distance_df[["x", "y", "z"]].values

        tree = cKDTree(coordinates)
        G = nx.Graph()

        # Query the KD-tree for each point to find neighbors within the distance_threshold
        for i, point in enumerate(coordinates):
            for j in tree.query_ball_point(point, distance_threshold):
                if i != j:  # Avoid self-loops
                    G.add_edge(i, j)

        # Find connected components
        connected_components = list(nx.connected_components(G))

        # Map each component back to the original DataFrame
        component_labels = np.zeros(len(distance_df), dtype=int)

        for component_id, component in enumerate(connected_components):
            for idx in component:
                component_labels[idx] = component_id + 1

        distance_df["component"] = component_labels

        return distance_df

    def _get_shapes_from_lables(self, distance_df):
        """_summary_

        :param labeled_array: _description_
        :type labeled_array: _type_
        :param alpha: Should be between 0 and 1, defaults to 1
        :type alpha: int, optional
        """
        alpha = self.options.get("alpha", 1)
        area_threshhold = self.options.get("area_threshhold", 100)

        labels = np.unique(
            distance_df["component"].values,
        )

        alpha_list = []

        for current_label in labels:
            if current_label == 0:
                continue

            # get the alpha shape
            # Get the indices of non-zero elements
            points = distance_df[["x", "y"]][
                distance_df["component"] == current_label
            ].values

            try:
                alpha_shape = alphashape.alphashape(points, alpha=alpha)
                if alpha_shape.area < area_threshhold:
                    continue
                alpha_list.append(alpha_shape)

            except QhullError:
                continue

        return alpha_list

    def _find_ids_in_polygon(
        self, distance_df, shapes, shape_dict, distance_dict, file
    ):
        # Convert points to GeoDataFrame
        points_gdf = gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in distance_df[["x", "y"]].values]
        )
        polygon_ids = pd.Series(
            index=distance_df.index, dtype="float64"
        )  # Temporary Series to hold polygon IDs

        # Create spatial index for points
        points_sindex = points_gdf.sindex

        for i, shape in enumerate(shapes):
            key = f"{file.parent.name}_ind_{i+1}"

            # Find points that could potentially interact with the shape using the spatial index
            possible_matches_index = list(points_sindex.intersection(shape.bounds))
            possible_matches = points_gdf.iloc[possible_matches_index]

            # Check if each possible point is contained by or touches the shape
            contained_or_touched = possible_matches.geometry.apply(
                lambda point: shape.contains(point) or shape.touches(point)
            )

            # Update polygon_ids for points contained or touched by the shape
            polygon_ids.iloc[possible_matches_index] = contained_or_touched.apply(
                lambda x: i + 1 if x else 0
            )

            # Update shape_dict after processing all points to minimize DataFrame operations
            shape_dict[key] = {
                "polygon": shape,
                "point_indices": list(polygon_ids[polygon_ids == i + 1].index),
            }

        # Update distance_df in one operation
        distance_df["polygon_id"] = polygon_ids
        distance_dict[file.parent.name] = distance_df


class ObjectByLevelset:
    """Representation a change object in the spatiotemporal domain"""

    def __init__(self, gdf, data, distance_dict, filter, oid, analysis=None):
        self._gdf = gdf
        self._analysis = analysis
        self._data = data
        self._oid = oid
        self._distance_dict = distance_dict
        self._filter = filter

        self._indices = None
        self._distances = None
        self._coordinates = None
        self._polygons = None

    @property
    def filter(self):
        return self._filter

    @property
    def oid(self):
        return self._oid

    @property
    def timesteps(self):
        """The timesteps that compose the object by change"""
        return [int(i) for i in self._gdf["first_epoch"].unique()]

    @property
    def interval(self):
        """The intervals that compose the object by change"""
        return list((self._gdf["second_epoch"] - self._gdf["first_epoch"]).unique())[0]

    @property
    def iou_thresholds(self):
        """The iou_thresholds that compose the object by change"""
        return list(self._gdf["iou_threshold"].unique())

    @property
    def indices(self):
        """Returns a dict with first_epoch as key and all indices inside the corresponding polygon"""

        if self._indices is None:
            index_dict = {}

            for i in self.timesteps:
                # get the indices of the polygon
                filter_str = f"change_{i}_change_{i+self.interval}_{self.filter}"

                mask = self._gdf.index.astype(str).str.contains(filter_str)

                point_indices = self._gdf[mask].point_indices.values[0]
                index_dict[int(i)] = point_indices

            self._indices = index_dict
        return self._indices

    @property
    def distances(self):

        if self._distances is None:
            distances = {}

            # this should iterate over all timesteps
            for i in self.timesteps:
                # get the indices of the polygon
                point_indices = self.indices[i]
                # get the distance data
                distance_df = self._distance_dict[
                    f"change_{i}_change_{i+self.interval}_{self.filter}"
                ]
                # get the distance data for the object
                distance_df = distance_df.loc[point_indices]
                distances[int(i)] = distance_df["total_change"].values

            self._distances = distances
        return self._distances

    @property
    def coordinates(self):

        if self._coordinates is None:
            coordinates_dict = {}

            # this should iterate over all timesteps
            for i in self.timesteps:
                # get the indices of the polygon
                point_indices = self.indices[i]
                # get the distance data
                distance_df = self._distance_dict[
                    f"change_{i}_change_{i+self.interval}_{self.filter}"
                ]
                # get the distance data for the object

                distance_df = distance_df.loc[point_indices]
                coordinates_dict[int(i)] = np.asarray(
                    distance_df[["x", "y", "z"]].values
                )

            self._coordinates = coordinates_dict
        return self._coordinates

    @property
    def polygons(self):

        if self._polygons is None:
            polygons_dict = {}

            for i in self.timesteps:
                polygons_dict[int(i)] = self._gdf[self._gdf["first_epoch"] == i][
                    "polygon"
                ].values[0]

            self._polygons = polygons_dict
        return self._polygons

    def change_histogram(self, nbins_x=10):
        # Create a histogram
        fig = make_subplots(
            rows=1,
            cols=1,
            subplot_titles=("Histogram of Values",),
        )

        for i in self.timesteps:
            fig.add_trace(
                go.Histogram(
                    x=self.distances[i],
                    name=f"Epoch {i}",
                    nbinsx=nbins_x,
                    legendgroup=f"epoch_{i}",
                ),
                row=1,
                col=1,
            )
        # Customize the histogram layout
        fig.update_xaxes(title_text="Distance", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)

        # Customize layout
        fig.update_layout(
            title_text="Object Plot",  # Title of the plot
            height=1000,
            bargap=0.2,  # Gap between bars of adjacent location coordinates
        )
        return fig

    def plot(self):
        fig = go.Figure()

        for key, geom in self.polygons.items():
            # Check if the geometry is a Polygon
            if isinstance(geom, Polygon):
                polygons = [geom]
            # Check if the geometry is a MultiPolygon
            elif isinstance(geom, MultiPolygon):
                polygons = geom.geoms
            else:
                continue  # Skip non-Polygon/MultiPolygon geometries

            for polygon in polygons:
                x, y = np.asarray(polygon.exterior.coords).T
                fig.add_trace(
                    go.Scatter(
                        x=x, y=y, name=f"Polygon_{self.oid}_epoch_{key}", mode="lines"
                    )
                )

        fig.update_layout(
            title_text="Plot of Polygons",
            scene=dict(aspectmode="cube"),
            height=500,
            width=500,
        )
        return fig
