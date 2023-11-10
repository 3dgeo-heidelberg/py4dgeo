import os
import json
import sys
import matplotlib.pyplot as plt
from typing import List
import imageio
import numpy as np
from shapely.geometry import Polygon



# Class definition of change event
class ChangeEvent:
    def __init__(
            self,
            object_id: int = None,
            pc_files: List[str] = None,
            timestamps: List[str] = None,
            epoch_count: int = None,
            change_rates: List[float] = None,
            duration: float = None,
            change_magnitudes: List[float] = None,
            spatial_extent: List[float] = None,
            start_end_location: List[List[float]] = None,
            event_type: str = None,
            plots_2d: str = None,
            plots_3d: str = None,
            plots_4d: str = None,

    ):
        self.object_id = object_id
        self.pc_files = pc_files
        self.timestamps = timestamps
        self.epoch_count = epoch_count
        self.change_rates = change_rates
        self.duration = duration
        self.change_magnitudes = change_magnitudes
        self.spatial_extent = spatial_extent
        self.start_end_location = start_end_location
        self.event_type = event_type
        self.plots_2d = plots_2d
        self.plots_3d = plots_3d
        self.plots_4d = plots_4d

        """
        :param object_id:
            The unique id of a change event object as integer.

        :param pc_files:
            The directory where pc files of each epoch of the change event object are stored as string.

        :param timestamps:
            The timestamps of all epochs defining the change event objects as a list of length = epoch count with
            strings of format "YYYY-MM-DD-MM-SS".

        :param epoch_count:
            The number of epochs defining the change event object as integer.

        :param change_rates:
            The change rate for each epoch of the change event object as a list of length = epoch_count.

        :param duration:
            The duration of the change event in seconds as float value.

        :param change_magnitudes:
            The magnitudes of change for each epoch of the change event object as a list of length = epoch_count.

        :param spatial_extent:
            The extent of the change event object given as bounding box as 2d list of shape (4, 2) with the inner 
            dimension with 4 coordinate pairs as float values.
            
        :param start_end_location:
            The start and end location of a change event object as coordinate pairs in a 2d list.

        :param event_type:
            The type of change of a change event objets given as string. A list of available change types (rockfall,
            landslide, etc.) might be defined.

        :param plots_2d:
            A list of file paths to all 2d visualizations generated from the change event object as string.

        :param plots_3d:
            A list of file paths to all 3d visualizations generated from the change event object as string.

        :param plots_4d:
            A list of file paths to all 4d visualizations generated from the change event object as string.



    """

    @classmethod
    def read_change_events_from_geojson(cls, geojson_filename):
        change_events = []

        with open(geojson_filename, 'r') as file:
            json_data = json.load(file)

        # Create ChangeEvent objects from each feature in geojson file and get properties
        for feature in json_data['features']:
            properties = feature['properties']
            geom = feature['geometry']

            if geom['type'] == 'MultiPoint':
                coordinates = geom['coordinates']
                start_end_location = [coordinates[0], coordinates[-1]]  # Erste und letzte Koordinaten

                change_event = cls(
                    object_id=properties.get('object_id'),
                    pc_files=properties.get('pc_files'),
                    timestamps=properties.get('timestamps'),
                    epoch_count=properties.get('epoch_count'),
                    change_rates=properties.get('change_rates'),
                    duration=properties.get('duration'),
                    change_magnitudes=properties.get('change_magnitudes'),
                    spatial_extent=properties.get('spatial_extent'),
                    start_end_location=start_end_location,
                    event_type=properties.get('event_type'),
                    plots_2d=properties.get('plots_2d'),
                    plots_3d=properties.get('plots_3d'),
                    plots_4d=properties.get('plots_4d')
                )

                change_events.append(change_event)

        # Return list of ChangeEvent objects
        return change_events

    def plot_2d(
            self,
            x_attribute="timestamps",
            y_attribute="change_magnitudes",
            plots_2d_dir=None

    ):
        """Plot a 2D line graph of specific attributes over epochs for a ChangeEvent object.

        :param x_attribute (str, optional):
            The attribute to be plotted on the X-axis.

        :param y_attribute (str, optional):
            The attribute to be plotted on the Y-axis.

        :param plots_2d_dir (str, optional):
            The directory where 2D plots will be saved. If None, a default directory structure will be created.

        :return: self.plots_2d
            An extended list of file paths to all 2d visualizations of the change event object.
        """

        if x_attribute not in self.__dict__ or y_attribute not in self.__dict__:
            raise ValueError("Invalid x or y plotting argument. Select from available attributes.")

        # Get values of features to be plotted
        x = getattr(self, x_attribute)
        y = getattr(self, y_attribute)

        # Configure plot layout
        plt.figure(figsize=(15, 15))
        plt.plot(x, y, marker='o', linestyle='-')
        plt.title(f'{y_attribute} of change event with object_id = 00{self.object_id} over all epochs')
        plt.xlabel(x_attribute)
        plt.ylabel(y_attribute)
        plt.grid(True)

        # Set the default plots_2d_dir if not provided
        if plots_2d_dir is None:
            default_dir = os.path.join("output", "2d_plots", f"00{self.object_id}")
            os.makedirs(default_dir, exist_ok=True)
            plots_2d_dir = default_dir

        # Make sure the directory exists
        os.makedirs(plots_2d_dir, exist_ok=True)
        print(plots_2d_dir)

        # Save plot to the specified directory
        plot_filename = f"plot2d_00{self.object_id}_{x_attribute}_{y_attribute}.png"
        plot_filepath = os.path.join(plots_2d_dir, plot_filename)
        plt.savefig(plot_filepath)
        print(plot_filepath)

        # Add filename to the list of plots of the change object
        if self.plots_2d is None:
            self.plots_2d = [plot_filepath]
        else:
            self.plots_2d.append(plot_filepath)

        plt.show()

    def plot_3d(
            self,
            plotting_attribute_index,
            attribute_name,
            plots_3d_dir=None
    ):
        """Plot multiple point clouds of a ChangeEvent object in the same 3D plot with colored points based on a
        specified attribute.

        :param plotting_attribute_index:
            The attribute index to be used for coloring the points as integer.

        :param attribute_name:
            The name of the attribute to be used for coloring the points as string.

        :return None
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xyz = []
        for pc_file in self.pc_files:
            point_cloud = self.read_point_cloud_from_xyz(pc_file)
            if point_cloud is not None:
                x = point_cloud[:, 0]
                y = point_cloud[:, 1]
                z = point_cloud[:, 2]
                attribute_column = point_cloud[:, plotting_attribute_index]

                min_length = min(len(x), len(y), len(z), len(attribute_column))
                points = list(zip(x[:min_length], y[:min_length], z[:min_length], attribute_column[:min_length]))
                xyz.extend(points)

        xyz = np.array(xyz)
        valid_points = xyz[~np.isnan(xyz[:, -1])]

        x = valid_points[:, 0]
        y = valid_points[:, 1]
        z = valid_points[:, 2]
        attribute_column = valid_points[:, 3]

        norm = plt.Normalize(attribute_column.min(), attribute_column.max())
        colors = plt.cm.viridis(norm(attribute_column))

        ax.scatter(x, y, z, c=colors, label=f'Attribute: {attribute_name}')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Scaling
        ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])

        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=norm))
        cbar.set_label(attribute_name)
        plt.title(f'{attribute_name} of change event with object_id = 00{self.object_id} over all epochs')
        plt.grid(True)

        if plots_3d_dir is None:
            default_dir = os.path.join("output", "3d_plots", f"00{self.object_id}")
            os.makedirs(default_dir, exist_ok=True)
            plots_3d_dir = default_dir

        os.makedirs(plots_3d_dir, exist_ok=True)

        plot_filename = f"plot3d_{self.object_id}_{attribute_name}.png"
        plot_filepath = os.path.join(plots_3d_dir, plot_filename)
        plt.savefig(plot_filepath)

        if self.plots_3d is None:
            self.plots_3d = [plot_filepath]
        else:
            self.plots_3d.append(plot_filepath)

        plt.show()

    def plot_4d(
            self,
            plotting_attribute_index,
            attribute_name,
            plots_4d_dir=None
    ):
        """Create a 3D animation of point clouds for different epochs.

        :param plotting_attribute:
            The attribute to be used for coloring the points as string.

        :param plots_4d_dir (str, optional):
            The directory where 4D plots will be saved. If None, a default directory structure will be created.

        :return None
        """
        plot_filename = f"plot4d_{self.object_id}_{attribute_name}"
        file_paths = []  # To store file paths

        if plots_4d_dir is None:
            default_dir = os.path.join("output", "4d_plots", f"00{self.object_id}")
            os.makedirs(default_dir, exist_ok=True)
            plots_4d_dir = default_dir

        images_dir = os.path.join(plots_4d_dir, plot_filename)
        os.makedirs(images_dir, exist_ok=True)

        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')

        for i, pc_file in enumerate(self.pc_files):
            point_cloud = self.read_point_cloud_from_xyz(pc_file)
            if point_cloud is not None:
                x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
                min_x = min(min_x, np.min(x))
                max_x = max(max_x, np.max(x))
                min_y = min(min_y, np.min(y))
                max_y = max(max_y, np.max(y))
                min_z = min(min_z, np.min(z))
                max_z = max(max_z, np.max(z))

        max_range = max(max_x - min_x, max_y - min_y, max_z - min_z)

        global_min_value = float('inf')
        global_max_value = float('-inf')

        for i, pc_file in enumerate(self.pc_files):
            point_cloud = self.read_point_cloud_from_xyz(pc_file)
            if point_cloud is not None:
                attribute_values = point_cloud[:, plotting_attribute_index]
                min_value, max_value = np.min(attribute_values), np.max(attribute_values)
                global_min_value = min(global_min_value, min_value)
                global_max_value = max(global_max_value, max_value)

        for i, pc_file in enumerate(self.pc_files):
            point_cloud = self.read_point_cloud_from_xyz(pc_file)
            if point_cloud is not None:
                x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
                attribute_values = point_cloud[:, plotting_attribute_index]

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                scatter = ax.scatter(x, y, z, c=attribute_values, cmap='viridis', vmin=global_min_value,
                                     vmax=global_max_value)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(f"{attribute_name} of change event with object_id = {self.object_id} in epoch {i}")

                ax.set_xlim(min_x, min_x + max_range)
                ax.set_ylim(min_y, min_y + max_range)
                ax.set_zlim(min_z, min_z + max_range)

                cbar = fig.colorbar(scatter)
                cbar.set_label(attribute_name)

                plot_filepath = os.path.join(images_dir, f"frame_{i:04d}.png")
                plt.savefig(plot_filepath)
                file_paths.append(plot_filepath)
                plt.close(fig)

        anim_file_path = create_change_animation(file_paths, attribute_name, self.object_id)

        if self.plots_4d is None:
            self.plots_4d = [anim_file_path]
        else:
            self.plots_4d.append(anim_file_path)
        return file_paths


    def read_point_cloud_from_xyz(self, pc_file):
        """Load a point cloud from a ascii file.

        :param pc_file:
            Path to the ascii file.

        :return point_cloud:
            A NumPy array representing the point cloud.
        """
        with open(pc_file, "r") as file:
            lines = file.readlines()

        point_cloud = []
        for line in lines:
            values = line.split()  # Annahme: Werte sind durch Leerzeichen getrennt
            x, y, z, distance = map(float, values)
            point_cloud.append([x, y, z, distance])

        return np.array(point_cloud)

    def visualize_change_event(self, visualization_parameters, plotting_attribute="c2c_dist"):
        """Generate plots for each ChangeEvent based on visualization parameters.

        :param event_list:
            A list of ChangeEvent objects to generate plots for.

        :param visualization_parameters:
            A dictionary with visualization parameters.

        :param plotting_attribute (str, optional):
            The attribute to be used to color the point clouds as string.
        """
        if "visualization_modes" in visualization_parameters:
            modes = visualization_parameters["visualization_modes"]
            if "2d" in modes:
                self.plot_2d()
            if "3d" in modes:
                self.plot_3d(-1, plotting_attribute)
            if "4d" in modes:
                self.plot_4d(-1, plotting_attribute)
            else:
                print("No visualization modes specified in visualization_parameters.")

    def write_change_event_in_geojson(self,
                                      change_event,
                                      output_file=None):
        """Writes a ChangeEvent object to a GeoJSON output file.

        :param change_event:
            A ChangeEvent object to be included in the GeoJSON file.

        :param output_file:
            The filename of the GeoJSON file to be created.

        :return None
        """

        geometry = {
            "type": "MultiPoint",
            "coordinates": change_event.start_end_location
        }

        feature = {
            "type": "Feature",
            "properties": {
                "duration": change_event.duration,
                "change_magnitudes": change_event.magnitude,
                "extent": change_event.extent,
                "change_type": change_event.change_type,
                "plots_2d": change_event.plots_2d,
                "plots_3d": change_event.plots_3d,
                "plots_4d": change_event.plots_4d
            },
            "geometry": geometry
        }

        geojson_data = {
            "type": "FeatureCollection",
            "features": [feature]
        }

        # Set the default output_file if not provided
        if output_file is None:
            default_dir = os.path.join("output", "updated_change_events", str(self.object_id))
            os.makedirs(default_dir, exist_ok=True)
            output_file = os.path.join(default_dir, "00", f"{self.object_id}_change_event.json")

        with open(output_file, 'w') as file:
            json.dump(geojson_data, file, indent=4)

def create_change_animation(images_paths, attribute_name, object_id, plots_4d_dir=None):
    """Generate animated plots for each ChangeEvent based on a list of input frames.

    :param images_paths:
        A list of file paths of plots to be animated.

    :param attribute_name:
        The attribute used to color the ChangeEvent objects in the plots.

    :param object_id:
        The id of the ChangeEvent object to be plotted.

    :param plots_4d_dir (str, optional):
        The directory where 4D plots will be saved. If None, a default directory structure will be created.

    :return anim file_path:
        The directory of the generated animation file.

    """
    images = []
    if plots_4d_dir is None:
        default_dir = os.path.join("output", "4d_plots", f"00{object_id}")
        os.makedirs(default_dir, exist_ok=True)
        plots_4d_dir = default_dir

    os.makedirs(plots_4d_dir, exist_ok=True)

    plot_filename = f"plot4d_00{object_id}_{attribute_name}.gif"
    anim_file_path = os.path.join(plots_4d_dir, plot_filename)
    for image in images_paths:
        images.append(imageio.imread(image))

    imageio.mimsave(anim_file_path, images, duration=1)

    return anim_file_path

def read_visualization_parameters(json_path):
    """Extract visualization parameters from a GeoJSON file.

    This function reads a GeoJSON file and extracts specific visualization parameters like observation
    period, spatial extent, event types, and visualization modes.

    :param: geojson_filename
        The filename of the input GeoJSON file.

    :return parameters:
        A dictionary containing the extracted visualization parameters.
    """

    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)

    parameters = {
        "observation_period": None,
        "duration_min_max": None,
        "spatial_extent": None,
        "eventTypes": None,
        "visualization_modes": None
    }

    if 'features' in json_data and json_data['features']:
        feature = json_data['features'][0]
        if 'properties' in feature:
            properties = feature['properties']
            parameters["observation_period"] = properties.get('observation_period')
            parameters["duration_min_max"] = properties.get('duration')
            parameters["spatial_extent"] = properties.get('spatial_extent')
            parameters["eventTypes"] = properties.get('eventTypes')
            parameters["visualization_modes"] = properties.get('visualization_modes')

    return parameters

def filter_relevant_events(event_list, visualization_parameters):
    """Retrieve a list of relevant ChangeEvent objects based on criteria defined in visualization parameters.

    :param: event_list:
        A list of ChangeEvent objects.

    :param visualization_parameters
        A dictionary with all parameters for filtering relevant ChangeEvent objects.

    :return relevant_events:
        A list of relevant ChangeEvent objects.
   """

    relevant_events = event_list

    # Check all change events for matching the provided criteria
    for key, value in visualization_parameters.items():
        if key == "eventTypes":
            relevant_events = [event for event in relevant_events if
                               any(event_type in event.event_type for event_type in value)]
        elif key == "duration":
            relevant_events = [event for event in relevant_events if value[0] <= event.duration <= value[1]]
        #elif key == "spatial_extent":
            #relevant_events = [event for event in relevant_events if is_inside_extent(event.spatial_extent, value)]
        #elif key == "observation_period":
            #relevant_events = [event for event in relevant_events if
                               #event.timestamps[0] >= value[0] and event.timestamps[-1] <= value[1]]


    return relevant_events

# helper functions
def is_inside_extent(event_extent, target_extent):
    """Check if an event extent is inside the target extent polygon.

    :param event_extent:
        The extent of the ChangeEvent as a list of 4 coordinate pairs.

    :param target_extent:
        The target extent polygon as a list of coordinate pairs.

    :return is_inside:
        True if the event extent is inside the target extent polygon, False otherwise.
    """
    # Convert the event extent and target extent to Shapely Polygon objects
    event_polygon = Polygon(event_extent)
    target_polygon = Polygon(target_extent)

    # Check if the event extent is completely within the target extent polygon
    is_inside = event_polygon.within(target_polygon)

    return is_inside

def main():

    # Read data
    visualization_parameters_json = sys.argv[1]
    change_events_geojson = sys.argv[2]

    # Get visualization parameters
    visualization_parameters = read_visualization_parameters(visualization_parameters_json)

    # Create ChangeEvent objects from geojson
    change_events = ChangeEvent.read_change_events_from_geojson(change_events_geojson)

    # Filter relevant ChangeEvent objects for visualization
    relevant_change_events = filter_relevant_events(change_events, visualization_parameters)

    # Plot ChangeEvent object as defined in visualization parameters
    for change_event in relevant_change_events:
        #change_event.plot_2d()
        change_event.visualize_change_event(visualization_parameters)
        # change_event.write_change_event_in_geojson(change_events)

if __name__ == "__main__":
    main()
