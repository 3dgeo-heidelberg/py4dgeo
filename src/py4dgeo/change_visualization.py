from __future__ import annotations
import sys
import imageio
import numpy as np
from shapely.geometry import Polygon
from pydantic import ValidationError
import json
import os
import matplotlib.pyplot as plt
from typing import List, Union
from typing import List, Tuple
from pydantic import BaseModel, PositiveInt, validate_model, ValidationError
from typing import List


###########################################

#               PlotClass                 #

###########################################
class Plot:
    # Get plot Object of the respective pydantic class and a ChangeEvent object
    plot: Union[LinePlot2d, Plot3d, Animation] = None
    change_event: ChangeEvent = None

    @classmethod
    def generate_plot(cls, plot_type, json_data, change_event):
        """Validate JSON file content and generate a Plot object for a ChangeEvent object based on the plot type defined
        in the input JSON.

        :param plot_type:
            The plot type to be generated.

        :param json_data:
            The content of the input json file as dictionary.

        :param change_event:
            The ChangeEvent object to be plotted.
        """

        # Initialize plot bject of respective pydantic plot class and validate json
        if plot_type == 'line_plot_2d':
            cls.plot = LinePlot2d(**json_data)
            cls.change_event = change_event  # Set the change_event for the class
            cls.generate_line_plot_2d()
        if plot_type == 'plot_3d':
            cls.plot = Plot3d(**json_data)
            cls.change_event = change_event  # Set the change_event for the class
            cls.generate_plot_3d()
        if plot_type == 'animation':
            cls.plot = Animation(**json_data)
            cls.change_event = change_event  # Set the change_event for the class
            cls.generate_animation()

    @classmethod
    def generate_line_plot_2d(cls):
        """Plot a 2D line graph of two variables for all epochs for a ChangeEvent object."""

        try:
            if cls.change_event:
                # Get values of features to be plotted
                x = getattr(cls.change_event, cls.plot.x_label)  # access attribute of ChangeEvent object
                y = getattr(cls.change_event, cls.plot.y_label)

                # Configure plot layout
                plt.figure(figsize=cls.plot.fig_size)
                plt.plot(x, y, marker='o', linestyle='-', linewidth=cls.plot.line_width)

                # Generate default plot title if not provided
                if cls.plot.plot_title is None:
                    cls.plot.plot_title = f'{cls.plot.y_label} of change event with object_id = 00{cls.change_event.object_id} over all epochs'

                plt.title(cls.plot.plot_title)
                plt.xlabel(cls.plot.x_label)
                plt.ylabel(cls.plot.y_label)
                plt.grid(True)

                # Set output dir and plot file name and save plot
                output_dir = os.path.join(cls.plot.plot_dir, "2d_plots")
                os.makedirs(output_dir, exist_ok=True)
                plot_file_name = f"ChangeEvent_00{cls.change_event.object_id}_2dLinePlot{cls.plot.x_label}_{cls.plot.y_label}.png"
                plot_file_path = os.path.join(output_dir, plot_file_name)
                plt.savefig(plot_file_path)

                # Show plot
                plt.show()

        except Exception as e:
            print(f"Could not create 2d line plot object: {e}")

    @classmethod
    def generate_plot_3d(cls):
        """Plot multiple point clouds of a ChangeEvent object in the same 3D plot with colored points based on a
        specified attribute."""

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xyz = []
        for pc_file in cls.change_event.pc_files:
            point_cloud = read_point_cloud_from_xyz(pc_file)
            if point_cloud is not None:
                x = point_cloud[:, 0]
                y = point_cloud[:, 1]
                z = point_cloud[:, 2]
                attribute_column = point_cloud[:, cls.plot.plotting_attribute_index]

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

        ax.scatter(x, y, z, c=colors, label=f'Attribute: {cls.plot.plotting_attribute}')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Scaling
        ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)])

        cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=norm))
        cbar.set_label(cls.plot.plotting_attribute)
        plt.title(f'{cls.plot.plotting_attribute} of change event with object_id = 00{cls.change_event.object_id} '
                  f'over all epochs')
        plt.grid(True)

        # make sure plot directories for Plot and ChangeEvent objects are set
        output_dir = os.path.join(cls.plot.plot_dir, "3d_plots")
        os.makedirs(output_dir, exist_ok=True)

        # set plot_file_name
        plot_file_name = f"ChangeEvent_00{cls.change_event.object_id}_Plot3d_{cls.plot.plotting_attribute}.png"
        plot_filepath = os.path.join(output_dir, plot_file_name)
        plt.savefig(plot_filepath)

        plt.savefig(plot_filepath)

        plt.show()

    @classmethod
    def generate_animation(cls):
        """Create a 3D animation of change events for different epochs."""

        file_paths = []  # To store file paths

        # Set output directories
        output_dir = os.path.join(cls.plot.plot_dir, "4d_plots")
        os.makedirs(output_dir, exist_ok=True)

        images_dir = os.path.join(output_dir, "frames")
        os.makedirs(images_dir, exist_ok=True)

        # Initialize variables for bounding box dimensions
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')

        # Iterate over point cloud files to find bounding box dimensions
        for i, pc_file in enumerate(cls.change_event.pc_files):
            point_cloud = read_point_cloud_from_xyz(pc_file)
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

        # Find global min and max values of the plotting attribute
        for i, pc_file in enumerate(cls.change_event.pc_files):
            point_cloud = read_point_cloud_from_xyz(pc_file)
            if point_cloud is not None:
                attribute_values = point_cloud[:, cls.plot.plotting_attribute_index]
                min_value, max_value = np.min(attribute_values), np.max(attribute_values)
                global_min_value = min(global_min_value, min_value)
                global_max_value = max(global_max_value, max_value)

        # Create 3D scatter plots for each epoch
        for i, pc_file in enumerate(cls.change_event.pc_files):
            point_cloud = read_point_cloud_from_xyz(pc_file)
            if point_cloud is not None:
                x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
                attribute_values = point_cloud[:, cls.plot.plotting_attribute_index]

                # Set up 3D scatter plot
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                # Create scatter plot with color mapping
                scatter = ax.scatter(x, y, z, c=attribute_values, cmap='viridis', vmin=global_min_value,
                                     vmax=global_max_value)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title(
                    f"{cls.plot.plotting_attribute} of change event with object_id = {cls.change_event.object_id} in epoch {i}")

                # Set plot limits
                ax.set_xlim(min_x, min_x + max_range)
                ax.set_ylim(min_y, min_y + max_range)
                ax.set_zlim(min_z, min_z + max_range)

                # Add color bar and labels
                cbar = fig.colorbar(scatter)
                cbar.set_label(cls.plot.plotting_attribute)

                # Save plot as an image file
                plot_filepath = os.path.join(images_dir, f"frame_{i:04d}.png")
                plt.savefig(plot_filepath)
                file_paths.append(plot_filepath)
                plt.close(fig)

        # Create animation from saved frames
        create_change_animation(file_paths, cls.plot.plotting_attribute, cls.change_event.object_id,
                                output_dir, cls.plot.anim_duration)


class LinePlot2d(BaseModel):
    """A pydantic class representing a 2D line plot configuration.

    Attributes:
        plot_type (str): Type of the plot.
        plot_title (Union[str, None]): Title of the plot (can be None if not provided).
        plot_dir (str): Directory where the plot will be saved.
        plotting_period_start_end (List[str]): List of two strings representing the start and end of the plotting period.
        fig_size (Union[Tuple[int, int], None]): Size of the figure (either Tuple or None, default is (15, 15)).
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
        line_width (int): Width of the line in the plot.
    """
    plot_type: str
    plot_title: Union[str, None]
    plot_dir: str
    plotting_period_start_end: List[str]
    fig_size: Union[Tuple[int, int], None] = (15, 15)  # either Tuple or None, default = (15, 15)
    x_label: str
    y_label: str
    line_width: int


class Plot3d(BaseModel):
    """A class representing a 3D plot configuration.

    Attributes:
        plot_type (str): Type of the plot.
        plot_title (Union[str, None]): Title of the plot (can be None if not provided).
        plot_dir (str): Directory where the plot will be saved.
        plotting_period_start_end (List[str]): List of two strings representing the start and end of the plotting period.
        fig_size (Union[Tuple[int, int], None]): Size of the figure (either Tuple or None, default is (15, 15)).
        plotting_attribute (str): Attribute used for coloring points in the 3D plot.
        plotting_attribute_index (PositiveInt): Index of the attribute used for plotting.
    """

    plot_type: str
    plot_title: Union[str, None]
    plot_dir: str
    plotting_period_start_end: List[str]
    fig_size: Union[Tuple[int, int], None] = (15, 15)
    plotting_attribute: str
    plotting_attribute_index: PositiveInt


class Animation(BaseModel):
    """A class representing an animation configuration.

     Attributes:
         plot_type (str): Type of the plot.
         plot_title (Union[str, None]): Title of the plot (can be None if not provided).
         plot_dir (str): Directory where the plot will be saved.
         plotting_period_start_end (List[str]): List of two strings representing the start and end of the plotting period.
         fig_size (Union[Tuple[int, int], None]): Size of the figure (either Tuple or None, default is (15, 15)).
         anim_duration (PositiveInt): Duration of the animation in frames.
         plotting_attribute (str): Attribute used for coloring points in the animation.
         plotting_attribute_index (PositiveInt): Index of the attribute used for plotting.
    """
    plot_type: str
    plot_title: Union[str, None]
    plot_dir: str
    plotting_period_start_end: List[str]
    fig_size: Union[Tuple[int, int], None] = (15, 15)
    anim_duration: PositiveInt
    plotting_attribute: str
    plotting_attribute_index: PositiveInt


def read_plotting_parameters(plotting_parameters_json):
    """Read plotting parameters from a JSON file.

    :param plotting_parameters_json (str):
        Path to the JSON file containing plotting parameters.

    :return plot_type, json_data (Tuple[str, dict])
        A tuple containing the plot type and the dictionary of plotting parameters.
    """
    with open(plotting_parameters_json, 'r') as file:
        json_data = json.load(file)
        plot_type = json_data.get('plot_type')

        valid_plot_types = ["line_plot_2d", "plot_3d", "animation"]

        if plot_type not in valid_plot_types:
            raise ValueError(f"Invalid plot_type '{plot_type}'. Allowed values are: {', '.join(valid_plot_types)}")
        return plot_type, json_data


###########################################

#            ChangeEventClass             #

###########################################

# Class definition of ChangeEvent

class ChangeEvent(BaseModel):
    """Class representing a change event.

     Attributes:
         object_id (int): Identifier for the change event.
         pc_files (List[str]): List of file paths to point cloud data.
         timestamps (List[str]): List of timestamps associated with each epoch.
         epoch_count (int): Number of epochs in the change event.
         change_rates (List[float]): List of change rates for each epoch.
         duration (float): Duration of the change event.
         change_magnitudes (List[float]): List of change magnitudes for each epoch.
         spatial_extent (List[List[float]]): List of lists representing spatial extent for each epoch.
         event_type (str): Type of the change event.
         plots_2d_dir (List[str]): List of directories for 2D plots.
         plots_3d_dir (List[str]): List of directories for 3D plots.
         plots_4d_dir (List[str]): List of directories for 4D plots.
     """
    object_id: int
    pc_files: List[str]
    timestamps: List[str]
    epoch_count: int
    change_rates: List[float]
    duration: float
    change_magnitudes: List[float]
    spatial_extent: List[List[float]]
    event_type: str
    plots_2d_dir: List[str]
    plots_3d_dir: List[str]
    plots_4d_dir: List[str]


class ChangeEventValidator:
    """Validator class for ChangeEvent objects.

    Attributes:
        change_event (ChangeEvent): Instance of the ChangeEvent class to be validated.
    """

    change_event: ChangeEvent

    @classmethod
    def validate_and_create_change_events(cls, json_data):
        """Validate and create ChangeEvent objects from a list of JSON data.

         :param json_data (List[Dict]):
             List of dictionaries containing information about change events.

         :return Optional[List[ChangeEvent]]:
             List of validated ChangeEvent objects or None if validation fails.
         """
        change_events = []
        try:
            for element in json_data:
                validated_data = validate_model(ChangeEvent, element)

                # Create ChangeEvent object
                change_event = ChangeEvent(**validated_data)
                change_events.append(change_event)

            return change_events

        except ValidationError as e:
            print(f"Error with validation: {e}")
            return None

###########################################

#            Helper functions             #

###########################################
def read_point_cloud_from_xyz(pc_file
                              ):
    """Load a point cloud from a xyz file.

    :param pc_file (str):
        Path to the ascii file.

    :return point_cloud (NumPy array):
        A NumPy array representing the point cloud.
    """

    with open(pc_file, "r") as file:
        lines = file.readlines()

    point_cloud = []
    for line in lines:
        values = line.split()
        x, y, z, distance = map(float, values)
        point_cloud.append([x, y, z, distance])

    return np.array(point_cloud)


def read_visualization_filter_parameters(json_path):
    """Extract visualization parameters from a JSON file.

    :param: json_path (str):
        The filename of the input JSON file.

    :return parameters (dict):
        A dictionary containing the extracted visualization parameters.
    """

    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)

    parameters = {
        "observation_period": None,
        "duration_min_max_sec": None,
        "spatial_extent": None,
        "event_types": None,
    }

    if 'features' in json_data and json_data['features']:
        feature = json_data['features'][0]
        if 'properties' in feature:
            properties = feature['properties']
            parameters["observation_period"] = properties.get('observation_period')
            parameters["duration_min_max_sec"] = properties.get('duration')
            parameters["spatial_extent"] = properties.get('spatial_extent')
            parameters["event_types"] = properties.get('event_types')

    return parameters


def filter_relevant_events(event_list,
                           visualization_filter_parameters
                           ):
    """Retrieve a list of relevant ChangeEvent objects based on criteria defined in visualization filtering parameters.

    :param: event_list (list):
        A list of ChangeEvent objects.

    :param visualization_filter_parameters (dict):
        A dictionary with all parameters for filtering relevant ChangeEvent objects.

    :return relevant_events (list):
        A list of relevant ChangeEvent objects.
   """

    relevant_events = event_list

    # Check all change events for matching the provided criteria
    for key, value in visualization_filter_parameters.items():
        if key == "event_types":
            relevant_events = [event for event in relevant_events if
                               any(event_type in event.event_type for event_type in value)]
        elif key == "duration":
            relevant_events = [event for event in relevant_events if value[0] <= event.duration <= value[1]]
        # further criteria to be implemented
        # elif key == "spatial_extent":
        # relevant_events = [event for event in relevant_events if is_inside_extent(event.spatial_extent, value)]
        # elif key == "observation_period":
        # relevant_events = [event for event in relevant_events if
        # event.timestamps[0] >= value[0] and event.timestamps[-1] <= value[1]]

    return relevant_events


def create_change_animation(images_paths,
                            attribute_name,
                            object_id,
                            plot_dir,
                            duration
                            ):
    """Generate animated plots for each ChangeEvent based on a list of input frames.

    :param images_paths (list):
        A list of file paths of plots to be animated.

    :param attribute_name (str):
        The attribute used to color the ChangeEvent objects in the plots.

    :param object_id (int):
        The id of the ChangeEvent object to be plotted.

    :param plots_4d_dir (str, optional):
        The directory where 4D plots will be saved. If None, a default directory structure will be created.

    :return anim_file_path (str):
        The directory of the generated animation file.

    """

    images = []
    output_dir = os.path.join(plot_dir, "animation")
    os.makedirs(output_dir, exist_ok=True)

    plot_filename = f"ChangeEvent__00{object_id}_animation_{attribute_name}.gif"
    anim_file_path = os.path.join(output_dir, plot_filename)
    for image in images_paths:
        images.append(imageio.imread(image))

    imageio.mimsave(anim_file_path, images, duration=duration)
    return anim_file_path


def is_inside_extent(event_extent, target_extent):
    """Check if an event extent is inside the target extent polygon.

    :param event_extent (list):
        The extent of the ChangeEvent as a list of 4 coordinate pairs.

    :param target_extent (list):
        The target extent polygon as a list of coordinate pairs.

    :return is_inside (bool):
        True if the event extent is inside the target extent polygon, False otherwise.
    """

    # Convert the event extent and target extent to Shapely Polygon objects
    event_polygon = Polygon(event_extent)
    target_polygon = Polygon(target_extent)

    # Check if the event extent is completely within the target extent polygon
    is_inside = event_polygon.within(target_polygon)

    return is_inside


def read_json_file(file_path):
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        return json_data
    except Exception as e:
        print(f"Fehler beim Lesen der JSON-Datei: {e}")
        return None


def main():
    # Read data from user
    visualization_filter_parameters_json = sys.argv[1]
    plotting_parameters_json = sys.argv[2]
    change_events_json = sys.argv[3]

    # Get visualization filtering parameters
    visualization_filter_parameters = read_visualization_filter_parameters(visualization_filter_parameters_json)

    # Read change event data, validate json file and generate ChangeEvent objects
    json_data = read_json_file(change_events_json)
    change_events = ChangeEventValidator.validate_and_create_change_events(json_data)

    # Filter relevant ChangeEvent objects for visualization
    relevant_change_events = filter_relevant_events(change_events, visualization_filter_parameters)
    print(relevant_change_events)

    # Generate plots for all change events after validating plotting parameters
    plot_type, json_data = read_plotting_parameters(plotting_parameters_json)
    for change_event in relevant_change_events:
        plot_object = Plot()
        plot_object.generate_plot(plot_type, json_data, change_event)


if __name__ == "__main__":
    main()
