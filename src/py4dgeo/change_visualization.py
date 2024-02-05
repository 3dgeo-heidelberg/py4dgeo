###########################################

#               Usage                 #

###########################################

# python change_visualization.py "visualization_filters.JSON" "change_events.json" "plot_config.json"

# demo data: https://github.com/3dgeo-heidelberg/py4dgeo/tree/visualization_module/demo/change_visualization
###########################################


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
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D
from PIL import Image


###########################################

#               PlotClass                 #

###########################################
class Plot:
    # Get plot Object of the respective pydantic class and a ChangeEvent object
    plot: Union[LinePlot2d, Plot3d, Animation, SpatialOverview] = None
    change_event: ChangeEvent = None

    @classmethod
    def generate_plot(cls, plot_type, json_data, change_event, relevant_change_events, plotting_period):
        """Validate JSON file content and generate a Plot object for a ChangeEvent object based on the plot type defined
        in the input JSON.

        :param plot_type:
            The plot type to be generated.

        :param json_data:
            The content of the input json file as dictionary.

        :param change_event:
            The ChangeEvent object(s) to be plotted.
        """

        # Initialize plot object of respective pydantic plot class and validate json
        # Takes only 1 ChangeEvent object as input
        if plot_type == 'line_plot_2d':
            cls.plot = LinePlot2d(**json_data)
            cls.change_event = change_event  # Set the change_event for the class
            cls.generate_line_plot_2d()
        # Takes only 1 ChangeEvent object as input
        if plot_type == 'plot_3d':
            cls.plot = Plot3d(**json_data)
            cls.change_event = change_event  # Set the change_event for the class
            cls.generate_plot_3d()
        # Takes only 1 ChangeEvent object as input
        if plot_type == 'animation':
            cls.plot = Animation(**json_data)
            cls.change_event = change_event  # Set the change_event for the class
            cls.generate_animation()
        # Takes ALL relevant ChangeEvent objects as input
        if plot_type == 'spatial_overview':
            cls.plot = SpatialOverview(**json_data)
            # cls.change_event.generate_spatial_overview(plotting_period, change_events)
            cls.generate_spatial_overview(plotting_period, relevant_change_events)

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

    @classmethod
    def generate_spatial_overview(cls, plotting_period, change_events):

        change_event_locations = get_change_event_locations(change_events)

        # Fix plot elements:

        range_image = r"J:\01_Projekte\AIMon50\Module_development\visualization_module\range_image_generation\test_shd_mean.png"
        logos = r"J:\01_Projekte\AIMon50\Module_development\visualization_module\input_change_vis\logos.png"

        # Define output
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.abspath(os.path.join(current_script_path, '..', '..', '..', '..'))
        plot_folder = "output_change_visualization\\end_user_plots\\spatial_overview"
        plot_folder_full_path = os.path.join(plot_path, plot_folder)

        # Überprüfe und erstelle übergeordnete Ordner, falls sie nicht existieren
        os.makedirs(plot_folder_full_path, exist_ok=True)

        outfile = os.path.join(plot_folder_full_path, "spatial_overview.png")


        # Load range image from file
        range_image = plt.imread(range_image)

        # Determine size of range image
        num_phi, num_theta = range_image.shape[:2]

        # Create a 2D axis
        fig, ax = plt.subplots()

        # Set the background color
        fig.set_facecolor('#A9A9A9')  # Hex code for medium dark gray

        # Plot range image
        im = ax.imshow(range_image, cmap=plt.cm.gray, extent=[0, num_theta, 0, num_phi], origin='upper')

        # Mark provided ChangeEvent objects as red points with labels
        for i, obj_coords in enumerate(change_event_locations, start=1):
            ax.scatter(obj_coords[0], obj_coords[1], c='red', marker='o')

        # Axis labels
        ax.set_xlabel('Distance [m]', color='white')
        ax.set_ylabel('Height [m]', color='white')

        # Color bar for the range image with label "Range"
        # cbar = plt.colorbar(im, ax=ax, label='Range')
        # cbar.ax.yaxis.label.set_color('white')

        # Title
        title_text = cls.plot.plot_title
        bbox_props_title = dict(boxstyle="round", fc="white", ec="none", alpha=0.7)
        # Place text on the figure level
        fig.text(0.5, 0.69, title_text, color='black', fontsize=12, va='center', ha='center', bbox=bbox_props_title)

        # Legend for ChangeEvent objects and selected time period in a joint box
        plotting_period_label = str(plotting_period[0] + " to " + plotting_period[1])
        legend_text = f'Rockfall events\nTime period: {plotting_period_label}'
        bbox_props_legend = dict(boxstyle="round", fc="white", ec="none", alpha=0.7)

        # Dummy element to add the symbol to the legend
        custom_legend1 = [
            Line2D([0], [0], marker='o', color='red', label=legend_text, markersize=8, linestyle='None')]

        # Add legend with 'loc' option
        ax.legend(handles=custom_legend1, bbox_to_anchor=(0.6, -0.4))
        ax.text(1.1, 0.6, '', transform=ax.transAxes, color='black', fontsize=10, va='center', ha='center',
                bbox=bbox_props_legend)

        # Set axis color to white
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('white')

        # Set ticks to white
        ax.tick_params(axis='both', colors='white')

        # Rotate Y-axis labels by 90 degrees
        ax.set_yticklabels(ax.get_yticklabels(), rotation=90)

        # Insert logos of AImon5.0 project
        img = plt.imread(logos)
        imagebox = OffsetImage(img, zoom=0.26)  # Adjust the zoom factor
        logos_image_position = (0.5, 1.2)
        ab = AnnotationBbox(imagebox, logos_image_position, frameon=False, xycoords='axes fraction',
                            boxcoords='axes fraction')
        ax.add_artist(ab)

        # Save plot as a PNG file
        plt.savefig(outfile, dpi=600)

        img = Image.open(outfile)

        # Get pixel coordinates for cropping image to content
        # width, height = img.size
        # x_position_percent = 1  # Adjust the percentage of the image width
        # y_position_percent = 0.85  # Adjust the percentage of the image height
        # Calculate pixel coordinates based on percentages
        # x_position = int(width * x_position_percent)
        # y_position = int(height * y_position_percent)
        # print(x_position, y_position)

        # Crop image
        cropped_img = img.crop((0, 720, 3840, 2448))
        cropped_img.save(outfile)

        # Show plot
        plt.show()


class LinePlot2d(BaseModel):
    """A pydantic class representing a 2D line plot configuration.

    Attributes:
        plot_type (str): Type of the plot.
        plot_title (Union[str, None]): Title of the plot (can be None if not provided).
        plot_dir (str): Directory where the plot will be saved (can be None if not provided)
        plotting_period_start_end (List[str]): List of two strings representing the start and end of the plotting period.
        fig_size (Union[Tuple[int, int], None]): Size of the figure (either Tuple or None, default is (15, 15)).
        x_label (str): Label for the X-axis.
        y_label (str): Label for the Y-axis.
        line_width (int): Width of the line in the plot.
    """
    plot_type: str
    plot_title: Union[str, None]
    plot_dir: Union[str, None]
    plotting_period_start_end: List[str]
    x_label: str
    y_label: str
    line_width: int


class Plot3d(BaseModel):
    """A class representing a 3D plot configuration.

    Attributes:
        plot_type (str): Type of the plot.
        plot_title (Union[str, None]): Title of the plot (can be None if not provided).
        plot_dir (str): Directory where the plot will be saved (can be None if not provided)
        plotting_period_start_end (List[str]): List of two strings representing the start and end of the plotting period.
        fig_size (Union[Tuple[int, int], None]): Size of the figure (either Tuple or None, default is (15, 15)).
        plotting_attribute (str): Attribute used for coloring points in the 3D plot.
        plotting_attribute_index (PositiveInt): Index of the attribute used for plotting.
    """

    plot_type: str
    plot_title: Union[str, None]
    plot_dir: Union[str, None]
    plotting_period_start_end: List[str]
    fig_size: Union[Tuple[int, int], None] = (15, 15)
    plotting_attribute: str
    plotting_attribute_index: PositiveInt


class Animation(BaseModel):
    """A class representing an animation configuration.

     Attributes:
         plot_type (str): Type of the plot.
         plot_title (Union[str, None]): Title of the plot (can be None if not provided).
         plot_dir (str): Directory where the plot will be saved (can be None if not provided)
         plotting_period_start_end (List[str]): List of two strings representing the start and end of the plotting period.
         fig_size (Union[Tuple[int, int], None]): Size of the figure (either Tuple or None, default is (15, 15)).
         anim_duration (PositiveInt): Duration of the animation in frames.
         plotting_attribute (str): Attribute used for coloring points in the animation.
         plotting_attribute_index (PositiveInt): Index of the attribute used for plotting.
    """
    plot_type: str
    plot_title: Union[str, None]
    plot_dir: Union[str, None]
    plotting_period_start_end: List[str]
    fig_size: Union[Tuple[int, int], None] = (15, 15)
    anim_duration: PositiveInt
    plotting_attribute: str
    plotting_attribute_index: PositiveInt


class SpatialOverview(BaseModel):
    """A pydantic class representing a spatial overview plot configuration.

    Attributes:
        plot_type (str): Type of the plot.
        plot_title (Union[str, None]): Title of the plot (can be None if not provided).
        range_image (str): Path to range image to be displayed in plot.
        plot_dir (str): Directory where the plot will be saved (can be None if not provided)
    """

    plot_type: str
    plot_title: str
    plot_dir: str


def read_plotting_parameters(plotting_parameters_json):
    """Read plotting parameters from a JSON file.

    :param plotting_parameters_json (str):
        Path to the JSON file containing plotting parameters.

    :return plot_types, json_data (Tuple[list, list])
        A tuple containing the list of plot types and the list of dictionaries of plotting parameters.
    """
    with open(plotting_parameters_json, 'r') as file:
        json_data = json.load(file)
        plot_types = [element.get('plot_type') for element in json_data]

        valid_plot_types = ["line_plot_2d", "plot_3d", "animation", "spatial_overview", "temporal_overview"]

        for plot_type in plot_types:
            if plot_type not in valid_plot_types:
                raise ValueError(f"Invalid plot_type '{plot_type}'. Allowed values are: {', '.join(valid_plot_types)}")

        return plot_types, json_data


###########################################

#            ChangeEventClass             #

###########################################

# Class definition of ChangeEvent

class ChangeEvent(BaseModel):
    """Class representing a change event.

     Attributes:
         object_id (int): Identifier for the change event.
         timestamps (List[str]): List of timestamps associated with each epoch.
         epoch_count (int): Number of epochs in the change event.
         pc_dir (List[str]): Path to change event point clouds.
         change_rates (List[float]): List of change rates for each epoch.
         duration_sec (float): Duration of the change event.
         change_magnitudes_m (List[float]): List of change magnitudes for each epoch.
         spatial_extent_bbox (List[List[float]]): List of lists representing spatial extent for each epoch.
         location (List[float]): List of 3D coordinates of change event.
         event_types ((List[str]): Type of the change event.
         plots_2d_dir (List[str]): List of directories for 2D plots (can be None if not provided).
         plots_3d_dir (List[str]): List of directories for 3D plots (can be None if not provided).
         plots_4d_dir (List[str]): List of directories for 4D plots (can be None if not provided).
     """
    object_id: int
    timestamps: List[str]
    epoch_count: int
    pc_dir: List[str]
    change_rates: List[float]
    duration_sec: float
    change_magnitudes_m: List[float]
    spatial_extent_bbox: List[List[float]]
    location: List[float]
    event_types: List[str]
    plots_2d_dir: Union[List[str], None]
    plots_3d_dir: Union[List[str], None]
    plots_4d_dir: Union[List[str], None]


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
                change_event = ChangeEvent(**validated_data[0])
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

    expected_parameters = {
        "observation_period": list,
        "duration_min_max_sec": list,
        "spatial_extent_bbox": list,
        "event_types": list,
    }

    for param, expected_type in expected_parameters.items():
        if param not in json_data:
            print(f"Error: Parameter '{param}' is missing.")
            return None
        else:
            param_value = json_data[param]
            if not isinstance(param_value, expected_type):
                print(
                    f"Error: Parameter '{param}' has an unexpected data type. Expected: {expected_type}, Actual: {type(param_value)}")
                return None

    return json_data


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
                               any(event_types in event.event_types for event_types in value)]
        elif key == "duration_min_max_sec":
            relevant_events = [event for event in relevant_events if value[0] <= event.duration_sec <= value[1]]
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


def get_change_event_locations(change_events):
    change_event_locations = []
    for change_event in change_events:
        location = change_event.location
        change_event_locations.append(location)
    return change_event_locations


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
    change_events_json = sys.argv[2]
    plot_config_json = sys.argv[3]

    # Get visualization filtering parameters
    visualization_filter_parameters_dict = read_visualization_filter_parameters(visualization_filter_parameters_json)

    # Read change event data, validate json file and generate ChangeEvent objects
    change_events_dict = read_json_file(change_events_json)
    change_events = ChangeEventValidator.validate_and_create_change_events(change_events_dict)

    # Filter relevant ChangeEvent objects for visualization
    relevant_change_events = filter_relevant_events(change_events, visualization_filter_parameters_dict)

    # Generate plots with all relevant change events after validating plotting parameters
    plotting_period = visualization_filter_parameters_dict["observation_period"]

    plot_types, json_data = read_plotting_parameters(plot_config_json)
    change_event = []
    for idx, plot_type in enumerate(plot_types):
        json_data = json_data[idx]
        plot_object = Plot()
        plot_object.generate_plot(plot_type, json_data, change_event, relevant_change_events, plotting_period)


if __name__ == "__main__":
    main()
