###########################################

#               Usage                 #

###########################################

# python change_visualization.py "change_visualization_config.json""

# demo data: https://github.com/3dgeo-heidelberg/py4dgeo/tree/visualization_module/demo/change_visualization
###########################################


from __future__ import annotations
import sys
import imageio.v2 as imageio
import numpy as np
from shapely.geometry import Polygon
from pydantic import ValidationError
import json
import os
from typing import List, Union
from typing import List, Tuple
from pydantic import BaseModel, PositiveInt, validate_model, ValidationError
from typing import List
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.lines import Line2D
from PIL import Image
import pickle


###########################################

#               PlotClass                 #

###########################################
class Plot:
    """
       A class for generating different types of plots.

       Attributes:
           plot (Union[SpatialOverview, None]): The plot object of the respective pydantic class and a ChangeEvent object.
           change_event (ChangeEvent): The ChangeEvent object associated with the plot.
    """
    # Get plot Object of the respective pydantic class and a ChangeEvent object
    plot: Union[SpatialOverview] = None
    change_event: ChangeEvent = None

    @classmethod
    def generate_plot(cls, plot_type, json_data, relevant_change_events, plotting_period, event_types):
        """Validate JSON file content and generate a Plot object for a ChangeEvent object based on the plot type defined
        in the input JSON.

        :param plot_type: The plot type to be generated.
        :type plot_type: str
        :param json_data: The content of the input json file as dictionary.
        :type json_data: dict
        :param relevant_change_events: The ChangeEvent object(s) to be plotted.
        :type relevant_change_events: List[ChangeEvent]
        :param plotting_period: The period for plotting.
        :type plotting_period: List[str]
        :param event_types: The types of events to be plotted.
        :type event_types: List[str]
        """

        # Takes ALL relevant ChangeEvent objects as input
        if plot_type == 'spatial_overview':
            # Initialize SpatialOverview plot object and generate specific plot type
            cls.plot = SpatialOverview(**json_data)
            cls.generate_spatial_overview(plotting_period, relevant_change_events, event_types)

    @classmethod
    def generate_spatial_overview(cls, plotting_period, change_events, event_types):
        """
        Generate a spatial overview plot.

        :param plotting_period:
            The period for plotting.
        :type plotting_period: List[str]
        :param change_events:
            The list of ChangeEvent objects.
        :type change_events: List[ChangeEvent]
        :param event_types:
            The types of events to be plotted.
        :type event_types: List[str]
        """

        change_event_locations = get_change_event_locations(change_events)

        # Define output
        cls.plot.plot_folder_full_path = os.path.abspath(
            os.path.join('..', '..', '..', 'output_change_visualization'))

        # Check for plot filename, create if not provided
        if cls.plot.plot_filename == 'null':
            plotting_period_start = plotting_period[0]
            plotting_period_end = plotting_period[1]
            event_types_str = ""
            for idx, event_type in enumerate(event_types):
                if idx > 0:
                    event_types_str += "+"
                event_types_str += event_type
            cls.plot.plot_filename = str(plotting_period_start+"_"+plotting_period_end+"_spatial_overview_"
                                         +event_types_str+".png")

        # Check for existing folder structure, create if not existing
        os.makedirs(cls.plot.plot_folder_full_path, exist_ok=True)
        cls.plot.outfile = os.path.join(cls.plot.plot_folder_full_path, cls.plot.plot_filename)

        # Create a 2D axis
        fig, ax = plt.subplots()

        # Load range image from file
        current_directory = os.path.abspath(__file__)
        range_image_filepath = os.path.abspath(
            os.path.join(current_directory, '..', '..', '..', cls.plot.range_image[1:])
        )
        range_image = plt.imread(range_image_filepath)

        # Determine size of range image
        num_phi, num_theta = range_image.shape[:2]

        # Plot range image
        im = ax.imshow(range_image, cmap=plt.cm.gray, extent=[0, num_theta, 0, num_phi], origin='upper')

        # Set the background color
        fig.set_facecolor(cls.plot.background_color)

        # Mark provided ChangeEvent objects as red points with labels'
        for i, obj_coords in enumerate(change_event_locations, start=1):
            ax.scatter(obj_coords[0], obj_coords[1], c=cls.plot.event_marker_color, marker=cls.plot.event_marker_symbol)

        # Axis labels
        ax.set_xlabel(cls.plot.x_label, color=cls.plot.x_label_color)
        ax.set_ylabel(cls.plot.y_label, color=cls.plot.y_label_color)

        # Title
        bbox_props_title = dict(boxstyle=cls.plot.bbox_title_boxstyle, fc=cls.plot.bbox_title_fontcolor,
                                ec="none", alpha=cls.plot.bbox_title_alpha)
        # Place text on the figure level
        fig.text(cls.plot.title_text_x, cls.plot.title_text_y, cls.plot.title_text, color=cls.plot.title_fontcolor,
                 fontsize=cls.plot.title_fontsize, va=cls.plot.title_va, ha=cls.plot.title_ha, bbox=bbox_props_title)

        # Legend for ChangeEvent objects and selected time period in a joint box
        plotting_period_label = str(plotting_period[0] + " to " + plotting_period[1])
        cls.plot.legend_text = f'Change events: {event_types}\nTime period: {plotting_period_label}'
        bbox_props_legend = dict(boxstyle=cls.plot.bbox_legend_boxstyle, fc=cls.plot.bbox_legend_fontcolor,
                                 ec="none", alpha=cls.plot.bbox_legend_alpha)

        # Dummy element to add the symbol to the legend
        custom_legend1 = [
            Line2D([0], [0], marker=cls.plot.event_marker_symbol, color=cls.plot.event_marker_color,
                   label=cls.plot.legend_text, markersize=8, linestyle='None')]

        # Add legend with 'loc' option
        ax.legend(handles=custom_legend1, bbox_to_anchor=(0.6, -0.4))
        ax.text(1.1, 0.6, '', transform=ax.transAxes, color='black', fontsize=10, va='center', ha='center',
                bbox=bbox_props_legend)

        # Set axis color to white
        ax.spines['bottom'].set_color(cls.plot.axis_spines_bottom_color)
        ax.spines['top'].set_color(cls.plot.axis_spines_top_color)
        ax.spines['right'].set_color(cls.plot.axis_spines_right_color)
        ax.spines['left'].set_color(cls.plot.axis_spines_left_color)

        # Set ticks to white
        ax.tick_params(axis=cls.plot.axis_ticks, colors=cls.plot.axis_ticks_color)

        # Rotate Y-axis labels by 90 degrees
        ax.set_yticklabels(ax.get_yticklabels(), rotation=cls.plot.axis_x_rotate)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=cls.plot.axis_y_rotate)

        # Insert logos of AImon5.0 project
        logos_filepath = os.path.abspath(
            os.path.join(current_directory, '..', '..', '..', cls.plot.logos[1:])
        )
        img = plt.imread(logos_filepath)
        imagebox = OffsetImage(img, zoom=cls.plot.logo_zoom)  # Adjust the zoom factor
        logos_image_position = (cls.plot.logo_x, cls.plot.logo_y)
        ab = AnnotationBbox(imagebox, logos_image_position, frameon=cls.plot.annotation_bbox_frameon,
                            xycoords=cls.plot.annotation_bbox_xycoords, boxcoords=cls.plot.annotation_bbox_coords)
        ax.add_artist(ab)

        # Save plot as a PNG file
        plt.savefig(cls.plot.outfile, dpi=cls.plot.img_res_dpi)

        img = Image.open(cls.plot.outfile)

        # Crop image
        cropped_img = img.crop((cls.plot.image_crop[0], cls.plot.image_crop[1], cls.plot.image_crop[2],
                                cls.plot.image_crop[3]))
        cropped_img.save(cls.plot.outfile)

        # Show plot
        plt.show()

#  @classmethod
#   def generate_temporal_overview(cls, change_event_data, plotting_period):
class SpatialOverview(BaseModel):
    """A pydantic class representing a spatial overview plot configuration.

    Attributes:
        plot_type (str): Type of the plot.
        plot_dir (str): Directory where the plot will be saved.
        outfile (str): Outfile name.
        range_image (str): Range image name.
        logos (str): Logos name.
        plot_folder_full_path (str): Full path of the plot folder.
        plot_filename (str): Name of the plot file.
        background_color (str): Background color of the plot.
        event_marker_symbol (str): Symbol for event marker.
        event_marker_color (str): Color for event marker.
        x_label (str): Label for x-axis.
        y_label (str): Label for y-axis.
        x_label_color (str): Color for x-axis label.
        y_label_color (str): Color for y-axis label.
        axis_spines_bottom_color (str): Color for bottom axis spine.
        axis_spines_top_color (str): Color for top axis spine.
        axis_spines_right_color (str): Color for right axis spine.
        axis_spines_left_color (str): Color for left axis spine.
        axis_ticks (str): Ticks for the axis.
        axis_ticks_color (str): Color for axis ticks.
        axis_x_rotate (str): Rotation for x-axis labels.
        axis_y_rotate (int): Rotation for y-axis labels.
        title_text (str): Text for the title.
        title_text_x (float): X-coordinate for title text.
        title_text_y (float): Y-coordinate for title text.
        title_boxstyle (str): Box style for title.
        title_fontcolor (str): Font color for title.
        title_fontsize (str): Font size for title.
        title_alpha (float): Alpha value for title.
        title_va (str): Vertical alignment for title.
        title_ha (str): Horizontal alignment for title.
        bbox_title_boxstyle (str): Box style for title bounding box.
        bbox_title_fontcolor (str): Font color for title bounding box.
        bbox_title_alpha (float): Alpha value for title bounding box.
        bbox_legend_boxstyle (str): Box style for legend bounding box.
        bbox_legend_fontcolor (str): Font color for legend bounding box.
        bbox_legend_fontsize (int): Font size for legend bounding box.
        bbox_legend_alpha (float): Alpha value for legend bounding box.
        plotting_period_label (str): Label for plotting period.
        legend_text (str): Text for legend.
        logo_zoom (float): Zoom factor for logo.
        logo_x (float): X-coordinate for logo.
        logo_y (float): Y-coordinate for logo.
        annotation_bbox_frameon (bool): Whether to draw the bounding box for annotation.
        annotation_bbox_xycoords (str): XY coordinates for annotation bounding box.
        annotation_bbox_coords (str): Coordinates for annotation bounding box.
        img_res_dpi (int): Resolution in dots per inch for images.
        image_crop (list): List specifying cropping parameters for images.
    """

    plot_type: str
    outfile: Union[str, None]
    range_image: str
    logos: str
    plot_folder_full_path: Union[str, None]
    plot_filename: Union[str, None]
    background_color: str
    event_marker_symbol: str
    event_marker_color: str
    x_label: str
    y_label: str
    x_label_color: str
    y_label_color: str
    axis_spines_bottom_color: str
    axis_spines_top_color: str
    axis_spines_right_color: str
    axis_spines_left_color: str
    axis_ticks: str
    axis_ticks_color: str
    axis_x_rotate: int
    axis_y_rotate: int
    title_text: str
    title_text_x: float
    title_text_y: float
    title_boxstyle: str
    title_fontcolor: str
    title_fontsize: str
    title_alpha: float
    title_va: str
    title_ha: str
    bbox_title_boxstyle: str
    bbox_title_fontcolor: str
    bbox_title_alpha: float
    bbox_legend_boxstyle: str
    bbox_legend_fontcolor: str
    bbox_legend_fontsize: int
    bbox_legend_alpha: float
    plotting_period_label: str
    legend_text: str
    logo_zoom: float
    logo_x: float
    logo_y: float
    annotation_bbox_frameon: bool
    annotation_bbox_xycoords: str
    annotation_bbox_coords: str
    img_res_dpi: int
    image_crop: list


def read_plotting_parameters(plotting_parameters_json):
    """
    Read plotting parameters from a JSON file.

    :param plotting_parameters_json:
        Path to the JSON file containing plotting parameters.
    :type plotting_parameters_json: str

    :return plot_types, json_data:
        A tuple containing the list of plot types and the list of dictionaries of plotting parameters.
    :rtype: Tuple[list, list]
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
         object_id int: Identifier for the change event.
         timestamps List[str]: List of timestamps associated with each epoch.
         pc_dir List[str]: Path to change event point clouds.
         epoch_count int: Number of epochs in the change event.
         change_rates List[float]: List of change rates for each epoch.
         duration_sec float: Duration of the change event.
         change_magnitudes_m List[float]: List of change magnitudes for each epoch.
         spatial_extent_bbox List[List[float]]: List of lists representing spatial extent for each epoch.
         location List[float]: List of 3D coordinates of change event.
         event_types: List[str]: Type of the change event.
     """
    object_id: int
    timestamps: List[str]
    pc_dir: List[str]
    epoch_count: int
    change_rates: List[float]
    duration_sec: float
    change_magnitudes_m: List[float]
    spatial_extent_bbox: List[List[float]]
    location: List[float]
    event_types: List[str]


class ChangeEventValidator:
    """Validator class for ChangeEvent objects.

    Attributes:
        change_event (ChangeEvent): Instance of the ChangeEvent class to be validated.
    """

    change_event: ChangeEvent

    @classmethod
    def validate_and_create_change_events(cls, json_data):
        """Validate and create ChangeEvent objects from a list of JSON data.

        :param json_data:
            List of dictionaries containing information about change events.
        :type json_data: List[Dict]

        :return Optional:
            List of validated ChangeEvent objects or None if validation fails.
        :rtype: Optional[List[ChangeEvent]]
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
def read_point_cloud_from_xyz(pc_file):
    """
    Load a point cloud from a xyz file.

    :param pc_file:
        Path to the ascii file.
    :type pc_file: str

    :return point_cloud:
        A NumPy array representing the point cloud.
    :rtype: numpy.ndarray
    """

    with open(pc_file, "r") as file:
        lines = file.readlines()

    point_cloud = []
    for line in lines:
        values = line.split()
        x, y, z, distance = map(float, values)
        point_cloud.append([x, y, z, distance])

    return np.array(point_cloud)


def read_visualization_filters(json_path):
    """Extract visualization parameters from a JSON file.

    :param json_path:
        The filename of the input JSON file.
    :type json_path: str

    :return parameters:
        A dictionary containing the extracted visualization parameters.
    :rtype: dict
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
                           visualization_filters
                           ):
    """Retrieve a list of relevant ChangeEvent objects based on criteria defined in visualization filtering parameters.

    :param event_list:
        A list of ChangeEvent objects.
    :type event_list: list

    :param visualization_filters:
        A dictionary with all parameters for filtering relevant ChangeEvent objects.
    :type visualization_filters: dict

    :return relevant_events:
        A list of relevant ChangeEvent objects.
    :rtype: list
   """

    relevant_events = event_list

    # Check all change events for matching the provided criteria
    for key, value in visualization_filters.items():
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
    """
    Generate animated plots for each ChangeEvent based on a list of input frames.

    :param images_paths:
        A list of file paths of plots to be animated.
    :type images_paths: list

    :param attribute_name:
        The attribute used to color the ChangeEvent objects in the plots.
    :type attribute_name: str

    :param object_id:
        The id of the ChangeEvent object to be plotted.
    :type object_id: int

    :param plots_4d_dir: str, optional
        The directory where 4D plots will be saved. If None, a default directory structure will be created.
    :type plots_4d_dir: str or None

    :return anim_file_path:
        The directory of the generated animation file.
    :rtype: str
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

    :param event_extent:
        The extent of the ChangeEvent as a list of 4 coordinate pairs.
    :type event_extent: list

    :param target_extent:
        The target extent polygon as a list of coordinate pairs.
    :type target_extent: list

    :return:
        True if the event extent is inside the target extent polygon, False otherwise.
    :rtype: bool
    """

    # Convert the event extent and target extent to Shapely Polygon objects
    event_polygon = Polygon(event_extent)
    target_polygon = Polygon(target_extent)

    # Check if the event extent is completely within the target extent polygon
    is_inside = event_polygon.within(target_polygon)

    return is_inside


def get_change_event_locations(change_events):
    """Get the 2D locations of the ChangeEvent objects.

    :param change_events:
        A list of ChangeEvent objects.
    :type change_events: list

    :return locations:
        A list of tuples representing the 2D locations of the ChangeEvent objects.
    :rtype: list
    """

    change_event_locations = []
    for change_event in change_events:
        location = change_event.location
        change_event_locations.append(location)
    return change_event_locations


def read_json_file(file_path):
    """Read JSON data from a file.

    :param file_path:
        The path to the JSON file.
    :type file_path: str

    :return:
        A dictionary containing the JSON data if successful, None otherwise.
    :rtype: dict or None
    """

    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        return json_data
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None


def main():
    """Main function to execute change visualization.

    This function reads the configuration file from the command-line argument and performs change visualization
    based on the provided configuration.

    Usage:
        python change_visualization.py change_visualization_config.json

    Returns:
        bool: True if change visualization is successful, False otherwise.
    """
    try:
        if len(sys.argv) != 2:
            print("Usage: python change_visualization.py change_visualization_config.json")
            return

        # Read the JSON configuration file
        config_json_path = sys.argv[1]
        config_data = read_json_file(config_json_path)

        # Extract absolute paths to input JSON files
        visualization_filters_json = config_data.get("visualization_filters_json")
        change_events_json = config_data.get("change_events_json")
        plot_config_json = config_data.get("plot_config_json")

        if config_data is None:
            print("Error reading configuration file.")
            return

        # Get visualization filtering parameters
        visualization_filter_parameters_dict = read_visualization_filters(visualization_filters_json)

        # Read change event data, validate json file and generate ChangeEvent objects
        change_events_dict = read_json_file(change_events_json)
        change_events = ChangeEventValidator.validate_and_create_change_events(change_events_dict)

        # Filter relevant ChangeEvent objects for visualization
        relevant_change_events = filter_relevant_events(change_events, visualization_filter_parameters_dict)

        # Generate plots with all relevant change events after validating plotting parameters
        plotting_period = visualization_filter_parameters_dict["observation_period"]
        event_types = visualization_filter_parameters_dict["event_types"]

        plot_types, json_data = read_plotting_parameters(plot_config_json)
        for idx, plot_type in enumerate(plot_types):
            json_data = json_data[idx]
            plot_object = Plot()
            plot_object.generate_plot(plot_type, json_data, relevant_change_events, plotting_period, event_types)
        return True

    except Exception as err:
        print(f"Change visualization failed: {err}")


if __name__ == "__main__":
    main()
