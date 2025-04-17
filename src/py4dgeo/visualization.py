import os
import laspy
import numpy as np
import cv2
import rasterio
import fiona
import json

from scipy.spatial import ConvexHull
from shapely.geometry import mapping, Polygon
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom

from pyproj import Transformer

from py4dgeo.util import xyz_2_spherical, Py4DGeoError

from py4dgeo.change_events import ChangeEventCollection


class PCloudProjection:
    """
    Point Cloud Projection Module.

    This module processes point clouds, creating 2D projections (color and range images)
    from either a top-down or scanner-based perspective.

    Classes:
        - PCloudProjection: Main class for handling point cloud processing and projection.

    Methods:
        - __init__: Initializes the PCloudProjection class with configuration parameters.
        - project_pc: Main function to execute the projection process.
        - load_pc_file: Loads point cloud data from .las or .laz files.
        - main_projection: Projects the point cloud into 2D image space.
        - create_shading: Calculates surface normals for image shading.
        - apply_shading_to_color_img: Applies lighting effects to color images.
        - apply_shading_to_range_img: Applies lighting effects to range images.
        - apply_smoothing: Smoothens images using Gaussian blur.
        - save_image: Saves generated images with metadata.
    """

    def __init__(
        self,
        epoch,
        projected_image_folder,
        project_name = "Untitled",
        cloud_path = "Unknown",
        make_range_image = True,
        make_color_image = False,
        resolution_cm = 4,
        rgb_light_intensity = 100,
        range_light_intensity = 100,
        apply_shading = True
    ):
        self.xyz = epoch.cloud
        self.camera_position = epoch.scanpos_info
        self.projected_image_folder = projected_image_folder
        self.project_name = project_name
        self.cloud_path = cloud_path
        self.make_range_image = make_range_image
        self.make_color_image = make_color_image
        self.resolution_cm = resolution_cm
        self.rgb_light_intensity = rgb_light_intensity
        self.range_light_intensity = range_light_intensity
        self.apply_shading = apply_shading


        if epoch.scanpos_info is None or len(epoch.scanpos_info) != 3:
            raise Py4DGeoError("Scan position needed. Please provide with epoch.scanpos_info")

        self.data_prep()
        self.main_projection()
        self.create_shading()
        if self.make_color_image:
            self.apply_shading_to_color_img()
            self.save_image()
        if self.make_range_image:
            self.apply_shading_to_range_img()
            self.save_image()


    # Define a function to remove isolated black pixels - Only for RGB image
    def remove_isolated_black_pixels(self, image, threshold=np.array([0.0, 0.0, 0.0])):
        """Function to process each pixel neighborhood"""

        # Convert the image in float
        image = image.astype(np.float32)
        # Create a kernel to compute the mean of neighboring pixels (same weight each)
        # And exclude the center pixel
        kernel = np.ones((3, 3), np.float32) / 8.0
        kernel[1, 1] = 0

        # Split the image into its color channels
        channels = cv2.split(image)

        # Apply the convolution to each channel (RGB) separately
        mean_channels = [cv2.filter2D(channel, -1, kernel) for channel in channels]

        # Merge the channels back together
        mean_image = cv2.merge(mean_channels)

        # Identify black pixels (all channels are zero)
        # A pixel is black if each of its RGB value is less than 41
        try:
            black_pixels_mask = np.all(image <= [40, 40, 40], axis=-1)
        except:
            # If the image only has one chanel (black and white image)
            black_pixels_mask = np.any(image <= 40, axis=-1)

        # Replace black pixels with the corresponding values from the mean
        image[black_pixels_mask] = mean_image[black_pixels_mask]

        # Convert image back to integer
        result_image = np.clip(image, 0, 255).astype(np.uint8)

        return result_image


    def save_image(self):
        # Save image with the current time
        if not os.path.exists(self.projected_image_folder):
            os.makedirs(self.projected_image_folder)
        filename = os.path.join(self.projected_image_folder,f"{self.project_name}_{self.image_type}Image.tif")

        raster = np.moveaxis(self.shaded_image, [0, 1, 2], [2, 1, 0])
        raster = np.rot90(raster, k=-1, axes=(1, 2))
        raster = np.flip(raster, axis=2)

        meta = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': None,
            'height': self.shaded_image.shape[0],
            'width': self.shaded_image.shape[1],
            'count': 3,  # number of bands
            "tiled": False,
            "compress": 'lzw'
        }
        
        custom_tags = {
                "pc_path": self.cloud_path,
                "image_path": filename,
                "make_range_image": self.make_range_image,
                "make_color_image": self.make_color_image,
                "resolution_cm": self.resolution_cm,
                "camera_position_x": self.camera_position[0],
                "camera_position_y": self.camera_position[1],
                "camera_position_z": self.camera_position[2],
                "pc_mean_x": self.mean_x,
                "pc_mean_y": self.mean_y,
                "pc_mean_z": self.mean_z,
                "rgb_light_intensity": self.rgb_light_intensity,
                "range_light_intensity": self.range_light_intensity,
                "h_img_res": self.h_img_res,
                "v_img_res": self.v_img_res,
                "h_fov_x": self.h_fov[0],
                "h_fov_y": self.h_fov[1],
                "v_fov_x": self.v_fov[0],
                "v_fov_y": self.v_fov[1],
                "res": self.v_res
            }

        # Write the raster
        with rasterio.open(filename, "w", **meta) as dest:
            dest.write(raster, [1,2,3])
            dest.update_tags(**custom_tags)


    def data_prep(self):
        # Load the .las/.laz file
        # with laspy.open(self.pc_path) as las_file:
        #     self.las_f = las_file.read()
        # x = np.array(self.las_f.x)
        # y = np.array(self.las_f.y)
        # z = np.array(self.las_f.z)
        if self.make_color_image:
            self.red = np.array(self.las_f.red)
            self.green = np.array(self.las_f.green)
            self.blue = np.array(self.las_f.blue)

            # Normalize RGB values if necessary (assuming they are in the range 0-65535)
            if self.red.max() > 255:
                self.red = (self.red / 65535.0 * 255).astype(np.uint8)
                self.green = (self.green / 65535.0 * 255).astype(np.uint8)
                self.blue = (self.blue / 65535.0 * 255).astype(np.uint8)

        #self.xyz = np.vstack((x, y, z)).T

        # Computing xyz coord means
        self.mean_x = np.mean(self.xyz[:,0])
        self.mean_y = np.mean(self.xyz[:,1])
        self.mean_z = np.mean(self.xyz[:,2])


    def main_projection(self):
        # Shift the point cloud by the camera position' coordinates so the latter is positionned on the origin
        self.xyz -= self.camera_position
        # Range between camera and the mean point of the point cloud
        range = np.sqrt(
            (
                (self.camera_position[0] - self.mean_x) ** 2
                + (self.camera_position[1] - self.mean_y) ** 2
                + (self.camera_position[2] - self.mean_z) ** 2
            )
        )
        # Getting vertical and horizontal resolutions in degrees. Both calculated with the range and the pixel dimension
        alpha_rad = np.arctan2(self.resolution_cm / 100, range)
        self.v_res = self.h_res = np.rad2deg(alpha_rad)

        # Get spherical coordinates
        r, theta, phi = xyz_2_spherical(self.xyz)  # Outputs r, theta (radians), phi (radians)
        # Convert radians to degrees
        theta_deg, phi_deg = np.rad2deg(theta), np.rad2deg(phi)

        # Discretize angles to image coordinates
        if np.floor(min(theta_deg)) == -180 and np.floor(max(theta_deg)) == 180:
            mask = theta_deg < 0
            theta_deg[mask] += 360
        
        self.h_fov = (np.floor(min(theta_deg)), np.ceil(max(theta_deg)))


        if np.floor(min(phi_deg)) == -180 and np.floor(max(phi_deg)) == 180:
            mask = phi_deg < 0
            phi_deg[mask] += 360
        
        self.v_fov = (np.floor(min(phi_deg)), np.ceil(max(phi_deg)))

        self.h_img_res = int((self.h_fov[1] - self.h_fov[0]) / self.h_res)
        self.v_img_res = int((self.v_fov[1] - self.v_fov[0]) / self.v_res)

        # Initialize range and color image
        self.range_image = np.full(
            (self.h_img_res, self.v_img_res, 3), 0, dtype=np.float32
        )
        self.color_image = np.full(
            (self.h_img_res, self.v_img_res, 3), 0, dtype=np.uint8
        )

        # Map angles to pixel indices
        u = np.round((theta_deg - self.h_fov[0]) / self.h_res).astype(int)
        v = np.round((phi_deg - self.v_fov[0]) / self.v_res).astype(int)

        # Filter points within range
        valid_indices = (
            (u >= 0) & (u < self.h_img_res) & (v >= 0) & (v < self.v_img_res)
        )
        self.u = u[valid_indices]
        self.v = v[valid_indices]
        self.r = r[valid_indices]
        self.r = (self.r-np.min(self.r))*255/np.max(self.r-np.min(self.r))
        if self.make_color_image:
            self.red = self.red[valid_indices]
            self.green = self.green[valid_indices]
            self.blue = self.blue[valid_indices]


    def create_shading(self):
        # Compute surface normals' components (gradient approximation)
        z_img = np.zeros((self.h_img_res, self.v_img_res))
        z_img[self.u, self.v] = self.r
        dz_dv, dz_du = np.gradient(z_img)

        # Compute normals with components
        self.normals = np.dstack((-dz_du, -dz_dv, np.ones_like(z_img)))
        self.norms = np.linalg.norm(self.normals, axis=2, keepdims=True)
        self.normals /= np.abs(self.norms)  # Normalize


    def apply_shading_to_color_img(self):
        # Populate
        self.color_image[self.u, self.v, 0] = self.red
        self.color_image[self.u, self.v, 1] = self.green
        self.color_image[self.u, self.v, 2] = self.blue
        # Compute shading (Lambertian model)
        # Light direction for the image to have the right shading
        light_dir_x = abs(self.camera_position[0] - self.mean_x)
        light_dir_y = abs(self.camera_position[1] - self.mean_y)
        light_dir_z = abs(self.camera_position[2] - self.mean_z)
        light_direction = np.array(
            [light_dir_x, light_dir_y, light_dir_z]
        )  # Direction of the light source
        light_direction = light_direction / np.linalg.norm(light_direction)  # Normalize

        dot_product = np.sum(self.norms * light_direction, axis=2)
        shading = np.clip(dot_product * self.rgb_light_intensity, 0, 1)

        if self.apply_shading:
            # Apply smoothed shading to the color image
            shaded_color_image = (self.color_image.astype(np.float32) * shading[..., np.newaxis])
        else:
            shaded_color_image = self.color_image.astype(np.float32)
        
        shaded_color_image = np.clip(shaded_color_image, 0, 255).astype(np.uint8)

        # Apply median filter to selectively remove isolated black pixels
        shaded_color_image = self.remove_isolated_black_pixels(shaded_color_image)

        self.shaded_image = self.apply_smoothing(shaded_color_image)

        # Call save_image function
        self.image_type = "Color"


    def apply_shading_to_range_img(self):
        # Populate the range image with the radius (scanner to point distance)
        self.range_image[self.u, self.v, 0] = \
            self.range_image[self.u, self.v, 1] = \
            self.range_image[self.u, self.v, 2] = \
        self.r+ self.range_light_intensity
        
        if self.apply_shading:
            # Shade the range image with the normals
            shaded_range_image = (
                self.range_image.astype(np.float32)
                * (
                    self.normals[:, :, -1] + self.normals[:, :, -2] + self.normals[:, :, -3]
                )[..., np.newaxis]
            )
        else:
            shaded_range_image = self.range_image.astype(np.float32)

        filter_255 = shaded_range_image>255.
        shaded_range_image[filter_255] = 255.

        filter_0 = shaded_range_image<0.
        shaded_range_image[filter_0] = 0.

        self.shaded_image = self.apply_smoothing(shaded_range_image)

        # Call save_image function
        self.image_type = "Range"

        
    def apply_smoothing(self, input_image):
        blur = cv2.GaussianBlur(input_image, (3, 3), 0)
        # Flip the image left to right
        output_image = np.fliplr(np.asarray(blur))

        return output_image



class ProjectChange:
    """
    Change Projection Module.

    This module processes spatial change events, projects them onto images,
    and generates GeoJSON and kml files for visualization in GIS tools.

    Classes:
        - ProjectChange: Handles loading change data, projecting onto images, 
                        and generating GeoJSON outputs.

    Methods:
        - __init__: Initializes the ProjectChange class with input parameters.
        - project_change: Main function to project changes and create GeoJSON files.
        - project_gis_layer: Helper function to handle GIS layer projection.
    """

    def __init__(self, change_event_file, project_name, projected_image_folder, projected_events_folder, epsg=4979):
        ##############################
        ### INITIALIZING VARIABLES ###
        self.project = project_name
        self.bg_img_folder = projected_image_folder
        self.path_change_events = change_event_file
        self.img = None
        self.pts = []
        self.geojson_name = os.path.join(projected_events_folder,"%s_change_events_pixel.geojson"%self.project)
        self.geojson_name_gis = os.path.join(projected_events_folder,"%s_change_events_gis.geojson"%self.project)
        self.epsg = epsg
        ##############################
        if not os.path.isdir(self.bg_img_folder):
            print("Missing some information, cannot find %s"%self.bg_img_folder)
            return 
        else:
            self.bg_img_path = os.path.join(self.bg_img_folder, os.listdir(self.bg_img_folder)[0])


    def project_change(self):
        
        # Get change events dictionnary in json file
        # change_events = utilities.read_json_file(self.path_change_events)
        change_events = ChangeEventCollection()
        change_events = change_events.load_from_file(self.path_change_events)
        # Create the schema for the attributes of the geojson
        schema = {
            'geometry': 'Polygon',
            'properties': {
                'event_type': 'str',
                'object_id': 'str',
                'X_centroid': 'float',
                'Y_centroid': 'float',
                'Z_centroid': 'float',
                't_min': 'str',
                't_max': 'str',
                'change_magnitudes_mean': 'float',
                'volumes_from_convex_hulls': 'float',
                'cluster_point_cloud': 'str',
                'cluster_point_cloud_chull': 'str'
                }
            }
        # Open the shapefile to be able to write each polygon in it
        geojson = fiona.open(self.geojson_name, 'w', 'GeoJSON', schema, fiona.crs.CRS.from_epsg(self.epsg), 'binary')
        geojson_gis = fiona.open(self.geojson_name_gis, 'w', 'GeoJSON', schema, fiona.crs.CRS.from_epsg(self.epsg))
        for change_event in change_events.events:
            
            #if 'undefined' in str(change_event['event_type']): continue

            # Fetch contour points in WGS84 coordinate system
            change_event_pts_og = change_event.convex_hull["points_building"]
            change_event_pts_og = np.asarray(change_event_pts_og)
            
            # Handle the empty array, if any
            if change_event_pts_og.shape[0] == 0:
                continue
            
            # GIS layer
            self.project_gis_layer(change_event_pts_og)
            # Add the polygon to the main geojson file
            geojson_gis.write({
                'geometry': mapping(self.polygon_gis),
                'properties': {
                    'event_type': str(change_event.event_type),
                    'object_id': str(change_event.object_id),
                    'X_centroid': float(self.centroid_gis[0]),
                    'Y_centroid': float(self.centroid_gis[1]),
                    'Z_centroid': float(self.centroid_gis[2]),
                    't_min': str(change_event.t_min),
                    't_max': str(change_event.t_max),
                    'change_magnitudes_mean': float(change_event.change_magnitudes["mean"]),
                    'volumes_from_convex_hulls': float(change_event.convex_hull["volume"]),
                    'cluster_point_cloud': str(change_event.cluster_point_cloud),
                    'cluster_point_cloud_chull': str(change_event.cluster_point_cloud_chull)
                }
            })
        geojson_gis.close()

        # Load EXIF data from an image
        try:
            # Retrieve the metadata
            #self.bg_img_path = os.path.join(os.getcwd(), self.bg_img_path)
            with rasterio.open(self.bg_img_path) as src:
                image_metadata_loaded = dict(src.tags().items())
        except:
            print("Missing some information, cannot project change into image")
            return

        # Get metadata of the image. Necessary for the projection of the change event points
        pc_mean_x = float(image_metadata_loaded['pc_mean_x'])
        pc_mean_y = float(image_metadata_loaded['pc_mean_y'])
        pc_mean_z = float(image_metadata_loaded['pc_mean_z'])
        camera_position_x = float(image_metadata_loaded['camera_position_x'])
        camera_position_y = float(image_metadata_loaded['camera_position_y'])
        camera_position_z = float(image_metadata_loaded['camera_position_z'])
        h_img_res = float(image_metadata_loaded['h_img_res'])
        v_img_res = float(image_metadata_loaded['v_img_res'])
        h_fov_x = float(image_metadata_loaded['h_fov_x'])
        h_fov_y = float(image_metadata_loaded['h_fov_y'])
        v_fov_x = float(image_metadata_loaded['v_fov_x'])
        v_fov_y = float(image_metadata_loaded['v_fov_y'])
        res = float(image_metadata_loaded['res'])
        
        for change_event in change_events.events:
            #if 'undefined' in str(change_event['event_type']): continue

            # Fetch contour points in WGS84 coordinate system
            change_event_pts_og = change_event.convex_hull["points_building"]
            change_event_pts_og = np.asarray(change_event_pts_og)
            
            # Handle the empty array, if any
            if change_event_pts_og.shape[0] == 0:
                continue
            
            change_event_pts = change_event_pts_og.copy()
            
            # Translation of point cloud coordinates for the scanner position of (0, 0, 0)
            change_event_pts = change_event_pts - np.asarray([camera_position_x, camera_position_y, camera_position_z])

            # Transformation from cartesian coordinates (x, y, z) to spherical coordinates (r, θ, φ)
            r, theta, phi = xyz_2_spherical(change_event_pts)
            theta, phi = np.rad2deg(theta), np.rad2deg(phi)

            # Transformation from spherical coordinates (r, θ, φ) to pixel coordinates (u, v)
            u = np.round((theta - h_fov_x) / res).astype(int)
            v = np.round((phi - v_fov_x) / res).astype(int)
            change_points_uv = np.c_[u, v]

            # Create the convex hull
            hull = ConvexHull(change_points_uv)

            # Order the points anti-clockwise
            list_points = []
            for simplex in hull.vertices:
                list_points.append([int(v_img_res - change_points_uv[simplex, 1]), -int(change_points_uv[simplex, 0])])
            
            # Create the polygon
            polygon = Polygon(np.array(list_points))

            # Compute centroid
            centroid = np.mean(change_event_pts_og, axis=0)

            # Add the polygon to the main shapefile
            geojson.write({
                'geometry': mapping(polygon),
                'properties': {
                    'event_type': str(change_event.event_type),
                    'object_id': str(change_event.object_id),
                    'X_centroid': float(centroid[0]),
                    'Y_centroid': float(centroid[1]),
                    'Z_centroid': float(centroid[2]),
                    't_min': str(change_event.t_min),
                    't_max': str(change_event.t_max),
                    'change_magnitudes_mean': float(change_event.change_magnitudes["mean"]),
                    'volumes_from_convex_hulls': float(change_event.convex_hull["volume"]),
                    'cluster_point_cloud': str(change_event.cluster_point_cloud),
                    'cluster_point_cloud_chull': str(change_event.cluster_point_cloud_chull)
                }
            })
        geojson.close()

        self.geojson2kml()


    def project_gis_layer(self, change_event_pts_og):
        change_event_pts_xy = change_event_pts_og[:,:2]
        # Create the convex hull
        hull = ConvexHull(change_event_pts_xy)
        # Order the points anti-clockwise
        list_points = []
        for simplex in hull.vertices:
            list_points.append([int(change_event_pts_xy[simplex, 1]), -int(change_event_pts_xy[simplex, 0])])
        
        # Create the polygon
        list_points = np.asarray(list_points)
        list_points.T[[0, 1]] = list_points.T[[1, 0]]
        list_points[:, 0] *= -1
        self.polygon_gis = Polygon(np.array(list_points))
        # Compute centroid
        self.centroid_gis = np.mean(change_event_pts_og, axis=0)


    def geojson2kml(self):
        self.kml_name_gis = self.geojson_name_gis.replace('.geojson', ".kml")
        self.kml_name_gis = f"{os.path.abspath('.')}/{self.kml_name_gis}"

        with open(self.geojson_name_gis, 'r') as file:
            geojson_data = json.load(file)

        transformer = Transformer.from_crs(f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True)
        #############################

        # Initialize KML structure
        kml = Element('kml', xmlns="http://www.opengis.net/kml/2.2")
        document = SubElement(kml, 'Document', id="root_doc")

        # Folder for Placemarks
        folder = SubElement(document, 'Folder')
        folder_name = SubElement(folder, 'name')
        folder_name.text = "change_events_gis"

        # Fetch fields
        field_names = list(geojson_data['features'][0]['properties'].keys())
        fields = []

        # Generate Placemarks from GeoJSON
        for feature in geojson_data.get("features", []):
            properties = feature.get("properties", {})
            geometry = feature.get("geometry", {})
            
            for field in field_names:
                propertie = str(properties.get(field, "string"))
                fields.append((field, propertie))
            
            # Extract coordinates
            coordinates = ""
            if geometry.get("type") == "Polygon":
                for ring in geometry.get("coordinates", []):
                    for lon, lat in ring:
                        lon, lat = transformer.transform(lon, lat)
                        coordinates += f"{lon},{lat} "
            
            # Create Placemark
            placemark = SubElement(folder, 'Placemark')
            style = SubElement(placemark, 'Style')
            line_style = SubElement(style, 'LineStyle')
            line_width = SubElement(line_style, 'width')
            line_width.text = "4"
            line_color = SubElement(line_style, 'color')
            line_color.text = "ff0000ff"
            poly_style = SubElement(style, 'PolyStyle')
            fill = SubElement(poly_style, 'fill')
            fill.text = "0"
            
            extended_data = SubElement(placemark, 'ExtendedData')
            schema_data = SubElement(extended_data, 'SchemaData', schemaUrl="#change_events_gis")
            
            # Add properties to SchemaData
            for name, value in fields:
                simple_data = SubElement(schema_data, 'SimpleData', name=name)
                simple_data.text = value

            # Add Polygon geometry
            if coordinates:
                polygon = SubElement(placemark, 'Polygon')
                outer_boundary = SubElement(polygon, 'outerBoundaryIs')
                linear_ring = SubElement(outer_boundary, 'LinearRing')
                coord_element = SubElement(linear_ring, 'coordinates')
                coord_element.text = coordinates.strip()
        
        ########################
        
        # Beautify the output XML
        kml_str = xml.dom.minidom.parseString(tostring(kml)).toprettyxml(indent="  ")
        with open(self.kml_name_gis, 'w') as file:
            file.write(kml_str)

        #############################