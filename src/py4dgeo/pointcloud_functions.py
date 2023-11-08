# import required modules
import laspy # import the laspy package for handling point cloud files
import numpy as np # import numpy for array handling

def read_las(infile,get_attributes=False,use_every=1):
    '''
    Function to read coordinates and optionally attribute information of point cloud data from las/laz file.

    :param infile: specification of input file (format: las or laz)
    :param get_attributes: if True, will return all attributes in file, otherwise will only return coordinates (default is False)
    :param use_every: value specifies every n-th point to use from input, i.e. simple subsampling (default is 1, i.e. returning every point)
    :return: Array of point coordinates of shape (N,3) with N number of points in input file (or subsampled by 'use_every')
    '''

    # read the file using the laspy read function
    indata = laspy.read(infile)

    # get the coordinates (XYZ) and stack them in a 3D array
    coords = np.vstack((indata.x, indata.y, indata.z)).transpose()

    # subsample the point cloud, if use_every = 1 will remain the full point cloud data
    coords = coords[::use_every, :]

    # read attributes if get_attributes is set to True
    if get_attributes == True:
        # get all attribute names in the las file as list
        las_fields= list(indata.points.point_format.dimension_names)

        # create a dictionary to store attributes
        attributes = {}

        # loop over all available fields in the las point cloud data
        for las_field in las_fields[3:]: # skip the first three fields, which contain coordinate information (X,Y,Z)
            attribute = np.array(indata.points[las_field]) # transpose shape to (N,1) to fit coordinates array
            if np.sum(attribute)==0: # if field contains only 0, it is empty
                continue
            # add the attribute to the dictionary with the name (las_field) as key
            attributes[las_field] = attribute[::use_every] # subsample by use_every, corresponding to point coordinates

        # return coordinates and attribute data
        return (coords, attributes)

    else: # get_attributes == False
        return (coords) # return coordinates only

def write_las(outpoints,outfilepath,attribute_dict={},correct_wkt_entry=True):

    '''
    :param outpoints: 3D array of points to be written to output file
    :param outfilepath: specification of output file (format: las or laz)
    :param attribute_dict: dictionary of attributes (key: name of attribute; value: 1D array of attribute values in order of points in 'outpoints'); if not specified, dictionary is empty and nothing is added
    :return: None
    '''
	
    # create a header for new las file
    hdr = laspy.LasHeader(version="1.4", point_format=6)

    # set the coordinate resolutions and offset in the header
    hdr.x_scale = 0.00025
    hdr.y_scale = 0.00025
    hdr.z_scale = 0.00025
    mean_extent = np.mean(outpoints, axis=0)
    hdr.x_offset = int(mean_extent[0])
    hdr.y_offset = int(mean_extent[1])
    hdr.z_offset = int(mean_extent[2])

    # create the las data
    las = laspy.LasData(hdr)

    # write coordinates into las data
    las.x = outpoints[:, 0]
    las.y = outpoints[:, 1]
    las.z = outpoints[:, 2]
    
    # add all dictionary entries to las data (if available)
    for key,vals in attribute_dict.items():
        if not key in las:
            las.add_extra_dim(laspy.ExtraBytesParams(
            name=key,
            type=type(vals[0])
            ))
        las[key] = vals

    # write las file
    las.write(outfilepath)
    
    # this is required because alobal encoding WKT flag must be set for point format 6 - 10 since las 1.4
    # otherwise programs such as pdal will not be able to read the file
    if correct_wkt_entry:
        filename = outfilepath
        f = open(filename, "rb+")
        f.seek(6)
        f.write(bytes([17, 0, 0, 0]));
        f.close()

    return
    
def transform_points(points, trafomat, reduction_point = [.0,.0,.0]):
    '''
    Applies a rigid transformation, i.e. rotation and translation, to 3D point data.
    :param points: 2D array of 3D points with shape (N,3)
    :param trafomat: 2D array of rigid transformation matrix with 3x3 rotation and 1x3 translation parameters
    :return: transformed points in 2D array of shape (N,3)
    '''
    
    rotation = np.array(trafomat)[:, :3]
    translation = np.array(trafomat)[:, 3].flatten()

    points -= centroid
    pts_rot = points.dot(rotation.T)
    pts_trafo = pts_rot + translation
    points_out = pts_trafo[:,:3] + centroid
    
    return points_out