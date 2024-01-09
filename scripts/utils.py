import argparse
import sys
import numpy as np

def parse_args():
    """
    parse paths to specific files
    """
    parser = argparse.ArgumentParser(
        description="Provide a path."
    )
    parser.add_argument(
        "--path",
        dest="path",
        type=str,
        help="path to relevant files",
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()

def map_to_cylinder(path, rad, axis=2):
    #maps points (unit vecs) to cylinder of known radius along axis (default z/2)
    #scaled_path = np.empty(path.shape)
    scales = np.empty(path.shape[0])
    #define axes perpendicular to the cylinder
    rad_axes = [0,1,2]
    rad_axes.remove(axis)
    
    #iterate through path and project point
    for i in range(path.shape[0]):
        vec = path[i]
        scale_rad = np.sqrt(np.sum([vec[j]**2 for j in rad_axes]))
        scale = rad / scale_rad
        scales[i] = scale
        #scaled_path[i] = vec * scale
    return scales#scaled_path

def get_y(angle,x):
    angle2 = np.pi-angle-np.pi/2
    return x * np.sin(angle) / np.sin(angle2)
