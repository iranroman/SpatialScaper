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

def get_y(angle,x):
    angle2 = np.pi-angle-np.pi/2
    return x * np.sin(angle) / np.sin(angle2)
