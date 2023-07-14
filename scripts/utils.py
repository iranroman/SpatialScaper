import argparse
import sys

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
