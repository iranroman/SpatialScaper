"""Argument parser functions."""

import argparse
import sys
import yaml


def parse_args():
    """
    Parse the following arguments for a default parser RoomScaper users.
    Args:
        cfg (str): path to the config file.
    """
    parser = argparse.ArgumentParser(
        description="Provide RoomScaper data synthesis pipeline."
    )
    parser.add_argument(
        "--config",
        dest="path_to_config",
        help="Path to the config files",
        default="configs/RoomScaper/ICASSP_2024.yaml",
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args, path_to_config=None):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument):
            `args`
            `path_to_config`
    """
    with open(path_to_config, "r") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return dotdict(cfg)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
