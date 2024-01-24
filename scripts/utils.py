import os
import requests
import tarfile
import zipfile
import argparse
import sys
import numpy as np
from tqdm import tqdm


def extract_tar(tar_path, destination):
    print(f"Extracting tar: {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(destination)


def extract_zip(zip_path, destination):
    """Extracts a zip file to the given destination."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(destination)
    except zipfile.BadZipFile:
        raise ValueError("The provided file is not a valid zip file.")


def download_file(url, local_dest_path):
    """Downloads a file from a URL to a local destination."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
        with open(local_dest_path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download file: {e}")


def map_to_cylinder(path, rad, axis=2):
    # maps points (unit vecs) to cylinder of known radius along axis (default z/2)
    # scaled_path = np.empty(path.shape)
    scales = np.empty(path.shape[0])
    # define axes perpendicular to the cylinder
    rad_axes = [0, 1, 2]
    rad_axes.remove(axis)

    # iterate through path and project point
    for i in range(path.shape[0]):
        vec = path[i]
        scale_rad = np.sqrt(np.sum([vec[j] ** 2 for j in rad_axes]))
        scale = rad / scale_rad
        scales[i] = scale
        # scaled_path[i] = vec * scale
    return scales  # scaled_path


def get_y(angle, x):
    angle2 = np.pi - angle - np.pi / 2
    return x * np.sin(angle) / np.sin(angle2)
