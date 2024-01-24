<<<<<<< scripts/prepare_utils.py
import os
import requests
import tarfile
import zipfile
from tqdm import tqdm

def extract_tar(tar_path, destination):
    print(f"Extracting tar: {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(destination)

def extract_zip(zip_path, destination):
    print(f"Extracting zip: {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination)

def check_dataset_exists(dataset_dir):
    return os.path.exists(dataset_dir) and os.listdir(dataset_dir)

def download_file(url, local_dest_path):
    response = requests.get(url, stream=True)

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # Adjust the block size as per your requirement

    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

    with open(local_dest_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()
=======
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
>>>>>>> scripts/utils_tmp.py
