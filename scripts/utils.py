import os
import requests
import tarfile
import zipfile
import argparse
import sys
import subprocess
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


def combine_multizip(filename, destination, shell=True):
    subprocess.run(
        f"zip -s 0 {filename} --out {destination}",
        shell=shell,
    )
