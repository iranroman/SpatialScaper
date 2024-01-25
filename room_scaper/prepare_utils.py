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
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destination)


def check_dataset_exists(dataset_dir):
    return os.path.exists(dataset_dir) and os.listdir(dataset_dir)


def download_file(url, local_dest_path):
    response = requests.get(url, stream=True)

    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # Adjust the block size as per your requirement

    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

    with open(local_dest_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()
