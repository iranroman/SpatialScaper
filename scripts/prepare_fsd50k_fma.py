import argparse

import os
import soundata
import requests
import shutil
import zipfile
from tqdm import tqdm
import pandas as pd
#import yaml
#import json
import librosa
#import soundata
import numpy as np
#import pandas as pd
#from prepare_utils import *

SOUND_EVENT_DATASETS_SUBDIR = 'sound_event_datasets'

class BaseDataSetup:
    def __init__(self, dataset_home=None, metadata_path=None):
        self.dataset_home = dataset_home
        self.metadata_path = metadata_path


FMA_REMOTES = {
    "name": "fma_small",
    "filename": "fma_small.zip",
    "base_url": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
    "metadata_url": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
}

CORRUPT_FMA_TRACKS = ["098565", "098567", "098569", "099134", "108925", "133297"]


def extract_zip(zip_path, destination):
    print(f"Extracting zip: {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destination)

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

def check_dataset_exists(dataset_dir):
    return os.path.exists(dataset_dir) and os.listdir(dataset_dir)

class FMADataSetup(BaseDataSetup):
    def __init__(self, ntracks_genre=10, split_prob=0.6, **kwargs):
        super().__init__(**kwargs)
        self.ntracks_genre = ntracks_genre
        self.split_prob = split_prob
        self.dataset_name = FMA_REMOTES["name"]
        self.base_url = FMA_REMOTES["base_url"]
        self.metadata_url = FMA_REMOTES["metadata_url"]
        self.zip_name = FMA_REMOTES["filename"]
        self.fma_to_dcase_dict = {}

    def prepare_dataset(self):
        # Check if fma_small exists, else create it
        if not check_dataset_exists(os.path.join(self.dataset_home, self.dataset_name)):
            print("Downloading FMA small dataset...")
            self.download_dataset()
        else:
            print("FMA small dataset already exists. Skipping download.")
        self.gen_dataset_splits()

    def download_dataset(self):
        os.makedirs(self.dataset_home, exist_ok=True)
        # download fma_small zip
        download_file(self.base_url, os.path.join(self.dataset_home, self.zip_name))
        extract_zip(os.path.join(self.dataset_home, self.zip_name), self.dataset_home)
        print("Done unzipping")
        # download fma metadata
        print("Downloading fma_small metadata...")
        download_file(
            self.metadata_url, os.path.join(self.dataset_home, 'meta_'+self.zip_name)
        )
        extract_zip(os.path.join(self.dataset_home, 'meta_'+self.zip_name), self.dataset_home)
        print("Done unzipping")

    def gen_dataset_splits(self):
        path_to_fsd50k_dcase = os.path.join(self.dataset_home, 'FSD50K_FMA_DCASE')
        tracks = pd.read_csv(
            os.path.join(self.dataset_home, "fma_metadata/tracks.csv"),
            header=[0, 1],
            index_col=0,
        )
        genres = tracks["track"]["genre_top"].unique()
        # Loop through the genre 8 classes
        for genre in genres:
            if genre != genre:
                continue
            os.makedirs(os.path.join(path_to_fsd50k_dcase,'music','train',genre),exist_ok=True)
            os.makedirs(os.path.join(path_to_fsd50k_dcase,'music','test',genre),exist_ok=True)
            # Get tracks by genre, consider only set from "small" tracks
            genre_tracks = tracks[
                (tracks["track", "genre_top"] == genre)
                & (tracks["set", "subset"] == "small")
            ]
            # Get ntracks_genre from the current genre
            tracks = genre_tracks[: self.ntracks_genre]
            for track_id, track in tracks.iterrows():
                if (
                    f"{track_id:06}" in CORRUPT_FMA_TRACKS
                ):  
                    # skip cor
                    # see: https://github.com/mdeff/fma/wiki#known-issues-errata, https://github.com/mdeff/fma/issues/49rupt tracks from fma_small
                    continue  

                # get track name by id
                subdir = f"{track_id:06}"[:3]
                fma_track_path = os.path.join(
                    self.dataset_home, self.dataset_name, subdir, f"{track_id:06}.mp3"
                )
                # Define DCASE path format variables
                super_class = "music"
                fold = "train" if np.random.rand() < self.split_prob else "test"
                class_name = str(genre)
                dcase_path = os.path.join(
                    path_to_fsd50k_dcase, super_class, fold, str(genre), f"{track_id:06}.mp3"
                )
                shutil.copyfile(fma_track_path,dcase_path)



class FSD50KDataSetup(BaseDataSetup):
    def __init__(
        self,
        dataset_name="fsd50k",
        download=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.download = download
        self.url_fsd_selected_txt = "https://zenodo.org/record/6406873/files/FSD50K_selected.txt"

        # Remove 'dataset_home' from kwargs before passing to FMADataLoad
        if "dataset_home" in kwargs:
            del kwargs["dataset_home"]

    def download_dataset(self):
        self.fsd50k.download()

    def prepare_dataset(self):
        assert (
            self.dataset_home is not None or not self.download
        ), "Dataset home path must be provided when download is enabled."
        self.fsd50k = soundata.initialize(self.dataset_name, self.dataset_home)
        if self.download:
            self.download_dataset()  # download fsd50k

    def download_txt(self,url):
        """Download file from a given URL"""
        response = requests.get(url)
        response.raise_for_status()  # Check if the download was successful
        return response.text.splitlines()

    def to_DCASE_format(self):
        # Download the lines from the URL
        lines = self.download_txt(self.url_fsd_selected_txt)

        # Loop through each line in the text
        for line in lines:
            new_dir = os.path.dirname(line.strip())
            filename = os.path.basename(line.strip())
            # Retrieve the new source directory for the file
            if "train" in new_dir:
                source = os.path.join(self.dataset_home, "FSD50K.dev_audio")
            elif "test" in new_dir:
                source = os.path.join(self.dataset_home, "FSD50K.eval_audio")
            else:
                raise ValueError("Invalid directory structure in the text file")
            # Full path of the source file
            src_file = os.path.join(source, filename)
            # New directory, check if it doesn't exist
            dest_path = os.path.join(self.dataset_home,'FSD50K_FMA_DCASE')
            os.makedirs(os.path.join(dest_path, new_dir), exist_ok=True)
            # Copy file to new directory
            shutil.copy(src_file, os.path.join(dest_path, new_dir, filename))

    def cleanup(self,target_subdir='FSD50K_FMA_DCASE'):
        for subdir in os.listdir(self.dataset_home):
            # Construct the full path to the subdirectory
            full_path = os.path.join(self.dataset_home, subdir)

            # Check if it is a directory and not the target directory
            if os.path.isdir(full_path) and subdir != target_subdir:
                # Remove the directory
                shutil.rmtree(full_path)
                print(f"Deleted: {full_path}")

        # Delete non-matching files
        for file in os.listdir(self.dataset_home):
            full_path = os.path.join(self.dataset_home, file)
            if os.path.isfile(full_path) and not glob.fnmatch.fnmatch(file, target_subdir):
                os.remove(full_path)
                print(f"Deleted file: {full_path}")

def prepare_fsd50k(args):
    fsd50k = FSD50KDataSetup(
        dataset_name="fsd50k",
        download=args.download_FSD,
        dataset_home=os.path.join(args.data_dir,SOUND_EVENT_DATASETS_SUBDIR)
        )
    fsd50k.prepare_dataset()
    fsd50k.to_DCASE_format()
    if args.cleanup:
        fsd50k.cleanup()
    return fsd50k

def prepare_fma(args):
    fma = FMADataSetup(
        dataset_home=os.path.join(args.data_dir,SOUND_EVENT_DATASETS_SUBDIR)
    )
    fma.prepare_dataset()


if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser(description='Process the data_dir argument.')

    # Add a data_dir argument
    parser.add_argument('--data_dir', type=str, default='datasets',help='Path to the data directory')
    parser.add_argument('--download_FSD', type=str, default='True', help='Whether to download the FSD50K dataset')
    parser.add_argument('--download_FMA', type=str, default='True', help='Whether to download the FMA dataset')
    parser.add_argument('--cleanup', type=str, default='False', help='Whether to download the FMA dataset')


    # Parse the arguments
    args = parser.parse_args()
    if args.download_FSD == 'True':
        args.download_FSD = True
    else:
        args.download_FSD = False
    if args.download_FMA == 'True':
        args.download_FMA = True
    else:
        args.download_FMA = False
    if args.cleanup == 'True':
        args.cleanup = True
    else:
        args.cleanup = False

    #prepare_fsd50k(args)

    prepare_fma(args)
