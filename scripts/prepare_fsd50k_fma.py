import argparse
import os
import shutil
import requests
import zipfile
import numpy as np
import pandas as pd
import librosa
import soundata
from tqdm import tqdm

from utils import extract_zip
from utils import download_file

# Constants
SOUND_EVENT_DATASETS_SUBDIR = "sound_event_datasets"
FMA_REMOTES = {
    "name": "fma_small",
    "filename": "fma_small.zip",
    "base_url": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
    "metadata_url": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
}
CORRUPT_FMA_TRACKS = ["098565", "098567", "098569", "099134", "108925", "133297"]
SKIP_GENRES = ["Electronic", "Experimental", "Instrumental"]
DCASE_FSD50K_SELECTED = "https://zenodo.org/record/6406873/files/FSD50K_selected.txt"


# Base class for dataset setup
class BaseDataSetup:
    def __init__(self, dataset_home=None, metadata_path=None):
        self.dataset_home = dataset_home
        self.metadata_path = metadata_path


# FMA dataset setup class
class FMADataSetup(BaseDataSetup):
    def __init__(self, ntracks_genre=20, split_prob=0.6, **kwargs):
        super().__init__(**kwargs)
        self.ntracks_genre = ntracks_genre
        self.split_prob = split_prob
        self.dataset_name = FMA_REMOTES["name"]
        self.base_url = FMA_REMOTES["base_url"]
        self.metadata_url = FMA_REMOTES["metadata_url"]
        self.zip_name = FMA_REMOTES["filename"]

    def prepare_dataset(self):
        """Prepares the FMA dataset by downloading and extracting it."""
        dataset_path = os.path.join(self.dataset_home, self.dataset_name)
        if not os.path.exists(dataset_path):
            print("Downloading FMA small dataset...")
            self.download_dataset()
        else:
            print("FMA small dataset already exists. Skipping download.")
        self.gen_dataset_splits()

    def download_dataset(self):
        """Downloads and extracts the FMA dataset."""
        os.makedirs(self.dataset_home, exist_ok=True)
        download_file(self.base_url, os.path.join(self.dataset_home, self.zip_name))
        extract_zip(os.path.join(self.dataset_home, self.zip_name), self.dataset_home)
        print("Done unzipping")
        # Download and extract metadata
        print("Downloading FMA small metadata...")
        download_file(
            self.metadata_url, os.path.join(self.dataset_home, "meta_" + self.zip_name)
        )
        extract_zip(
            os.path.join(self.dataset_home, "meta_" + self.zip_name), self.dataset_home
        )
        print("Done unzipping metadata")

    def gen_dataset_splits(self, target_subdir="FSD50K_FMA"):
        """Generates dataset splits for training and testing."""
        path_to_fsd50k_dcase = os.path.join(self.dataset_home, target_subdir)
        tracks = pd.read_csv(
            os.path.join(self.dataset_home, "fma_metadata/tracks.csv"),
            header=[0, 1],
            index_col=0,
        )
        genres = tracks["track"]["genre_top"].unique()
        for genre in genres:
            if genre in SKIP_GENRES:
                continue
            if pd.isna(genre):
                continue
            genre_tracks = tracks[
                (tracks["track", "genre_top"] == genre)
                & (tracks["set", "subset"] == "small")
            ]
            selected_tracks = genre_tracks[: self.ntracks_genre]
            if not genre_tracks.empty:
                genre_dir_train = os.path.join(
                    path_to_fsd50k_dcase, "music", "train", genre
                )
                genre_dir_test = os.path.join(
                    path_to_fsd50k_dcase, "music", "test", genre
                )
                os.makedirs(genre_dir_train, exist_ok=True)
                os.makedirs(genre_dir_test, exist_ok=True)
            for track_id, track in selected_tracks.iterrows():
                if str(track_id).zfill(6) in CORRUPT_FMA_TRACKS:
                    continue
                fma_track_path = os.path.join(
                    self.dataset_home,
                    self.dataset_name,
                    str(track_id).zfill(6)[:3],
                    f"{track_id:06}.mp3",
                )
                fold = "train" if np.random.rand() < self.split_prob else "test"
                dcase_path = os.path.join(
                    path_to_fsd50k_dcase, "music", fold, genre, f"{track_id:06}.mp3"
                )
                shutil.copyfile(fma_track_path, dcase_path)

    # Additional methods for FMADataSetup can be added here, Cleanup, for example


class FSD50KDataSetup(BaseDataSetup):
    def __init__(self, dataset_name="fsd50k", download=False, **kwargs):
        """Initializes the FSD50K dataset setup."""
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.download = download
        self.url_fsd_selected_txt = DCASE_FSD50K_SELECTED

    def download_dataset(self):
        """Downloads the FSD50K dataset."""
        if self.download:
            try:
                self.fsd50k.download()
            except Exception as e:
                raise RuntimeError(f"Failed to download FSD50K dataset: {e}")

    def prepare_dataset(self):
        """Prepares the FSD50K dataset by initializing and possibly downloading it."""
        if self.dataset_home is None and self.download:
            raise ValueError(
                "Dataset home path must be provided when download is enabled."
            )
        self.fsd50k = soundata.initialize(self.dataset_name, self.dataset_home)
        self.download_dataset()

    def download_txt(self, url):
        """Download file from a given URL"""
        response = requests.get(url)
        response.raise_for_status()  # Check if the download was successful
        return response.text.splitlines()

    def to_DCASE_format(self, target_subdir="FSD50K_FMA"):
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
            dest_path = os.path.join(self.dataset_home, target_subdir)
            os.makedirs(os.path.join(dest_path, new_dir), exist_ok=True)
            # Copy file to new directory
            shutil.copy(src_file, os.path.join(dest_path, new_dir, filename))

    def cleanup(self, target_subdir="FSD50K_FMA"):
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
            if os.path.isfile(full_path) and not glob.fnmatch.fnmatch(
                file, target_subdir
            ):
                os.remove(full_path)
                print(f"Deleted file: {full_path}")


# Function to prepare the FSD50K dataset
def prepare_fsd50k(args):
    """Prepares the FSD50K dataset based on provided arguments."""
    fsd50k = FSD50KDataSetup(
        dataset_name="fsd50k",
        download=args.download_FSD,
        dataset_home=os.path.join(args.data_dir, SOUND_EVENT_DATASETS_SUBDIR),
    )
    fsd50k.prepare_dataset()
    fsd50k.to_DCASE_format()
    if args.cleanup:
        fsd50k.cleanup()


# Function to prepare the FMA dataset
def prepare_fma(args):
    """Prepares the FMA dataset based on provided arguments."""
    fma = FMADataSetup(
        dataset_home=os.path.join(args.data_dir, SOUND_EVENT_DATASETS_SUBDIR)
    )
    fma.prepare_dataset()


# Main execution logic
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process the data_dir argument.")
    parser.add_argument(
        "--data_dir", type=str, default="datasets", help="Path to the data directory"
    )
    parser.add_argument(
        "--download_FSD",
        action="store_true",
        help="Whether to download the FSD50K dataset",
    )
    parser.add_argument(
        "--download_FMA",
        action="store_true",
        help="Whether to download the FMA dataset",
    )
    parser.add_argument(
        "--cleanup", action="store_true", help="Whether to cleanup after download"
    )
    args = parser.parse_args()
    prepare_fsd50k(args)
    prepare_fma(args)
