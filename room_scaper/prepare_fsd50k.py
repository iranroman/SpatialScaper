import os
import yaml
import json
import soundata
import numpy as np
import pandas as pd
from room_scaper.prepare_utils import *

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class BaseDataLoad:
    def __init__(self, dataset_home=None, metadata_path=None):
        self.dataset_home = dataset_home
        self.metadata_path = metadata_path


FMA_REMOTES = {
    "name": "fma_small",
    "filename": "fma_small.zip",
    "base_url": "https://os.unil.cloud.switch.ch/fma/fma_small.zip",
    "metadata_url": "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip",
}


class FMADataLoad(BaseDataLoad):
    def __init__(self, ntracks_genre=10, split_prob=0.6, **kwargs):
        super().__init__(**kwargs)
        self.ntracks_genre = ntracks_genre
        self.split_prob = split_prob
        self.dataset_name = FMA_REMOTES["name"]
        self.base_url = FMA_REMOTES["base_url"]
        self.metadata_url = FMA_REMOTES["metadata_url"]
        self.zip_name = FMA_REMOTES["filename"]
        self.fma_to_dcase_dict = {}

    def load_dataset(self):
        # Check if fma_small exists, else create it
        if not check_dataset_exists(os.path.join(self.dataset_home, self.dataset_name)):
            print("Downloading FMA small dataset...")
            self.download_dataset()
        else:
            print("FMA small dataset already exists. Skipping download.")
        self.gen_dataset_splits()

    def save_fma_to_dcase_json(self):
        with open(
            os.path.join(self.metadata_path, "fma_to_dcase.json"), "w"
        ) as json_file:
            json.dump(self.fma_to_dcase_dict, json_file, indent=4)

    def download_dataset(self):
        os.makedirs(self.dataset_home, exist_ok=True)
        os.makedirs(self.metadata_path, exist_ok=True)
        # download fma_small zip
        download_file(self.base_url, os.path.join(self.dataset_home, self.zip_name))
        extract_zip(os.path.join(self.dataset_home, self.zip_name), self.dataset_home)
        print("Done unzipping")
        # download fma metadata
        print("Downloading fma_small metadata...")
        download_file(
            self.metadata_url, os.path.join(self.metadata_path, self.zip_name)
        )
        extract_zip(os.path.join(self.metadata_path, self.zip_name), self.metadata_path)
        print("Done unzipping")

    def gen_dataset_splits(self):
        tracks = pd.read_csv(
            os.path.join(self.metadata_path, "fma_metadata/tracks.csv"),
            header=[0, 1],
            index_col=0,
        )
        genres = tracks["track"]["genre_top"].unique()
        # Loop through the genre 8 classes
        for genre in genres:
            # Get tracks by genre, consider only set from "small" tracks
            genre_tracks = tracks[
                (tracks["track", "genre_top"] == genre)
                & (tracks["set", "subset"] == "small")
            ]
            # Get ntracks_genre from the current genre
            train_tracks = genre_tracks[: self.ntracks_genre]
            for track_id, track in train_tracks.iterrows():
                # get track name by id
                subdir = f"{track_id:06}"[:3]
                fma_track_path = os.path.join(
                    self.dataset_home, self.dataset_name, subdir, f"{track_id:06}.mp3"
                )
                # Based on prob decide test vs. train split
                save_dir = None
                # Define DCASE path format variables
                super_class = "music"
                fold = "train" if np.random.rand() < self.split_prob else "test"
                class_name = str(genre)
                dcase_path = os.path.join(
                    super_class, fold, str(genre), f"{track_id:06}.mp3"
                )
                # generate dcase to fma map
                if fold not in self.fma_to_dcase_dict:
                    self.fma_to_dcase_dict[fold] = {}
                self.fma_to_dcase_dict[fold][dcase_path] = fma_track_path


class FSD50KDataLoad(BaseDataLoad):
    def __init__(
        self,
        dataset_name="fsd50k",
        download=False,
        music_home=None,
        music_metadata=None,
        dcase_sound_events_txt=None,
        ntracks_genre=10,
        split_prob=0.6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_name = dataset_name
        self.download = download
        self.music_home = music_home
        self.music_metadata = music_metadata
        self.dcase_sound_events_txt = dcase_sound_events_txt
        self.url_fsd_selected_txt = (
            "https://zenodo.org/record/6406873/files/FSD50K_selected.txt"
        )
        self.fsd_to_dcase_dict = {}  # Dict containing FSD50K to DCASE file mappings

        # Remove 'dataset_home' from kwargs before passing to FMADataLoad
        if "dataset_home" in kwargs:
            del kwargs["dataset_home"]
            del kwargs["metadata_path"]

        self.music = FMADataLoad(
            dataset_home=self.music_home,
            metadata_path=self.music_metadata,
            ntracks_genre=ntracks_genre,
            split_prob=split_prob,
        )

    def download_dataset(self):
        self.fsd50k.download()

    def load_dataset(self):
        assert (
            self.dataset_home is not None or not self.download
        ), "Dataset home path must be provided when download is enabled."
        self.fsd50k = soundata.initialize(self.dataset_name, self.dataset_home)
        if self.download:
            self.download_dataset()  # download fsd50k
        self.music.load_dataset()  # load fma
        self.music.gen_dataset_splits()
        self.music.save_fma_to_dcase_json()
        self.gen_fsd_to_dcase_dict()

    def save_fsd_to_dcase_json(self):
        with open(
            os.path.join(self.metadata_path, "fsd_to_dcase.json"), "w"
        ) as json_file:
            json.dump(self.fsd_to_dcase_dict, json_file, indent=4)

    def get_filenames(self, fold="train"):
        if fold not in self.fsd_to_dcase_dict.keys():
            raise ValueError(
                "Invalid fold argument. The fold must be 'train' or 'test'."
            )
        return self.fsd_to_dcase_dict[fold]

    def gen_fsd_to_dcase_dict(self):
        os.makedirs(self.dataset_home, exist_ok=True)
        os.makedirs(self.metadata_path, exist_ok=True)

        if not os.path.exists(self.dcase_sound_events_txt):
            download_file(self.url_fsd_selected_txt, self.dcase_sound_events_txt)

        with open(self.dcase_sound_events_txt, "r") as file:
            for line in file:
                dcase_path = line.strip()
                parts = line.split("/")
                track_id = os.path.basename(dcase_path)
                super_class = parts[0]
                fold = parts[1]
                class_name = parts[2]
                dcase_path = f"{super_class}/{fold}/{class_name}/{track_id}"
                fsd50k_fold = (
                    "FSD50K.dev_audio" if fold == "train" else "FSD50K.eval_audio"
                )
                if fold not in self.fsd_to_dcase_dict:
                    self.fsd_to_dcase_dict[fold] = {}
                self.fsd_to_dcase_dict[fold][dcase_path] = os.path.join(
                    self.dataset_home, f"{fsd50k_fold}/{track_id}"
                )
        self.fsd_to_dcase_dict["train"].update(self.music.fma_to_dcase_dict["train"])
        self.fsd_to_dcase_dict["test"].update(self.music.fma_to_dcase_dict["test"])


PARAM_CONFIG = {
    "dataset_home": "/datasets/FSD50K",  # add /path/to (not path/to/dir)
    "metadata_path": os.path.join(
        PARENT_DIR, "dcase_datagen/metadata"
    ),  # add /path/to (not path/to/dir)
    "dcase_sound_events_txt": os.path.join(
        PARENT_DIR, "dcase_datagen/metadata/sound_event_fsd50k_filenames.txt"
    ),
    "download": False,
    "music_home": "/datasets/fma",  # add /path/to (not path/to/dir)
    "music_metadata": os.path.join(
        PARENT_DIR, "dcase_datagen/metadata"
    ),  # add /path/to (not path/to/dir)
    "ntracks_genre": 10,
    "split_prob": 0.6,
}


def prepare_fsd50k(config=PARAM_CONFIG):
    fsd50k = FSD50KDataLoad(**config)
    fsd50k.load_dataset()
    print("Saved JSON FSD50K to DCASE data format")
    fsd50k.save_fsd_to_dcase_json()
    return fsd50k


if __name__ == "__main__":
    prepare_fsd50k()
