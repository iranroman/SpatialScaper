import argparse
import os
import shutil
import requests
import zipfile
import librosa
import numpy as np
import soundfile as sf
from spatialscaper.sofa_utils import create_srir_sofa
from pathlib import Path

from .utils import download_file, extract_zip

METU_URL = "https://zenodo.org/record/2635758/files/spargair.zip"


def download_and_extract(url, extract_to):
    # Download the file
    local_filename = url.split("/")[-1]
    download_file(url, extract_to)

    # Extract the file
    extract_zip(local_filename, extract_to)

    # Remove the zip file
    os.remove(local_filename)


def prepare_metu(dataset_path):
    spargpath = Path(dataset_path) / "spargair" / "em32"
    nEMchans = 32
    XYZs = os.listdir(spargpath)

    def XYZ_2_xyz(XYZ):
        X, Y, Z = XYZ
        x = (3 - int(X)) * 0.5
        y = (3 - int(Y)) * 0.5
        z = (int(Z) - 2) * 0.5
        return x, y, z

    IRs = []
    xyzs = []
    for XYZ in XYZs:
        xyz = XYZ_2_xyz(XYZ)
        xyzs.append(xyz)

        wavpath = spargpath / XYZ
        X = []
        for i in range(nEMchans):
            wavfile = wavpath / f"IR{i+1:05d}.wav"
            x, sr = sf.read(wavfile)
            X.append(x)
        IRs.append(np.array(X))

    filepath = Path(dataset_path) / "spargair" / "metu_sparg.sofa"
    rirs = np.array(IRs)
    source_pos = np.array(xyzs)
    mic_pos = np.array([[0, 0, 0]])

    create_srir_sofa(
        filepath,
        rirs,
        source_pos,
        mic_pos,
        db_name="METU-SPARG",
        room_name="classroom",
        listener_name="em32",
        sr=sr,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and prepare METU SPARG dataset."
    )
    parser.add_argument(
        "--path",
        default="datasets/rir_datasets",
        help="Path to store and process the dataset.",
    )
    args = parser.parse_args()

    download_and_extract(METU_URL, args.path)
    prepare_metu(args.path)
