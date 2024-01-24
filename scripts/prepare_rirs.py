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

from utils import download_file, extract_zip

METU_URL = "https://zenodo.org/record/2635758/files/spargair.zip"

TAU_REMOTES = {
    'TAU-SRIR_DB.z01':'https://zenodo.org/records/6408611/files/TAU-SRIR_DB.z01?download=1',
    'TAU-SRIR_DB.z02':'https://zenodo.org/records/6408611/files/TAU-SRIR_DB.z02?download=1',
    'TAU-SRIR_DB.z03':'https://zenodo.org/records/6408611/files/TAU-SRIR_DB.z03?download=1',
    'TAU-SRIR_DB.zip':'https://zenodo.org/records/6408611/files/TAU-SRIR_DB.zip?download=1',
    'TAU-SNoise_DB.z01':'https://zenodo.org/records/6408611/files/TAU-SNoise_DB.z01?download=1',
    'TAU-SNoise_DB.zip':'https://zenodo.org/records/6408611/files/TAU-SNoise_DB.zip?download=1',
    }


def download_and_extract(url, extract_to):
    # Download the file
    local_filename = url.split("/")[-1]
    download_file(url, extract_to/local_filename)

    # Extract the file
    extract_zip(extract_to/local_filename,extract_to)

    # Remove the zip file
    os.remove(extract_to/local_filename)


def prepare_metu(dataset_path):
    spargpath = Path(dataset_path) / "raw_RIRs"/"spargair" / "em32"
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

    filepath = Path(dataset_path) / "sofa_RIRs" / "metu_sparg.sofa"
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

    os.makedirs(Path(args.path)/'raw_RIRs', exist_ok=True)
    os.makedirs(Path(args.path)/'sofa_RIRs', exist_ok=True)

    # METU
    download_and_extract(METU_URL, Path(args.path)/'raw_RIRs')
    prepare_metu(Path(args.path))

    # TAU
    dest_path = Path(args.path)/'raw_RIRs'
    for filename,url in TAU_REMOTES.items():
        download_file(url, dest_path/filename)
    subprocess.run(["zip", "-s", "0", dest_path/'TAU-SRIR_DB.zip', "--out", dest_path/'single.zip'], shell=True)
    extract_zip(dest_path/'single.zip',dest_path)
    subprocess.run(["zip", "-s", "0", dext_path/'TAU-SNoise_DB.zip', "--out", dest_path/'single.zip'], shell=True)
    extract_zip(dest_path/'single.zip',dest_path)
