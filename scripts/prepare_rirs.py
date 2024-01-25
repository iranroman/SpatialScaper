import os
import argparse
import os
import shutil
import requests
import zipfile
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

from utils import download_file, extract_zip, combine_multizip
from spatialscaper import sofa_utils, tau_utils

METU_URL = "https://zenodo.org/record/2635758/files/spargair.zip"

TAU_REMOTES = {
    "TAU-SRIR_DB.z01": "https://zenodo.org/records/6408611/files/TAU-SRIR_DB.z01?download=1",
    "TAU-SRIR_DB.z02": "https://zenodo.org/records/6408611/files/TAU-SRIR_DB.z02?download=1",
    "TAU-SRIR_DB.z03": "https://zenodo.org/records/6408611/files/TAU-SRIR_DB.z03?download=1",
    "TAU-SRIR_DB.zip": "https://zenodo.org/records/6408611/files/TAU-SRIR_DB.zip?download=1",
    "TAU-SNoise_DB.z01": "https://zenodo.org/records/6408611/files/TAU-SNoise_DB.z01?download=1",
    "TAU-SNoise_DB.zip": "https://zenodo.org/records/6408611/files/TAU-SNoise_DB.zip?download=1",
}

NTAU_ROOMS = 9

TAU_DB_NAME = "TAU-SRIR-DB-SOFA"


def create_single_sofa_file(aud_fmt, tau_db_dir, sofa_db_dir, db_name):
    db_dir = sofa_db_dir
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    for room_idx in range(NTAU_ROOMS):
        # Load flattened (and flipped) rirs/paths from TAU-SRIR database
        rirs, source_pos, mic_pos, room = sofa_utils.load_flat_tau_srir(
            tau_db_dir, room_idx, aud_fmt=aud_fmt
        )

        filepath = os.path.join(db_dir, f"{room}_{aud_fmt}.sofa")
        comment = f"SOFA conversion of {room} from TAU-SRIR-DB"

        print(
            f"Creating .sofa file for {aud_fmt}, Room: {room} (Progress: {room_idx + 1}/9)"
        )

        # Create .sofa files with flattened rirs/paths + metadata
        sofa_utils.create_srir_sofa(
            filepath,
            rirs,
            source_pos,
            mic_pos,
            db_name=db_name,
            room_name=room,
            listener_name=aud_fmt,
            sr=24000,
            comment=comment,
        )


def download_and_extract(url, extract_to):
    # Download the file
    local_filename = url.split("/")[-1]
    download_file(url, extract_to / local_filename)

    # Extract the file
    extract_zip(extract_to / local_filename, extract_to)

    # Remove the zip file
    os.remove(extract_to / local_filename)


def prepare_metu(dataset_path):
    spargpath = Path(dataset_path) / "raw_RIRs" / "spargair" / "em32"
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

    filepath = Path(dataset_path) / "sofa_RIRs" / "metu_sparg_em32.sofa"
    rirs = np.array(IRs)
    source_pos = np.array(xyzs)
    mic_pos = np.array([[0, 0, 0]])

    sofa_utils.create_srir_sofa(
        filepath,
        rirs,
        source_pos,
        mic_pos,
        db_name="METU-SPARG",
        room_name="classroom",
        listener_name="em32",
        sr=sr,
    )


def download_tau(dest_path):
    # Download combine and extract zip files
    for filename, url in TAU_REMOTES.items():
        download_file(url, dest_path / filename)
    combine_multizip(f"{dest_path/'TAU-SRIR_DB.zip'}", f"{dest_path/'single.zip'}")
    extract_zip(dest_path / "single.zip", dest_path)
    combine_multizip(f"{dest_path/'TAU-SNoise_DB.zip'}", f"{dest_path/'single.zip'}")
    extract_zip(dest_path / "single.zip", dest_path)


def prepare_tau(path_raw, path_sofa, formats=["foa", "mic"]):
    # generate Sofa files
    tau_db_dir = f"{path_raw/'TAU-SRIR_DB'}"
    sofa_db_dir = path_sofa
    for aud_fmt in formats:
        print(f"Starting .sofa creation for {aud_fmt} format.")
        create_single_sofa_file(aud_fmt, tau_db_dir, sofa_db_dir, TAU_DB_NAME)
        print(f"Finished .sofa creation for {aud_fmt} format.")


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

    os.makedirs(Path(args.path) / "raw_RIRs", exist_ok=True)
    os.makedirs(Path(args.path) / "sofa_RIRs", exist_ok=True)

    # METU
    download_and_extract(METU_URL, Path(args.path) / "raw_RIRs")
    prepare_metu(Path(args.path))

    # TAU
    dest_path = Path(args.path) / "raw_RIRs"
    download_tau(dest_path)
    dest_path_sofa = Path(args.path) / "sofa_RIRs"
    prepare_tau(dest_path, dest_path_sofa)
