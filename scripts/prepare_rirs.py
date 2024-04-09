import os
import math
import random
import argparse
import shutil
import requests
import zipfile
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import pysofaconventions as pysofa

from utils import download_file, extract_zip, combine_multizip
from spatialscaper import sofa_utils, tau_utils

FS = 24000

METU_URL = "https://zenodo.org/record/2635758/files/spargair.zip"

TAU_REMOTES = {
    "TAU-SRIR_DB.z01": "https://zenodo.org/records/6408611/files/TAU-SRIR_DB.z01?download=1",
    "TAU-SRIR_DB.z02": "https://zenodo.org/records/6408611/files/TAU-SRIR_DB.z02?download=1",
    "TAU-SRIR_DB.z03": "https://zenodo.org/records/6408611/files/TAU-SRIR_DB.z03?download=1",
    "TAU-SRIR_DB.zip": "https://zenodo.org/records/6408611/files/TAU-SRIR_DB.zip?download=1",
    "TAU-SNoise_DB.z01": "https://zenodo.org/records/6408611/files/TAU-SNoise_DB.z01?download=1",
    "TAU-SNoise_DB.zip": "https://zenodo.org/records/6408611/files/TAU-SNoise_DB.zip?download=1",
}

ARNI_URL = "https://zenodo.org/records/5720724/files/6dof_SRIRs_eigenmike_raw.zip"

NTAU_ROOMS = 9

TAU_DB_NAME = "TAU-SRIR-DB-SOFA"
ARNI_DB_NAME = "ARNI-SRIR-DB-SOFA"


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
            sr=FS,
            comment=comment,
        )


def download_and_extract(url, extract_to):
    # Ensure the extract_to directory exists
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    # Extract the filename from the URL
    local_filename = url.split("/")[-1]
    zip_path = extract_to / local_filename
    # Check if the extracted directory already exists
    extracted_dir = extract_to / local_filename.replace(".zip", "")
    if extracted_dir.is_dir() or zip_path.is_file():
        print(
            f"Data already present in {extracted_dir}. Skipping download and extraction."
        )
    else:
        # Download and extract the file
        download_file(url, zip_path)
        extract_zip(zip_path, extract_to)

        # remove the zip file after extraction
        os.remove(zip_path)


def prepare_metu(dataset_path):
    spargpath = Path(dataset_path) / "source_data" / "spargair" / "em32"
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

    filepath = Path(dataset_path) / "spatialscaper_RIRs" / "metu_sparg_em32.sofa"
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


def compute_azimuth_elevation(receiver_pos, source_pos):
    # Calculate the vector from the receiver to the source
    vector = [
        source_pos[0] - receiver_pos[0],
        source_pos[1] - receiver_pos[1],
        source_pos[2] - receiver_pos[2],
    ]
    # Calculate the azimuth angle
    azimuth = math.atan2(vector[0], vector[1])
    # Calculate the elevation angle
    distance = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    elevation = math.asin(vector[2] / distance)
    return azimuth, elevation, distance


def get_absorption_level_arni(filename):
    return int(filename.split("_")[4].replace("percent", ""))


def center_and_translate_arni(receiver_pos, source_pos):
    # Given two points, center the receiver coordinate at zero and tranlate the source
    y1, x1, z1 = receiver_pos[0], receiver_pos[1], receiver_pos[2]
    y2, x2, z2 = source_pos[0], source_pos[1], source_pos[2]
    # compute translation of the source (loud speaker)
    # add small perturbation to have unique coordinate for trajectory generation purposes
    translation_y = -y1 + random.uniform(-0.0001, 0.0001)
    translation_x = -x1 + random.uniform(-0.0001, 0.0001)
    translation_z = z1 + random.uniform(-0.0001, 0.0001)
    # apply tranlation, note that the receiver (mic) remains at the same height
    receiver_centered = [0, 0, 0]
    source_translated = [x2 + translation_x, y2 + translation_y, translation_z - z2]
    return receiver_centered, source_translated


def create_single_sofa_file_arni(aud_fmt, arni_db_dir, sofa_db_dir, room="ARNI"):
    db_dir = sofa_db_dir
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    sofa_files_absorption = [
        file for file in os.listdir(arni_db_dir) if file.endswith(".sofa")
    ]

    assert (
        len(sofa_files_absorption) != 0
    ), f"Error: {arni_db_dir} contains no .sofa files"
    comment = f"SOFA conversion of {room} translated into a single trayectory"
    # Sort the sofa_files based on increasing absorption levels
    sorted_sofa_files = sorted(sofa_files_absorption, key=get_absorption_level_arni)

    filepath = os.path.join(db_dir, f"arni_mic.sofa")
    source_pos, mic_pos, rirs = [], [], []
    for abs_idx, sofa_abs_file in enumerate(sorted_sofa_files):
        # Load flattened (and flipped) rirs/paths from TAU-SRIR database
        sofa = pysofa.SOFAFile(os.path.join(arni_db_dir, sofa_abs_file), "r")
        print(
            f"Creating .sofa file for {aud_fmt}, Room: {room} (Progress: {abs_idx + 1}/{len(sofa_files_absorption)})"
        )
        if not sofa.isValid():
            print("Error: the file is invalid")
            break

        sourcePositions = sofa.getVariableValue(
            "SourcePosition"
        )  # get sound source position
        listenerPosition = sofa.getVariableValue("ListenerPosition")  # get mic position
        # get RIR data
        rirdata = sofa.getDataIR()
        num_meas, num_ch = rirdata.shape[0], rirdata.shape[1]
        num_meas = 15  # take only mics 1, 2, 3, 4, 5, exclude 6, 7
        angles_mic_src = [
            math.degrees(compute_azimuth_elevation(lis, src)[0])
            for lis, src in zip(listenerPosition[:num_meas], sourcePositions[:num_meas])
        ]
        # sort rir measurements in increasing or decreasing order since we move back and forth
        meas_sorted_ord = (
            np.argsort(angles_mic_src)[::-1]
            if (abs_idx % 2) == 0
            else np.argsort(angles_mic_src)
        )
        sorted_angles_mic_src = [angles_mic_src[i] for i in meas_sorted_ord]
        rir, mic_loc, src_loc = [], [], []
        for meas in meas_sorted_ord:  # for each meas in decreasing order
            # add impulse response
            irdata = rirdata[meas, :, :]
            irdata_resamp = librosa.resample(irdata, orig_sr=48000, target_sr=FS)
            rir.append(
                irdata_resamp[[5, 9, 25, 21], :]
            )  # add em32 rir data w/ hard-coded chans for tetra mic
            cent_receiv, trans_source = center_and_translate_arni(
                listenerPosition[meas], sourcePositions[meas]
            )
            mic_loc.append(
                cent_receiv
            )  # add mic coordinate position (centered at zero)
            src_loc.append(
                trans_source
            )  # add source (loud speaker) position (translated w.r.t microphone centered at zero)
        rirs.extend(rir)
        mic_pos.extend(mic_loc)
        source_pos.extend(src_loc)

    rirs = np.array(rirs)
    mic_pos = np.array(mic_pos)
    source_pos = np.array(source_pos)

    # Create .sofa files with flattened rirs/paths + metadata
    sofa_utils.create_srir_sofa(
        filepath,
        rirs,
        source_pos,
        mic_pos,
        room_name=room,
        listener_name=aud_fmt,
        sr=FS,
        comment=comment,
    )


def prepare_arni(path_raw, path_sofa, formats=["mic"]):
    # generate Sofa files
    arni_db_dir = f"{path_raw/'6dof_SRIRs_eigenmike_raw'}"
    sofa_db_dir = path_sofa
    for aud_fmt in formats:
        print(f"Starting .sofa creation for {aud_fmt} format.")
        create_single_sofa_file_arni(aud_fmt, arni_db_dir, sofa_db_dir, ARNI_DB_NAME)
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

    os.makedirs(Path(args.path) / "source_data", exist_ok=True)
    os.makedirs(Path(args.path) / "spatialscaper_RIRs", exist_ok=True)

    ## METU
    #download_and_extract(METU_URL, Path(args.path) / "source_data")
    #prepare_metu(Path(args.path))

    ## TAU
    #dest_path = Path(args.path) / "source_data"
    #download_tau(dest_path)
    #dest_path_sofa = Path(args.path) / "spatialscaper_RIRs"
    #prepare_tau(dest_path, dest_path_sofa)

    # ARNI
    DEBUG_FLAG = True # if True: right of y axis is negative, else right of y axis is positive
    dest_path = Path(args.path) / "source_data"
    download_and_extract(ARNI_URL, Path(args.path) / "source_data")
    dest_path_sofa = Path(args.path) / "spatialscaper_RIRs"
    prepare_arni(dest_path, dest_path_sofa)
