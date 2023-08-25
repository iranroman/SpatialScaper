import librosa
import numpy as np
import os
import math
import pickle
from pysofaconventions import *

from room_scaper import sofa_utils

'''
Important:
    This approach makes a single trayectory from the different DOAs on mics 1-5. It sorts the RIRs based on DOAs azimuth in 
    decreasing order. The absorption level are considered part of the same trayectory, we loop back and forth across the 6 levels.
'''

FS = 48000 # original impulse reponse sampling rate
NEW_FS = 24000 # new sampling rate (same as DCASE Synth)

def get_absorption_level(filename):
    return int(filename.split("_")[4].replace("percent", ""))

def center_and_translate(receiver_pos, source_pos):
    # Given two points, center the receiver coordinate at zero and tranlate the source
    x1, y1, z1 = receiver_pos[0], receiver_pos[1], receiver_pos[2]
    x2, y2, z2 = source_pos[0], source_pos[1], source_pos[2]
    # compute translation of the source (loud speaker)
    translation_x = -x1
    translation_y = -y1
    # apply tranlation, note that the receiver (mic) remains at the same height
    receiver_centered = [0, 0, z1]
    source_translated = [x2 + translation_x, y2 + translation_y, z2]
    return receiver_centered, source_translated

def compute_azimuth_elevation(receiver_pos, source_pos):
    # Calculate the vector from the receiver to the source
    vector = [source_pos[0] - receiver_pos[0], source_pos[1] - receiver_pos[1], source_pos[2] - receiver_pos[2]]
    # Calculate the azimuth angle
    azimuth = math.atan2(vector[0], vector[1])
    # if azimuth < 0:
    #     azimuth += math.pi
    # Calculate the elevation angle
    distance = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    elevation = math.asin(vector[2] / distance)
    return azimuth, elevation, distance

def create_single_sofa_file(aud_fmt, arni_db_dir, sofa_db_dir, room="ARNI"):
    db_dir = os.path.join(sofa_db_dir, aud_fmt)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    sofa_files_absorption = [file for file in os.listdir(arni_db_dir) if file.endswith(".sofa")]

    assert len(sofa_files_absorption) != 0, f"Error: {arni_db_dir} contains no .sofa files"
    comment = f"SOFA conversion of {room} translated into a single trayectory"
    # Sort the sofa_files based on increasing absorption levels
    sorted_sofa_files = sorted(sofa_files_absorption, key=get_absorption_level)

    filepath = os.path.join(db_dir, f'{room}.sofa')
    source_pos, mic_pos, rirs = [], [], [] 
    for abs_idx, sofa_abs_file in enumerate(sorted_sofa_files):
        # Load flattened (and flipped) rirs/paths from TAU-SRIR database
        sofa = SOFAFile(os.path.join(arni_db_dir, sofa_abs_file),'r')
        print(f"Creating .sofa file for {aud_fmt}, Room: {room} (Progress: {abs_idx + 1}/{len(sofa_files_absorption)})")
        if not sofa.isValid():
            print("Error: the file is invalid")
            break

        sourcePositions = sofa.getVariableValue('SourcePosition') # get sound source position
        listenerPosition = sofa.getVariableValue('ListenerPosition') # get mic position
        # get RIR data
        rirdata = sofa.getDataIR()
        num_meas, num_ch = rirdata.shape[0], rirdata.shape[1]
        num_meas = 15 # take only mics 1, 2, 3, 4, 5, exclude 6, 7
        angles_mic_src = [math.degrees(compute_azimuth_elevation(lis, src)[0]) \
                            for lis, src in zip(listenerPosition[:num_meas], sourcePositions[:num_meas])]
        # sort rir measurements in increasing or decreasing order since we move back and forth
        meas_sorted_ord = np.argsort(angles_mic_src)[::-1] if (abs_idx % 2) == 0 else np.argsort(angles_mic_src)
        sorted_angles_mic_src = [angles_mic_src[i] for i in meas_sorted_ord]
        rir, mic_loc, src_loc = [], [], []
        for meas in meas_sorted_ord: # for each meas in decreasing order
            # add impulse response
            irdata = rirdata[meas, :, :]
            irdata_resamp = librosa.resample(irdata, orig_sr=FS, target_sr=NEW_FS)
            rir.append(irdata_resamp[[5,9,25,21], :]) # add em32 rir data w/ hard-coded chans for tetra mic
            cent_receiv, trans_source = center_and_translate(listenerPosition[meas], sourcePositions[meas])
            mic_loc.append(cent_receiv) # add mic coordinate position (centered at zero)
            src_loc.append(trans_source) # add source (loud speaker) position (translated w.r.t microphone centered at zero)
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
        sr=24000,
        comment=comment
    )


arni_db_dir = '/home/iran/datasets/6dof_SRIRs_eigenmike_raw'
sofa_db_dir = 'TAU_DB/TAU_SRIR_DB_SOFA'
aud_fmt='mic'
print(f"Starting .sofa creation for the ARNI dataset.")
create_single_sofa_file(aud_fmt, arni_db_dir, sofa_db_dir)
print(f"Finished .sofa creation for {aud_fmt} format, per traj.")
