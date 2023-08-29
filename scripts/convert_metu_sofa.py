import os
import argparse
import librosa
import numpy as np

from room_scaper import sofa_utils

# Reference METU outter trayectory:  bottom outter trayectory
REF_OUT_TRAJ = ["034", "024", "014", "004", "104", "204",
            "304", "404", "504", "604", "614", "624",
            "634", "644", "654", "664", "564", "464",
            "364", "264", "164", "064", "054", "044"]
# Reference METU inner trayectory:  bottom inner trayectory
REF_IN_TRAJ = ["134", "124", "114", "214","314", "414", "514", "524",
                "534", "544", "554", "454", "354", "254", "154", "145"]

def get_mic_xyz():
    """
    Get em32 microphone coordinates in 3D space
    """
    return [(3 - 3) * 0.5, (3 - 3) * 0.5, (2 - 2) * 0.3 + 1.5]

def create_single_sofa_file(aud_fmt, metu_db_dir, sofa_db_dir, room="METU"):
    db_dir = os.path.join(sofa_db_dir, aud_fmt)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    comment = f"SOFA conversion of {room} translated into a single trayectory"
    filepath = os.path.join(db_dir, f'{room}.sofa')

    if aud_fmt != "em32" and aud_fmt != "mic":
        parser.error("You must provide a valid microphone name: em32, mic")

    top_height = 5
    mic_xyz = get_mic_xyz()
    # Data structure (potentially to be used once this is fully integrated)
    room = "metu"

    source_pos, mic_pos, rirs = [], [], [] 

    # Outter trayectory: bottom to top
    for height in range(0, top_height):
        for num in REF_OUT_TRAJ:
            # Coords computed based on documentation.pdf from METU Sparg
            x = (3 - int(num[0])) * 0.5
            y = (3 - int(num[1])) * 0.5
            z = (2 - (int(num[2])-height)) * 0.3 + 1.5
            source_xyz = [x, y, -1*z] # note -1 since METU is flipped up-side-down
            
            source_pos.append(source_xyz)
            mic_pos.append(mic_xyz) 
            rir_name = num[0] + num[1] + str(int(num[2])-height)
            ir_path = os.path.join(metu_db_dir, rir_name, f"IR_{aud_fmt}.wav")
            irdata, sr = librosa.load(ir_path, mono=False, sr=48000)
            irdata_resamp = librosa.resample(irdata, orig_sr=sr, target_sr=24000)
            rirs.append(irdata_resamp)
    
    # Inner trayectory: top to bottom
    for height in range(top_height-1, -1, -1):
        for num in REF_IN_TRAJ:
            # Coords computed based on documentation.pdf from METU Sparg
            x = (3 - int(num[0])) * 0.5
            y = (3 - int(num[1])) * 0.5
            z = (2 - (height)) * 0.3 + 1.5
            source_xyz = [x, y, -1*z] # note -1 since METU is flipped up-side-down
            
            source_pos.append(source_xyz)
            mic_pos.append(mic_xyz) 
            rir_name = num[0] + num[1] + str(height)
            ir_path = os.path.join(metu_db_dir, rir_name, f"IR_{aud_fmt}.wav")
            irdata, sr = librosa.load(ir_path, mono=False, sr=48000)
            irdata_resamp = librosa.resample(irdata, orig_sr=sr, target_sr=24000)
            rirs.append(irdata_resamp)
    
    rirs = np.array(rirs)
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

metu_db_dir = "/home/iran/datasets/spargair/em32"
sofa_db_dir = 'TAU_DB/METU_SRIR_DB_SOFA'
aud_fmt='mic'
print(f"Starting .sofa creation for the METU dataset.")
create_single_sofa_file(aud_fmt, metu_db_dir, sofa_db_dir, room="METU")
