import os
import numpy as np
import soundfile as sf
from room_scaper import sofa_utils

PATH_TO_ROOMS = '/home/iran/SELD-data-generator/TAU_DB/TAU_SRIR_DB_SOFA'
FORMAT = 'mic'
ROOMS = [
    #'tc352',
    #'sc203',
    #'bomb_shelter',
    #'pc226',
    #'pb132',
    #'se203',
    #'tb103',
    #'sa203',
    #'gym',
    #'arni',
    'metu',
]
OUTPATH = 'scripts/wav_rirs'
FS = 24000

for room in ROOMS:

    path_to_sofas = os.path.join(PATH_TO_ROOMS, FORMAT, room)
    sofas = [f for f in os.listdir(path_to_sofas) if 'sofa' in f]

    for sofa in sofas:

        micdata = sofa_utils.load_rir(os.path.join(path_to_sofas, sofa))

        output_directory = os.path.join(OUTPATH, room, sofa[:-5]) 
        # Create the output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)

        # Iterate through the recordings
        for recording_idx in range(micdata.shape[0]):
            recording = micdata[recording_idx]
            
            # Create the output filename
            wav_filename = os.path.join(output_directory, f"rir_{recording_idx}.wav")
            
            # Save the multi-channel WAV file
            sf.write(wav_filename, recording.T, FS)
            
            print(f"Recording {recording_idx} saved as: {wav_filename}")

