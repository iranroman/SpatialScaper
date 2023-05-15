import os
import argparse
import pickle
import librosa
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from sparg_utils import *

METU_PATH = "/Users/adrianromanguzman/Documents/repos/SELD-data-generator/spargair/em32"

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('microphone', type=str, help='Name of the microphone type to be processed: em32, mic')
    args = parser.parse_args()
    microphone = args.microphone
    if microphone != "em32" and microphone != "mic":
        parser.error("You must provide a valid microphone name: em32, mic")
    # Hard coded bottom trayectory
    outter_trayectory_bottom = ["034", "024", "014", "004", "104", "204",
                                "304", "404", "504", "604", "614", "624",
                                "634", "644", "654", "664", "564", "464",
                                "364", "264", "164", "064", "054", "044"]
    top_height = 5
    mic_xyz = get_mic_xyz()
    rirdata_dict = {}
    # Data structure (potentially to be used once this is fully integrated)
    room = "metu"
    rirdata_dict[room] = {}
    rirdata_dict[room]['doa_xyz'] = [[]]
    rirdata_dict[room]['dist'] = [[]]
    rirdata_dict[room]['rir'] = [[]]

    for height in range(0, top_height): # loop through heights
        heights_list = []   # list of heights
        doa_xyz = []        # list of unit vectors for each doa per height
        hir_list = []       # impulse responses list per height
        for num in outter_trayectory_bottom:
            # Coords computed based on documentation.pdf from METU Sparg
            x = (3 - int(num[0])) * 0.5
            y = (3 - int(num[1])) * 0.5
            z = (2 - int(num[2])+height) * 0.3 + 1.5
            source_xyz = [x, y, z]
            azi, ele, dist = az_ele_from_source(mic_xyz, source_xyz)
            uvec_xyz = unit_vector(azi, ele)
            doa_xyz.append(uvec_xyz)
            heights_list.append(dist) 
            rir_name = num[0] + num[1] + str(int(num[2])-height)
            print("RIR file: ", rir_name)
            ir_path = os.path.join(METU_PATH, rir_name, f"IR_{microphone}.wav")
            irdata, sr = librosa.load(ir_path, mono=False, sr=48000)
            irdata_resamp = librosa.resample(irdata, orig_sr=sr, target_sr=24000)
            hir_list.append(irdata_resamp)
        rirdata_dict[room]['doa_xyz'][0].append(doa_xyz)
        rirdata_dict[room]['dist'][0].append(heights_list)
        rirdata_dict[room]['rir'][0].append(hir_list)

    with open('{}.pkl'.format(f"rir_{microphone}"), 'wb') as outp:
        pickle.dump(rirdata_dict[room]['rir'], outp, pickle.HIGHEST_PROTOCOL)

    with open('{}.pkl'.format(f"doa_xyz_{microphone}"), 'wb') as outp:
        pickle.dump(rirdata_dict[room]['doa_xyz'], outp, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
