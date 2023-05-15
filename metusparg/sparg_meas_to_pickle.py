import os
import pickle
import librosa
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from sparg_utils import *

METU_PATH = "/Users/adrianromanguzman/Documents/repos/SELD-data-generator/spargair/em32"
mic_xyz = [(3 - 3) * 0.5, (3 - 3) * 0.5, (2 - 2) * 0.3 + 1.5]
mic_nch = 32
top_height = 5

outter_trayectory_bottom = ["034", "024", "014", "004", "104", "204", "304", "404", "504", "604", "614", "624", "634", "644", "654", "664", "564", "464", "364", "264", "164", "064", "054", "044"]

x_coords = []
y_coords = []
z_coords = []

rirdata_dict = {}
room = "metu"
rirdata_dict[room] = {}
rirdata_dict[room]['doa_xyz'] = [[]]
rirdata_dict[room]['dist'] = [[]]
rirdata_dict[room]['rir'] = [[]]

for height in range(0, top_height): # loop through heights
    heights_list = []
    doa_xyz = []
    hir_list = [] # impulse responses list per height
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
        '''
        # Note: uncomment code below to merge IRs into a multi-channel .wav
        # Load the 32 eigenmike IR wavefiles and concat in a single numpy array
        cmd_mix_all_ch = ""
        for mic_idx in [6,10,26,22]: #range(1,mic_nch+1):
            cmd_mix_all_ch += f' {os.path.join(METU_PATH, rir_name)}/IR{mic_idx:05}.wav '
        # SoX -M to merge 32 into multi-channel signal
        ir32ch_path = os.path.join(METU_PATH, rir_name, "IR_MIC.wav")
        print("generating", ir32ch_path)
        os.system(f'sudo sox -M {cmd_mix_all_ch} {ir32ch_path}')
        '''
        print("RIR file: ", rir_name)
        ir32ch_path = os.path.join(METU_PATH, rir_name, "IR_MIC.wav")
        # ir4ch_path = os.path.join(METU_PATH, rir_name, "IR_MIC.wav")
        irdata, sr = librosa.load(ir32ch_path, mono=False, sr=48000)
        irdata_resamp = librosa.resample(irdata, orig_sr=sr, target_sr=24000)
        hir_list.append(irdata_resamp)
        # # Used for plotting only
        # x_coords.append(x)
        # y_coords.append(y)
        # z_coords.append(z)
    rirdata_dict[room]['doa_xyz'][0].append(doa_xyz)
    rirdata_dict[room]['dist'][0].append(heights_list)
    rirdata_dict[room]['rir'][0].append(hir_list)

with open('{}.pkl'.format("rir_MIC"), 'wb') as outp:
    pickle.dump(rirdata_dict[room]['rir'], outp, pickle.HIGHEST_PROTOCOL)

with open('{}.pkl'.format("doa_xyz_MIC"), 'wb') as outp:
    pickle.dump(rirdata_dict[room]['doa_xyz'], outp, pickle.HIGHEST_PROTOCOL)


# # Plotting measurements
# initpoint_xyz = [(3 - int(0)) * 0.5, (3 - int(3)) * 0.5, (2 - 2) * 0.3 + 1.5]
# az, el, _ = az_ele_from_source(mic_xyz, initpoint_xyz)
# initpt_xyz = unit_vector(az, el)
# fig = plt.figure()
# ax = plt.axes(projection ='3d')
# # plotting
# ax.scatter(x_coords, y_coords, z_coords, 'blue')
# # ax.scatter(x_coords_inner, y_coords_inner, z_coords_inner)
# ax.scatter(mic_xyz[0], mic_xyz[1], mic_xyz[2], 'red')
# ax.quiver(mic_xyz[0], mic_xyz[1], mic_xyz[2], initpt_xyz[0], initpt_xyz[1], initpt_xyz[2])
# plt.xlabel("X")
# plt.ylabel("Y")
# ax.set_title('METU RIR measurements')
# plt.show()