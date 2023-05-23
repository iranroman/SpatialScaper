import librosa
import numpy as np
import os
import math
import pickle
from pysofaconventions import *
import matplotlib.pyplot as plt

from arni_utils import *

'''
Important:
    This approach makes a single trayectory from the different DOAs on mics 1-5. It sorts the RIRs based on DOAs azimuth in 
    decreasing order. The absorbtion level defines a different trayectory, making a total of 5 trayectories.
'''

RIR_DB = '/mnt/ssdt7/RIR-datasets/Arni/6dof_SRIRs_eigenmike_raw/'
FS = 48000 # original impulse reponse sampling rate
NEW_FS = 24000 # new sampling rate (same as DCASE Synth)

# Load the .sofa file
rir_db_path = RIR_DB
rir_files = os.listdir(rir_db_path)

num_traj = len(rir_files)

assert num_traj != 0, f"Error: {rir_db_path} contains no .sofa files"

rirdata_dict = {}
room = "arni"
rirdata_dict[room] = {}
rirdata_dict[room]['doa_xyz'] = [[] for i in range(num_traj)]
rirdata_dict[room]['dist'] = [[] for i in range(num_traj)]
rirdata_dict[room]['rir'] = {'mic':[[] for i in range(num_traj)]}

for meas_traj, sofa_file_traj in enumerate(rir_files): # for each .sofa a.k.a trajectory
    print(f"Computing trayectory {meas_traj} as:", sofa_file_traj)
    sofa = SOFAFile(os.path.join(rir_db_path, sofa_file_traj),'r')
    if not sofa.isValid():
        print("Error: the file is invalid")
        break
    sourcePositions = sofa.getVariableValue('SourcePosition') # get sound source position
    listenerPosition = sofa.getVariableValue('ListenerPosition') # get mic position
    # get RIR data
    rirdata = sofa.getDataIR()
    num_meas, num_ch = rirdata.shape[0], rirdata.shape[1]
    meas_per_mic = 3 # equal the number of meas per trajectory
    num_meas = 15 # set num_meas to 15 to keep south mics only
    angles_mic_src = [math.degrees(compute_azimuth_elevation(lis, src)[0]) \
                        for lis, src in zip(listenerPosition[:num_meas], sourcePositions[:num_meas])]
    meas_sorted_ord = np.argsort(angles_mic_src)[::-1]
    sorted_angles_mic_src = [angles_mic_src[i] for i in meas_sorted_ord]
    doa_xyz, dists, hir_data = [], [], [] # assume only one height
    for meas in meas_sorted_ord: # for each meas in decreasing order
        # add impulse response
        irdata = rirdata[meas, :, :]
        irdata_resamp = librosa.resample(irdata, orig_sr=FS, target_sr=NEW_FS)
        hir_data.append(irdata_resamp)
        azi, ele, dis = compute_azimuth_elevation(listenerPosition[meas], sourcePositions[meas])
        uvec_xyz = unit_vector(azi, ele)
        doa_xyz.append(uvec_xyz)
        dists.append(dis)
    rirdata_dict[room]['doa_xyz'][meas_traj].append(np.array(doa_xyz))
    rirdata_dict[room]['rir']['mic'][meas_traj].append(np.transpose(np.array(hir_data),(2,1,0))) # (Nsamps, Nch, N_ir)
    
with open('{}.pkl'.format(f"rirs_13_arni"), 'wb') as outp: # should go inside TAU_DB/TAU-SRIR_DB/rirs_13_arni.pkl
    pickle.dump(rirdata_dict[room]['rir'], outp, pickle.HIGHEST_PROTOCOL)

with open('{}.pkl'.format(f"doa_xyz_arni"), 'wb') as outp:
    pickle.dump(rirdata_dict[room]['doa_xyz'], outp, pickle.HIGHEST_PROTOCOL)



