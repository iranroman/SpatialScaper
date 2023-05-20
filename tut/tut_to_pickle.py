import os
import math
import argparse
import pickle
import numpy as np
import scipy.io
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt


def compute_unit_vector(elevation_deg, azimuth_deg):
    # from degree angles to radians
    elevation_rad = math.radians(elevation_deg)
    azimuth_rad = math.radians(azimuth_deg)
    # get the x, y, z components of the unit vector
    x = math.cos(elevation_rad) * math.cos(azimuth_rad)
    y = math.cos(elevation_rad) * math.sin(azimuth_rad)
    z = math.sin(elevation_rad)
    length = math.sqrt(x**2 + y**2 + z**2)
    x /= length
    y /= length
    z /= length
    return x, y, z

TUT_PATH = "/mnt/ssdt7/RIR-datasets/TUT/Tietotalo_RIR.mat"

def main():
    tut_mat = scipy.io.loadmat(TUT_PATH)
    NFFT = 1025
    elevation_deg = np.arange(-40, 50, 10) # list of elevation degrees on IRs
    azimuth_deg = np.arange(-180, 190, 10) # list of azimuth degrees on IRs  
    tut_rirdata = tut_mat["rir_DB"]
    tut_rirdata = tut_rirdata.transpose(0, 1, 3, 5, 2, 4)
    print(tut_rirdata.shape)
    n_traj, n_height, n_azimuth, n_ch, n_fft, n_blocks = tut_rirdata.shape
    # Data structure (potentially to be used once this is fully integrated)
    room = "tut"
    rirdata_dict = {}
    rirdata_dict[room] = {}
    rirdata_dict[room]['doa_xyz'] = [[]*n_traj]
    rirdata_dict[room]['dist'] = [[]*n_traj]
    rirdata_dict[room]['rir'] = {'mic':[[]*n_traj]}
    for traj in range(n_traj):
        for height in tqdm(range(n_height)):
            doa_xyz = []  # list of unit vectors for each doa per height
            hirdata = []  # impulse responses list per height
            dists = [] # list of distances
            for azi in range(n_azimuth):
                irchan_list = [] # list of impulse responses per channel
                for ch in range(n_ch):
                    # Compute time-domain impulse response
                    isfft_data = librosa.istft(tut_rirdata[traj,height,azi,ch,:,:].T, win_length=None, n_fft=NFFT,
                                                window='hann', center=True, dtype=None, length=None)
                    irchan_list.append(isfft_data)
                irdata = np.stack(irchan_list, axis=1)
                hirdata.append(irdata)
                dists.append(1) # TODO: add a correct distance here
                # Calculate unit vector from azimuth and elevation
                uvec = compute_unit_vector(elevation_deg[height], azimuth_deg[azi])
                doa_xyz.append(uvec)
            rirdata_dict[room]['doa_xyz'][traj].append(np.array(doa_xyz))
            rirdata_dict[room]['rir']['mic'][traj].append(np.transpose(np.array(hirdata), (1, 2, 0))) # (Nsamps, Nch, N_ir)

    with open('{}.pkl'.format(f"rir_12_tut"), 'wb') as outp: # should go inside TAU_DB/TAU-SRIR_DB/rirs_12_tut.pkl
        pickle.dump(rirdata_dict[room]['rir'], outp, pickle.HIGHEST_PROTOCOL)

    with open('{}.pkl'.format(f"doa_xyz_tut"), 'wb') as outp:
        pickle.dump(rirdata_dict[room]['doa_xyz'], outp, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
