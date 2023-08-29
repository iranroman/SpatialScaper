import numpy as np
from scipy.io import loadmat
import os
import h5py
import pickle

import pyroomacoustics as pra
from pyroomacoustics import directivities as dr
from pyroomacoustics.experimental.rt60 import measure_rt60
import argparse
import tau_loading
import sys
sys.path.append('../data')
import sofa_utils

tau_room_list = ["bomb_shelter",
             "gym",
             "pb132",
             "pc226",
             "sa203",
             "sc203",
             "se203",
             "tb103",
             "tc352"]

mat_file_list = ['rirs_01_bomb_shelter.mat',
                 'rirs_02_gym.mat',
                 'rirs_03_pb132.mat',
                 'rirs_04_pc226.mat',
                 'rirs_05_sa203.mat',
                 'rirs_06_sc203.mat',
                 'rirs_08_se203.mat',
                 'rirs_09_tb103.mat',
                 'rirs_10_tc352.mat']

#arbitrary room dimensions I made up hehehehehehehe
tau_dim_list = [[50,50,12],
                 [26,16,9],
                 [9, 7, 4],
                 [6,6,3],
                 [20,15,6],
                 [11,8,4],
                 [15,13,4],
                 [20,15,6],
                 [6,5,3]]

def get_y(angle,x):
    angle2 = np.pi-angle-np.pi/2 
    return x * np.sin(angle) / np.sin(angle2)

def deg2rad(deg):
    return deg * 2 * np.pi / 360

def rad2deg(rad):
    return rad * 360 / (2*np.pi)

def plot_energy_db(ax, rir, fs=24000):

    # The power of the impulse response in dB
    power = rir**2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    # remove the possibly all zero tail
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]
    ax.plot(energy_db)

def get_tetra_mics():
    #return geometry of standard tetrahedral mic config as in TAU-SRIR dataset
    
    #coordinates stored in radius (m), azimuth (deg) and elevation (deg)
    m1_coords = [.042, 45, 35]
    m2_coords = [.042, -45, -35]
    m3_coords = [.042, 135, -35]
    m4_coords = [.042, -135, 35]
    
    mic_coords = [m1_coords, m2_coords, m3_coords, m4_coords]
    mic_dirs = [dr.CardioidFamily(orientation=dr.DirectionVector(azimuth=coord[1],
                                                                 colatitude=90-coord[2],
                                                                 degrees=True),
                                  pattern_enum=dr.DirectivityPattern.HYPERCARDIOID,)
                for coord in mic_coords]
    
    return mic_coords, mic_dirs

def center_mic_coords(mic_coords, mic_center):
    mic_locs = np.empty((0,3))
    for coord in mic_coords:
        rad, azi, ele = coord
        azi = deg2rad(azi)
        ele = deg2rad(ele)
        x_offset = rad * np.cos(azi) * np.cos(ele)
        y_offset = rad * np.sin(azi) * np.cos(ele)
        z_offset = rad * np.sin(ele)
        mic_loc = mic_center + np.array([x_offset, y_offset, z_offset])
        mic_locs = np.vstack([mic_locs,mic_loc])
    return mic_locs

def unitvec_to_cartesian(path_unitvec, height, dist):
    if type(dist) == np.ndarray:
        z_offset = height
        rad = np.sqrt(dist[0]**2 + (dist[2]+z_offset)**2)
        scaled_path = map_to_cylinder(path_unitvec, rad, axis=1)
    else:    
        scaled_path = map_to_cylinder(path_unitvec, dist, axis=2)
    return scaled_path

def map_to_cylinder(path, rad, axis=2):
    #maps points (unit vecs) to cylinder of known radius along axis (default z/2)
    scaled_path = np.empty(path.shape)
    rad_axes = [0,1,2]
    rad_axes.remove(axis)
    for i in range(path.shape[0]):
        vec = path[i]
        scale_rad = np.sqrt(np.sum([vec[j]**2 for j in rad_axes]))
        scale = rad / scale_rad
        scaled_path[i] = vec * scale
    return scaled_path

parser = argparse.ArgumentParser()

parser.add_argument("room_name", type=str,
                    help="name of room")

parser.add_argument("--output", dest="output_dir", type=str, 
                    help="directory for file output", required=False,
                    default='/scratch/ci411/SRIR_DATASETS/TAU-SIM-SOFA')

parser.add_argument("--tau-db-dir", dest="tau_db_dir", type=str,
                    help="directory for TAU-SRIR-DB", required=False,
                    default="/scratch/ci411/SRIR_DATASETS/TAU_SRIR_DB/TAU-SRIR_DB")

parser.add_argument("--decay-db", dest="decay_db", type=int,
                    help="decay db for estimating rt60", required=False,
                    default=15)

parser.add_argument("--max-order", dest="max_order", type=int, required=False,
                    help="maximum order of reflections for ISM sim",
                    default=25)

parser.add_argument("--noise-var", dest="noise_var", type=float, required=False,
                    help="noise variance passed to 'sigma2_awgn' room parameter",
                    default=1)

parser.add_argument("--rir-len", dest="rir_len", type=int, required=False,
                    help="the length of stored rirs in samples",
                    default=7200)

parser.add_argument("--sr", dest="sr", type=int, required=False,
                    help="the sample rate of the simulation/stored data",
                    default=24000)

parser.add_argument("--single-path", dest="single_path", type=bool, required=False,
                    help="debugging option for computing a single path",
                    default=False)

parser.add_argument("--mic-center", dest="mic_center", type=list, required=False,
                    help="center of microphone array (in meters)", default=[0.05, 0.05, 0.05])

parser.add_argument("--db-name", dest="db_name", type=str, required=False,
                    help="name of database creating",
                    default="TAU-SIM-SOFA")

parser.add_argument("--flip", dest="flip", type=bool, required=False,
                    help="flip every other height, as in DCASE generator",
                    default=True)


if __name__ == "__main__":

    args = parser.parse_args()

    #microphone array definitions
    print("Defining mics...")
    mic_coords, mic_dirs = get_tetra_mics() #this can be subbed for other mic configs
    n_mics = len(mic_coords)
    
    mic_loc_center = np.array(args.mic_center)
    mic_locs = center_mic_coords(mic_coords, mic_loc_center)

    #load room info
    print("Loading room info...")
    room_idx = tau_room_list.index(args.room_name)
    #load paths
    paths, paths_meta, room_meta = tau_loading.load_paths(room_idx, args.tau_db_dir)
    t_type = room_meta['trajectory_type']
    
    #sample rirs for rt60 to calculate MAC
    rir_file = [filename for filename in os.listdir(args.tau_db_dir) if args.room_name in filename][0]
    rir_path = os.path.join(args.tau_db_dir, rir_file)
    samples = tau_loading.load_rir_sample(rir_path, t_type=t_type)
    rt = []
    for i in range(samples.shape[0]):
        rt.append(measure_rt60(samples[i], fs=args.sr, decay_db=args.decay_db))
    rt = np.array(rt) * (60/args.decay_db)
    rt_avg = np.average(rt)

    room_dim = tau_dim_list[room_idx]
    e_absorption, _ = pra.inverse_sabine(rt_avg, room_dim)

    #place mics in center-ish of room
    room_center = np.array([room_dim[0]/2, room_dim[1]/2, 0])
    mic_center = room_meta['microphone_position'] + room_center
    centered_mics = mic_locs + mic_center

    #get dataset dimensions
    n_traj, n_heights = paths.shape
    path_len = len(paths[0,0])

    if args.single_path:
        n_traj = 1
        n_heights = 1

    #check for outputdir (create if doesn't exist)
    room_rir_dir = os.path.join(args.output_dir, 'mic')
    if not os.path.exists(room_rir_dir):
        os.makedirs(room_rir_dir)

    #iterating through paths and simulating (one at a time)
    print("Computing rirs...")
    path_stack = np.empty((0, 3))
    rir_stack = np.empty((0, n_mics, args.rir_len))
    
    for i in range(n_traj):
        mic_array_list = []
        for j in range(n_heights):

            room = pra.ShoeBox(room_dim, fs=args.sr, 
                             materials=pra.Material(e_absorption),
                             max_order=args.max_order, 
                             sigma2_awgn=args.noise_var)
            room.add_microphone_array(centered_mics.T, directivity=mic_dirs)
            print("t{}h{}".format(i,j))
            path = paths[i,j]
            centered_path = path + mic_center
            path_rirs = np.empty((n_mics, len(path), args.rir_len))
            for source in centered_path:
                try:
                    room.add_source(np.maximum(source,0)) #force source in room
                except ValueError:
                    print("Source at {} is not inside room of dimensions {}".format(source, room_dim))
            room.compute_rir()
            for k in range(n_mics):
                for l in range(len(path)):
                    path_rirs[k,l] = room.rir[k][l][:args.rir_len]
            
            if args.flip:
                if j%2==1:
                    #flip every other height, as in DCASE
                    path_rirs = path_rirs[::-1]
                    path = path[::-1]
            
            path_rirs = np.moveaxis(path_rirs, [0,1,2], [1,0,2])
            rir_stack = np.concatenate((rir_stack, path_rirs), axis=0)
            path_stack = np.concatenate((path_stack, path), axis=0)

    sofa_path = os.path.join(room_rir_dir, f'{args.room_name}.sofa')
    print(f"Storing result to {sofa_path}")
    
    sofa_utils.create_srir_sofa(sofa_path, rir_stack, path_stack, room_meta['microphone_position'],\
                     db_name=args.db_name, room_name=args.room_name, listener_name='mic')