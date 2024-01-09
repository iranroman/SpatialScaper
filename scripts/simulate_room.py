import numpy as np
import os
import pickle

import pyroomacoustics as pra
from pyroomacoustics import directivities as dr
import argparse
from room_scaper import sofa_utils, room_sim

parser = argparse.ArgumentParser()

parser.add_argument("room_name", type=str,
                    help="name of new room")

parser.add_argument("x_dim", type=float,
                    help="x dimension of new room")

parser.add_argument("y_dim", type=float,
                    help="y dimension of new room")

parser.add_argument("z_dim", type=float,
                    help="z dimension of new room")

parser.add_argument("--heights", dest="heights", type=list,
                    help="list of heights to set trajectories",
                    default=[.4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.])

parser.add_argument("--density-scale", dest="density_scale", type=int, required=False,
                    help="inverse of sampling density (e.g. 2 = half density)", default=1)

parser.add_argument("--output-dir", dest="output_dir", type=str, 
                    help="directory for file output",
                    default='/scratch/ci411/SRIR_ABLATION/ROOM_VOLS')

parser.add_argument("--tau-db-dir", dest="tau_db_dir", type=str, required=False,
                    help="directory for TAU-SRIR-DB",
                    default='/scratch/ci411/SRIR_DATASETS/TAU_SRIR_DB/TAU-SRIR_DB')

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

parser.add_argument("--mic-center", dest="mic_center", type=list, required=False,
                    help="center of microphone array (in meters)", default=[0.05, 0.05, 0.05])

parser.add_argument("--flip", dest="flip", type=bool, required=False,
                    help="flip every other height, as in DCASE generator",
                    default=True)

parser.add_argument("--db-name", dest="db_name", type=str, required=False,
                    help="name of database created in script",
                    default="RoomGen")


def get_distances(room_dims):
    dist_large = np.floor(min([room_dims[0],room_dims[1]]))/2
    dist_small = np.floor(min([room_dims[0],room_dims[1]]))/4
    return [dist_small, dist_large]

args = parser.parse_args()

room_dims = [args.x_dim, args.y_dim, args.z_dim]
print(room_dims)

#microphone array definitions
print("Defining mics...")
mic_coords, mic_dirs = room_sim.get_tetra_mics() #this can be subbed for other mic configs
n_mics = len(mic_coords)

mic_loc_center = np.array(args.mic_center)
mic_locs = room_sim.center_mic_coords(mic_coords, mic_loc_center)
    
#define room conditions
distances = get_distances(room_dims)

m = pra.make_materials(
    ceiling="hard_surface",
    floor="carpet_soft_10mm",
    east="gypsum_board",
    west="gypsum_board",
    north="gypsum_board",
    south="gypsum_board",
)

room_center = np.array([room_dims[0]/2, room_dims[1]/2, 0.1])
centered_mics = mic_locs + room_center

#define trajectories
for i, dist in enumerate(distances):
    path_stack_traj = np.empty((0, 3))
    rir_stack_traj = np.empty((0, n_mics, args.rir_len))
    n_meas = 0
    for j, height in enumerate(args.heights):
        print(f"Computing d{dist}, h{height}")
        room = pra.ShoeBox(
           room_dims, fs=args.sr, materials=m, max_order=args.max_order,
           air_absorption=True
        )
        room.add_microphone_array(centered_mics.T, directivity=mic_dirs)
        ts = np.linspace(0,2*np.pi, 360//args.density_scale)
        ring = [[dist * np.cos(t), dist * np.sin(t), height] for t in ts]
        n_meas += len(ring)
        if args.flip and j%2==1:
            ring = ring[::-1]
        for source in ring:
            room.add_source(np.maximum(source,0)) #force source in room
            
        path_rirs = np.empty((n_mics, len(ring), args.rir_len))
        room.compute_rir()
            
        for k in range(n_mics):
            for l in range(len(ring)):
                path_rirs[k,l] = room.rir[k][l][:args.rir_len]

        path_rirs = np.moveaxis(path_rirs, [0,1,2], [1,0,2])
        
        rir_stack_traj = np.concatenate((rir_stack_traj, path_rirs), axis=0)
        path_stack_traj = np.concatenate((path_stack_traj, ring), axis=0)
        
    sofa_path = os.path.join(args.output_dir, 'mic', args.room_name, f'{args.room_name}_t{i}.sofa')
    print(f"Storing result to {sofa_path}")
    sofa_utils.create_srir_sofa(sofa_path, rir_stack_traj, path_stack_traj, np.array([room_center for _ in range(n_meas)]),\
                                db_name=args.db_name, room_name=args.room_name, listener_name='mic')