import numpy as np
import os
import pickle
import soundfile

import pyroomacoustics as pra
from pyroomacoustics import directivities as dr
from pyroomacoustics.experimental.rt60 import measure_rt60
import argparse
from room_scaper import sofa_utils, tau_loading, room_sim

FOA_REAPER_PATH = "scripts/wav_rirs_foa"

tau_room_list = [
    "bomb_shelter",
    "gym",
    "pb132",
    "pc226",
    "sa203",
    "sc203",
    "se203",
    "tb103",
    "tc352",
]

mat_file_list = [
    "rirs_01_bomb_shelter.mat",
    "rirs_02_gym.mat",
    "rirs_03_pb132.mat",
    "rirs_04_pc226.mat",
    "rirs_05_sa203.mat",
    "rirs_06_sc203.mat",
    "rirs_08_se203.mat",
    "rirs_09_tb103.mat",
    "rirs_10_tc352.mat",
]

# arbitrary room dimensions I made up hehehehehehehe
tau_dim_list = [
    [50, 50, 12],
    [26, 16, 9],
    [9, 7, 4],
    [6, 6, 3],
    [20, 15, 6],
    [11, 8, 4],
    [15, 13, 4],
    [20, 15, 6],
    [6, 5, 3],
]

parser = argparse.ArgumentParser()

parser.add_argument("tau_db_dir", type=str, help="directory for TAU-SRIR-DB")

parser.add_argument("output_dir", type=str, help="directory for file output")

parser.add_argument(
    "--room",
    dest="room",
    type=str,
    help="room to simulate",
    required=False,
    default=None,
)

parser.add_argument(
    "--decay-db",
    dest="decay_db",
    type=int,
    help="decay db for estimating rt60",
    required=False,
    default=15,
)

parser.add_argument(
    "--max-order",
    dest="max_order",
    type=int,
    required=False,
    help="maximum order of reflections for ISM sim",
    default=25,
)

parser.add_argument(
    "--noise-var",
    dest="noise_var",
    type=float,
    required=False,
    help="noise variance passed to 'sigma2_awgn' room parameter",
    default=1,
)

parser.add_argument(
    "--rir-len",
    dest="rir_len",
    type=int,
    required=False,
    help="the length of stored rirs in samples",
    default=7200,
)

parser.add_argument(
    "--sr",
    dest="sr",
    type=int,
    required=False,
    help="the sample rate of the simulation/stored data",
    default=24000,
)

parser.add_argument(
    "--single-path",
    dest="single_path",
    type=bool,
    required=False,
    help="debugging option for computing a single path",
    default=False,
)

parser.add_argument(
    "--mic-center",
    dest="mic_center",
    type=list,
    required=False,
    help="center of microphone array (in meters)",
    default=[0.05, 0.05, 0.05],
)

parser.add_argument(
    "--db-name",
    dest="db_name",
    type=str,
    required=False,
    help="name of database created in script",
    default="TAU-SIM-SOFA",
)

parser.add_argument(
    "--flip",
    dest="flip",
    type=bool,
    required=False,
    help="flip every other height, as in DCASE generator",
    default=True,
)


args = parser.parse_args()

# microphone array definitions
print("Defining mics...")
(
    mic_coords,
    mic_dirs,
) = room_sim.get_tetra_mics()  # this can be subbed for other mic configs
n_mics = len(mic_coords)

mic_loc_center = np.array(args.mic_center)
mic_locs = room_sim.center_mic_coords(mic_coords, mic_loc_center)

if args.room is None:
    room_list = tau_room_list
else:
    room_list = [args.room]


for room_idx, room_name in enumerate(room_list):
    print(f"Loading room info for {room_name}...")
    # load paths
    paths, paths_meta, room_meta = tau_loading.load_paths(room_idx, args.tau_db_dir)
    t_type = room_meta["trajectory_type"]

    # sample rirs for rt60 to calculate MAC
    rir_file = [
        filename for filename in os.listdir(args.tau_db_dir) if room_name in filename
    ][0]
    rir_path = os.path.join(args.tau_db_dir, rir_file)
    samples = tau_loading.load_rir_sample(rir_path, t_type=t_type)
    rt = []
    for i in range(samples.shape[0]):
        rt.append(measure_rt60(samples[i], fs=args.sr, decay_db=args.decay_db))
    rt = np.array(rt) * (60 / args.decay_db)
    rt_avg = np.average(rt)

    room_dim = tau_dim_list[room_idx]
    e_absorption, _ = pra.inverse_sabine(rt_avg, room_dim)

    # place mics in center-ish of room
    room_center = np.array([room_dim[0] / 2, room_dim[1] / 2, 0])
    mic_center = room_meta["microphone_position"] + room_center
    centered_mics = mic_locs + mic_center

    # get dataset dimensions
    n_traj, n_heights = paths.shape
    path_len = len(paths[0, 0])

    if args.single_path:
        print("Simulating only a single path")
        n_traj = 1
        n_heights = 1

    # check for outputdir (create if doesn't exist)
    room_rir_dir = os.path.join(args.output_dir, "foa", room_name)
    if not os.path.exists(room_rir_dir):
        os.mkdir(room_rir_dir)

    # iterating through paths and simulating (one at a time)
    print("Computing rirs...")
    path_stack_all = np.empty((0, 3))
    rir_stack_all = np.empty((0, n_mics, args.rir_len))
    path_to_room_foa_rirs = os.path.join(FOA_REAPER_PATH, room_name)

    for i in range(n_traj):
        mic_array_list = []
        path_stack_traj = np.empty((0, 3))
        rir_stack_traj = np.empty((0, n_mics, args.rir_len))

        nroom_foa_rirs = len(
            os.listdir(os.path.join(path_to_room_foa_rirs, f"{room_name}_t{i}"))
        )

        for j in range(nroom_foa_rirs):
            path_rirs, sr = soundfile.read(
                os.path.join(
                    path_to_room_foa_rirs, f"{room_name}_t{i}", f"rir_{j}-foa.wav"
                )
            )
            path_rirs = path_rirs.T[np.newaxis, ...]

            rir_stack_traj = np.concatenate((rir_stack_traj, path_rirs), axis=0)

        for j in range(n_heights):
            path = paths[i, j]

            if args.flip:
                if j % 2 == 1:
                    # flip every other height, as in DCASE
                    path = path[::-1]

            path_stack_traj = np.concatenate((path_stack_traj, path), axis=0)

        sofa_path = os.path.join(room_rir_dir, f"{room_name}_t{i}.sofa")
        print(f"Storing result to {sofa_path}")

        sofa_utils.create_srir_sofa(
            sofa_path,
            rir_stack_traj,
            path_stack_traj,
            room_meta["microphone_position"],
            db_name=args.db_name,
            room_name=room_name,
            listener_name="mic",
        )

        rir_stack_all = np.concatenate((rir_stack_all, rir_stack_traj), axis=0)
        path_stack_all = np.concatenate((path_stack_all, path_stack_traj), axis=0)

    sofa_path = os.path.join(args.output_dir, "foa", f"{room_name}.sofa")
    print(f"Storing result to {sofa_path}")

    sofa_utils.create_srir_sofa(
        sofa_path,
        rir_stack_all,
        path_stack_all,
        room_meta["microphone_position"],
        db_name=args.db_name,
        room_name=room_name,
        listener_name="mic",
    )
