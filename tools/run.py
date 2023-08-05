from room_scaper.utils.parser import parse_args, load_config
from room_scaper.data.utils import get_path_to_room_files
import yaml
import pickle
import os
import numpy as np
import librosa

MOVE_THRESHOLD = 3

def cart2sph(xyz):
    return_list = False
    if len(np.shape(xyz)) == 2:
        return_list = True
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
    else:
        x = xyz[0]
        y = xyz[1]
        z = xyz[2]
    
    azimuth = np.arctan2(y, x) * 180. / np.pi
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2)) * 180. / np.pi
    if return_list:
        return np.stack((azimuth,elevation),axis=0)
    else:
        return np.array([azimuth, elevation])


def get_fold_files(foldname, filenames):
    """
    assuming the foldname is in the
    relevant filenames path
    """
    fold_files = [fname for fname in filenames if foldname in fname.split('/')]
    #fold_durs = [float(f.split('/')[-1]) for f in filedurs if foldname in f.split('/')]
    if fold_files:
        sampleperm = np.random.permutation(len(fold_files))
        return [fold_files[i] for i in sampleperm]
    else:
        import warnings
        warnings.warn(f'No files found for fold {foldname}')

def get_sound_event_filenames(path):
    if path.endswith('.txt'):
        filenames = []
        with open(path) as file:
            while line := file.readline():
                filenames.append(line.strip())
    # TODO: make this work with recursive listing of files in a directory
    return filenames

def load_pickle(filename):
    file = open(filename,'rb')
    object_file = pickle.load(file)
    file.close()
    return object_file

def get_room_trajectories(path_to_room_files):
    '''
    room_name: a string with the name
    '''
    room_trajs = load_pickle(os.path.join(path_to_room_files,'metadata','doa_xyz.pkl'))
    return room_trajs

def get_traj_n_heights_n_rirs(ntraj, room_trajs):
    n_heights = len(room_trajs[ntraj])
    n_rirs = sum([len(room_trajs[ntraj][iheight]) for iheight in range(n_heights)])
    return n_heights, n_rirs

def get_traj_all_doas(room_trajs, ntraj, n_heights, n_rirs):
    all_doas = np.zeros((n_rirs, 3))
    n_rirs_accum = 0
    flip = 0
    for nheight in range(n_heights):
        n_rirs_nh = len(room_trajs[ntraj][nheight])
        doa_xyz = room_trajs[ntraj][nheight]
        #   stack all doas of trajectory together
        #   flip the direction of each second height, so that a
        #   movement can jump from the lower to the higher smoothly and
        #   continue moving the opposite direction
        if flip:
            nb_doas = np.shape(doa_xyz)[0]
            all_doas[n_rirs_accum + np.arange(n_rirs_nh), :] = doa_xyz[np.flip(np.arange(nb_doas)), :]
        else:
            all_doas[n_rirs_accum + np.arange(n_rirs_nh), :] = doa_xyz
        
        n_rirs_accum += n_rirs_nh
        flip = not flip
        
    return all_doas
            
def get_traj_doas(room_trajs, n_traj):

    traj_doas = []
    for ntraj in range(n_traj):

        n_heights, n_rirs = get_traj_n_heights_n_rirs(ntraj, room_trajs)

        all_doas = get_traj_all_doas(room_trajs, ntraj, n_heights, n_rirs)

        traj_doas.append(all_doas)

    return traj_doas

def get_filename_class_duration(fold_event_filenames, path_to_dataset, class_dict=None):

    irand = np.random.randint(len(fold_event_filenames))

    filename = fold_event_filenames[irand]
    filepath = os.path.join(path_to_dataset, filename.split('/')[-1]) # HACKY, BAD, Will improve with Adrian's contrib
    filedur = librosa.get_duration(path=filepath)
    classid = filename.split('/')[0] # HACKY, BAD, Will improve with Adrian's contrib
    if class_dict:
       classid = class_dict[classid]
    return {'filename': filename, 'filepath': filepath, 'classid':classid, 'filedur':filedur}

def find_saturated_timepoints(track_events, max_polyphony, dt=0.1):

    if track_events:
        track_events = np.array(track_events)
        times = []
        for event_time in track_events:
            t_ = np.arange(*event_time, dt)
            times.extend([round(t,1) for t in t_])
        saturated_times = set([x for x in times if times.count(x) >= max_polyphony])
    else:
        saturated_times = []
    return saturated_times

def get_event_start_stop(max_time, filedur, satur_times, dt):
    event_start = np.random.choice(np.arange(0, max_time, dt))
    event_stop = event_start + (np.ceil(filedur*10)/10)
    event_times = np.arange(event_start, event_stop+dt, dt)
    event_times = [round(t,1) for t in event_times]

    if any([t in satur_times for t in event_times]):
        return get_event_start_stop(max_time, filedur, satur_times, dt)
    else:
        return [round(event_start,1), round(event_stop,1)]

def place_event_in_track_time(track_events, source_file_metadata, max_polyphony, track_dur, dt):

    satur_times = find_saturated_timepoints(track_events, max_polyphony, dt)

    # here we need to incorporate the placement of events in time using
    # distributions like scaper does
    # in the meantime:
    filedur = source_file_metadata['filedur']
    max_time = track_dur - filedur
    event_start, event_stop = get_event_start_stop(max_time, filedur, satur_times, dt)
    track_events.append([event_start, event_stop])
    return track_events

def sort_by_event_onset(all_events_meta, event_onsets):

    event_onsets = np.array(event_onsets)
    idx = np.argsort(event_onsets[:,0])

    return [all_events_meta[i] for i in idx]


def generate_nth_mixture_dict(all_events_meta):
    nth_mixture = {}
    nth_mixture['files'] = np.array([event['filename'] for event in all_events_meta])
    nth_mixture['class'] = np.array([event['classid'] for event in all_events_meta])
    nth_mixture['event_onoffsets'] = np.array([event['event_onoffsets'] for event in all_events_meta])
    nth_mixture['sample_onoffsets'] = np.array([np.array([0.0,np.floor(event['filedur']*10)/10]) for event in all_events_meta])
    nth_mixture['trajectory'] = np.array([event['traj'] for event in all_events_meta])
    nth_mixture['isMoving'] = np.array([event['isMoving'] for event in all_events_meta])
    nth_mixture['isFlippedMoving'] = np.array([event['isFlippedMoving'] for event in all_events_meta])
    nth_mixture['speed'] = np.array([event['speed'] for event in all_events_meta])
    nth_mixture['rirs'] = [np.array(event['rirs']) for event in all_events_meta]
    nth_mixture['doa_azel'] = [np.array(event['doa_azel']) for event in all_events_meta]
    nth_mixture['snr'] = all_events_meta[0]['snr'] # TODO: how to make each event have its own SNR?

    return nth_mixture

def get_event_riridx(n_traj, source_file_metadata, traj_doas, speed_set, MOVE_THRESHOLD=3):

    # trajectory
    ev_traj = np.random.randint(0, n_traj)
    nRirs = traj_doas[ev_traj].shape[0]
    if source_file_metadata['filedur'] <= MOVE_THRESHOLD:
        is_moving = 0 
    else:
        # randomly moving or static
        is_moving = np.random.randint(0,2)

    if is_moving:
        ev_nspeed = np.random.randint(0,len(speed_set))
        ev_speed = speed_set[ev_nspeed]
        # check if with the current speed there are enough
        # RIRs in the trajectory to move through the full
        # duration of the event, otherwise, lower speed
        event_duration_nl = source_file_metadata['filedur']*10
        while len(np.arange(0,nRirs,ev_speed/10)) <= event_duration_nl:
            ev_nspeed = ev_nspeed-1
            if ev_nspeed == -1:
                break
            ev_speed = speed_set[ev_nspeed]
        
        is_flipped_moving = np.random.randint(0,2)
        event_span_nl = event_duration_nl * ev_speed / 10.
            
        if is_flipped_moving:
            # sample length is shorter than all the RIRs
            # in the moving trajectory
            if ev_nspeed+1:
                end_idx = event_span_nl + np.random.randint(0, nRirs-event_span_nl+1)
                start_idx = end_idx - event_span_nl
                riridx = start_idx + np.arange(0, event_span_nl, dtype=int)
                riridx = riridx[np.arange(0,len(riridx),ev_speed/10,dtype=int)] #pick every nth RIR based on speed
                riridx = np.flip(riridx)
            else:
                riridx = np.arange(event_span_nl,0,-1)-1
                riridx = riridx - (event_span_nl-nRirs)
                riridx = riridx[np.arange(0, len(riridx), ev_speed/10, dtype=int)]
                riridx[riridx<0] = 0
        else:
            if ev_nspeed+1:
                start_idx = np.random.randint(0, nRirs-event_span_nl+1)
                riridx = start_idx + np.arange(0,event_span_nl,dtype=int) - 1
                riridx = riridx[np.arange(0,len(riridx),ev_speed/10,dtype=int)]
            else:
                riridx = np.arange(0,event_span_nl)
                riridx = riridx[np.arange(0,len(riridx),ev_speed/10,dtype=int)]
                riridx[riridx>nRirs-1] = nRirs-1
    else:
        is_flipped_moving = 0
        ev_speed = 0
        riridx = np.array([np.random.randint(0,nRirs)])
    riridx = riridx.astype('int')

    return riridx, is_moving, is_flipped_moving, ev_speed, ev_traj

def DCASE_main():

    # 0. parse the yaml with the config. Define if you are generating data in 
    # the "training" or "testing" fold then, for each audio mixture you generate:
    # parse config arguments
    args = parse_args()
    cfg = load_config(args, args.path_to_config)
    print('generating data using parameters:')
    print(yaml.dump(dict(cfg), allow_unicode=True, default_flow_style=False))

    
    event_filenames = get_sound_event_filenames(cfg.PATH_TO_SOUND_EVENT_FILES)

    fold_names = cfg.FOLD_NAMES
    snr_range = cfg.SNR_RANGE
    max_polyphony = cfg.MAX_POLYPHONY
    mixtures = []

    # iterate over fold names
    for ifold, fold in enumerate(fold_names):
        
        fold_event_filenames = get_fold_files(
                fold, 
                event_filenames)
        n_mixtures_per_fold = cfg.N_MIX_PER_FOLD[ifold]

        #######################
        # synthesize metadata #
        #######################

        # 1. determine a room and gather possible IRs and locations. If synthetic room, 
        # just define room shape parameters (the IRs will be generated only when relevant)
        # iterate over rooms
        fold_rooms = cfg.FOLD_ROOMS[fold] 
        room_mixtures=[]
        for room_name in fold_rooms:

            # get the room's relevant info
            path_to_room_files = get_path_to_room_files(room_name)
            room_trajs = get_room_trajectories(path_to_room_files)
            n_traj = len(room_trajs) 
            traj_doas = get_traj_doas(room_trajs, n_traj) # TO BE SUBSTITUTED by "draw traj"

            # 2. add sound events to the mixture by determininig:
            for imixture in range(n_mixtures_per_fold//len(fold_rooms)):

                # this can also be probabilistic
                nevents = cfg.NUM_EVENTS_IN_MIXTURE

                track_events = [] # to ensure MAX_POLYPHONY
                all_events_meta = []
                for ievent in range(nevents):
                    # a. determine the label and the specific source file. The source time will
                    # be from 0 to its total duration. Map the label to the DCASE index
                    source_file_metadata = get_filename_class_duration(fold_event_filenames, cfg.PATH_TO_DATASET, cfg.CLASS_DICT)

                    # b. assign an event onoffset in the duration of the track. Make sure that no
                    # more than MAX_POLYPHONY events overlap at a time
                    track_events = place_event_in_track_time(track_events, source_file_metadata, cfg.MAX_POLYPHONY, cfg.MIXTURE_DUR, 1/cfg.METADATA_SR)
                    source_file_metadata['event_onoffsets'] = track_events[-1]

                    # c. determine the SNR from SNR_RANGE
                    source_file_metadata['snr'] = np.random.randint(*cfg.SNR_RANGE)

                    # d. determine whether the event_effects "pitch_shift" or "time_stretch"
                    # should be applied, and with which parameters

                    # e. place the sound in an initial precise location (a=[x,y,z] coordinates)
                    # real_rooms: if the coordinate does not exist, place its NN (warn user)

                    # f. determine whether the sound will be static or move in a trajectory:
                    # from a to b, around a (what speed?), random path (what speed?)
                    riridx, is_moving, is_flipped_moving, ev_speed, ev_traj = get_event_riridx(n_traj, source_file_metadata, traj_doas, cfg.SPEED_SET)

                    source_file_metadata['traj'] = ev_traj
                    source_file_metadata['isMoving'] = is_moving
                    source_file_metadata['isFlippedMoving'] = is_flipped_moving
                    source_file_metadata['speed'] = ev_speed
                    source_file_metadata['rirs'] = riridx
                    source_file_metadata['doa_azel'] = [cart2sph(traj_doas[ev_traj][riridx,:])]

                    all_events_meta.append(source_file_metadata)

                all_events_meta = sort_by_event_onset(all_events_meta, track_events)

                nth_mixture = generate_nth_mixture_dict(all_events_meta)
                nth_mixture['room'] = room_name

                # sanity check
                for k,v in nth_mixture.items(): 
                    print(k)
                    input()
                    print(v)
                    input()

    # 3. determine whether the mixture will undergo "channel_swap" augmentations
    # (channel_swap only applies to RIRs from real rooms) 

    # 4. dump resulting file with annotations



if __name__ == "__main__":
    DCASE_main()
