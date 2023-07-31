from room_scaper.utils.parser import parse_args, load_config
from room_scaper.data.utils import get_path_to_room_files
import yaml
import pickle
import os
import numpy as np

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


def get_fold_files_and_durs(foldname, filenames, filedurs):
    """
    assuming the foldname is in the
    relevant filenames path
    """
    fold_files = [fname for fname in filenames if foldname in fname.split('/')]
    fold_durs = [float(f.split('/')[-1]) for f in filedurs if foldname in f.split('/')]
    assert len(fold_files) == len(fold_durs)
    if fold_files:
        sampleperm = np.random.permutation(len(fold_files))
        return [fold_files[i] for i in sampleperm], [fold_durs[i] for i in sampleperm], 
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

def get_sound_event_filedurs(path):
    if path.endswith('.txt'):
        filedurs = []
        with open(path) as file:
            while line := file.readline():
                filedurs.append(line.strip())
    # TODO: make this work with recursive listing of files in a directory
    return filedurs
        
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

def initialize_mixture_metadata(room_name, snr_range):
    nth_mixture = {
            'files': np.array([]), 
            'class': np.array([]), 
            'event_onoffsets': np.array([]),
            'sample_onoffsets': np.array([]), 
            'trajectory': np.array([]), 
            'isMoving': np.array([]), 
            'isFlippedMoving': np.array([]),
            'speed': np.array([]), 
            'rirs': [], 
            'doa_azel': np.array([],dtype=object)
    }
    nth_mixture['room'] = room_name
    nth_mixture['snr'] = np.random.choice(range(*snr_range))
    return nth_mixture

def get_layer_event_indices(fold_event_durs, mixture_events_dur_sum, event_counter, sample_counter):
    TRIMMED_SAMPLE_AT_END = 0
    #fetch event samps till they add up to the target event time per layer
    event_time_in_layer = 0
    event_idx_in_layer = []

    while event_time_in_layer < mixture_events_dur_sum:
        #get event duration
        ev_duration = np.ceil(fold_event_durs[sample_counter]*10.)/10.
        event_time_in_layer += ev_duration
        event_idx_in_layer.append(sample_counter)
        
        event_counter += 1
        sample_counter += 1
  
        if sample_counter == len(fold_event_durs):
            sample_counter = 0
    overshoot = mixture_events_dur_sum - (event_time_in_layer - ev_duration)
    if overshoot:
        event_counter -= 1
        sample_counter -= 1 if sample_counter else len(fold_event_durs) - 1
        event_time_in_layer -= ev_duration
        event_idx_in_layer = event_idx_in_layer[:-1]
    return event_idx_in_layer, event_counter, sample_counter

def get_silence_gaps(nevents_in_layer, layer_silence_dur, min_gap_length):
    # split silences between events
    # randomize N split points uniformly for N events (in
    # steps of 100msec)
    mult_silence = np.round(layer_silence_dur*10.)
    
    mult_min_gap_len = np.round(min_gap_length*10.)
    if nevents_in_layer > 1:
        
        silence_splits = np.sort(np.random.randint(1, mult_silence, nevents_in_layer-1))
        #force gaps smaller then _min_gap_len to it
        gaps = np.diff(np.concatenate(([0],silence_splits,[mult_silence])))
        smallgaps_idx = np.argwhere(gaps[:(nevents_in_layer-1)] < mult_min_gap_len)
        while np.any(smallgaps_idx):
            temp = np.concatenate(([0], silence_splits))
            silence_splits[smallgaps_idx] = temp[smallgaps_idx] + mult_min_gap_len
            gaps = np.diff(np.concatenate(([0],silence_splits,[mult_silence])))
            smallgaps_idx = np.argwhere(gaps[:(nevents_in_layer-1)] < mult_min_gap_len)
        if np.any(gaps < mult_min_gap_len):
            min_idx = np.argwhere(gaps < mult_min_gap_len)
            gaps[min_idx] = mult_min_gap_len
        # if gaps[nb_samples_in_layer-1] < mult_min_gap_len:
        #     gaps[nb_samples_in_layer-1] = mult_min_gap_len
        
    else:
        gaps = np.array([mult_silence])

    while np.sum(gaps) > layer_silence_dur*10.:
        silence_diff = np.sum(gaps) - layer_silence_dur*10.
        picked_gaps = np.argwhere(gaps > np.mean(gaps))
        eq_subtract = silence_diff / len(picked_gaps)
        picked_gaps = np.argwhere((gaps - eq_subtract) > mult_min_gap_len)
        gaps[picked_gaps] -= eq_subtract
    return gaps
                            

def get_event_riridx_onoffset(nl, silence_gaps, event_idx_in_layer, fold_event_durs, fold_event_filenames, cfg, time_idx, n_traj, traj_doas):
    # TODO: modify to not pass `cfg`
    gap_nl = int(silence_gaps[nl])
    time_idx += gap_nl
    event_nl = event_idx_in_layer[nl]
    event_duration_nl = np.ceil(fold_event_durs[event_nl]*10.)
    event_class_nl = cfg.CLASS_DICT[fold_event_filenames[event_nl].split('/')[0]] 
    onoffsets = np.array([0,fold_event_durs[event_nl]])

    sample_onoffsets = np.floor(onoffsets*10.)/10.

    # trajectory
    ev_traj = np.random.randint(0, n_traj)
    nRirs = traj_doas[ev_traj].shape[0]
    if event_duration_nl <= MOVE_THRESHOLD*10:
        is_moving = 0 
    else:
        if cfg.CLASS_MOBILITY == 2:
            # randomly moving or static
            is_moving = np.random.randint(0,2)
        else:
            # only static or moving depending on class
            is_moving = cfg.CLASS_MOBILITY[event_class_nl]

    if is_moving:
        ev_nspeed = np.random.randint(0,len(cfg.SPEED_SET))
        ev_speed = cfg.SPEED_SET[ev_nspeed]
        # check if with the current speed there are enough
        # RIRs in the trajectory to move through the full
        # duration of the event, otherwise, lower speed
        while len(np.arange(0,nRirs,ev_speed/10)) <= event_duration_nl:
            ev_nspeed = ev_nspeed-1
            if ev_nspeed == -1:
                break

            ev_speed = cfg.SPEED_SET[ev_nspeed]
        
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

    return event_duration_nl, riridx, sample_onoffsets, ev_traj, is_moving, is_flipped_moving, ev_speed, event_nl, time_idx

def update_nth_mixture(nth_mixture, riridx, time_idx, event_duration_nl, sample_onoffsets, traj_doas, nl, layer, ev_traj, fold_event_filenames, is_moving, is_flipped_moving, ev_speed, event_nl, cfg):
    if nl == 0 and layer==0:
        nth_mixture['event_onoffsets'] = np.array([[time_idx/10., (time_idx+event_duration_nl)/10.]])
        nth_mixture['doa_azel'] = [cart2sph(traj_doas[ev_traj][riridx,:])]
        nth_mixture['sample_onoffsets'] = [sample_onoffsets]
    else:
        nth_mixture['event_onoffsets'] = np.vstack((nth_mixture['event_onoffsets'], np.array([time_idx/10., (time_idx+event_duration_nl)/10.])))
        nth_mixture['doa_azel'].append(cart2sph(traj_doas[ev_traj][riridx,:]))
        nth_mixture['sample_onoffsets'].append(sample_onoffsets)
                 
    nth_mixture['files'] = np.append(nth_mixture['files'], fold_event_filenames[event_nl])
    nth_mixture['class'] = np.append(nth_mixture['class'], cfg.CLASS_DICT[fold_event_filenames[event_nl].split('/')[0]]) 
    nth_mixture['trajectory'] = np.append(nth_mixture['trajectory'], ev_traj)
    nth_mixture['isMoving'] = np.append(nth_mixture['isMoving'], is_moving)
    nth_mixture['isFlippedMoving'] = np.append(nth_mixture['isFlippedMoving'], is_flipped_moving)
    nth_mixture['speed'] = np.append(nth_mixture['speed'], ev_speed)
    nth_mixture['rirs'].append(riridx)
    return nth_mixture


def populate_mixture(cfg, sample_counter, fold_event_durs, fold_event_filenames, n_traj, traj_doas, max_polyphony, nth_mixture):
    event_counter = 0

    # populate each "layer" in the mixture
    for layer in range(max_polyphony):

        event_idx_in_layer, event_counter, sample_counter = get_layer_event_indices(fold_event_durs, cfg.MIXTURE_DUR - cfg.MIXTURE_LAYER_SILENCE, event_counter, sample_counter)
        nevents_in_layer = len(event_idx_in_layer)

        silence_gaps = get_silence_gaps(nevents_in_layer, cfg.MIXTURE_LAYER_SILENCE, cfg.MIN_GAP_BTW_LAYER_EVENTS)

        # distribute each event in the timeline
        time_idx = 0
        for nl in range(nevents_in_layer):

            event_duration_nl, riridx, sample_onoffsets, ev_traj, is_moving, is_flipped_moving, ev_speed, event_nl, time_idx = get_event_riridx_onoffset(nl, silence_gaps, event_idx_in_layer, fold_event_durs, fold_event_filenames, cfg, time_idx, n_traj, traj_doas)

            nth_mixture = update_nth_mixture(nth_mixture, riridx, time_idx, event_duration_nl, sample_onoffsets, traj_doas, nl, layer, ev_traj, fold_event_filenames, is_moving, is_flipped_moving, ev_speed, event_nl, cfg)

            time_idx += event_duration_nl

    sort_idx = np.argsort(nth_mixture['event_onoffsets'][:,0])
    nth_mixture['files'] = nth_mixture['files'][sort_idx]
    nth_mixture['class'] = nth_mixture['class'][sort_idx]
    nth_mixture['event_onoffsets'] = nth_mixture['event_onoffsets'][sort_idx]
    #nth_mixture['sample_onoffsets'] = nth_mixture['sample_onoffsets'][sort_idx]
    nth_mixture['trajectory'] = nth_mixture['trajectory'][sort_idx]
    nth_mixture['isMoving'] = nth_mixture['isMoving'][sort_idx]
    nth_mixture['isFlippedMoving'] = nth_mixture['isFlippedMoving'][sort_idx]
    nth_mixture['speed'] = nth_mixture['speed'][sort_idx]
    nth_mixture['rirs'] = np.array(nth_mixture['rirs'],dtype=object)
    nth_mixture['rirs'] = nth_mixture['rirs'][sort_idx]
    new_doas = np.zeros(len(sort_idx),dtype=object)
    new_sample_onoffsets = np.zeros(len(sort_idx),dtype=object)
    upd_idx = 0
    for idx in sort_idx:
        new_doas[upd_idx] = nth_mixture['doa_azel'][idx].T
        new_sample_onoffsets[upd_idx] = nth_mixture['sample_onoffsets'][idx]
        upd_idx += 1
    nth_mixture['doa_azel'] = new_doas
    nth_mixture['sample_onoffsets'] = new_sample_onoffsets 

    return nth_mixture

def main():
    '''
    main function to trigger the data generation
    '''

    # parse config arguments
    args = parse_args()
    cfg = load_config(args, args.path_to_config)
    print('generating data using parameters:')
    print(yaml.dump(dict(cfg), allow_unicode=True, default_flow_style=False))

    
    event_filenames = get_sound_event_filenames(cfg.PATH_TO_SOUND_EVENT_FILES)
    event_durs = get_sound_event_filedurs(cfg.PATH_TO_SOUND_EVENT_DURS)


    fold_names = cfg.FOLD_NAMES
    snr_range = cfg.SNR_RANGE
    max_polyphony = cfg.MAX_POLYPHONY
    mixtures = []

    # iterate over fold names
    for ifold, fold in enumerate(fold_names):
        
        fold_event_filenames, fold_event_durs = get_fold_files_and_durs(
                fold, 
                event_filenames, 
                event_durs)
        n_mixtures_per_fold = cfg.N_MIX_PER_FOLD[ifold]

        # iterate over rooms
        fold_rooms = cfg.FOLD_ROOMS[fold]
        room_mixtures=[]
        for room_name in fold_rooms:
            fold_mixture = {'mixture': []}
            fold_mixture['roomidx'] = room_name

            # get the room's relevant info
            path_to_room_files = get_path_to_room_files(room_name)
            room_trajs = get_room_trajectories(path_to_room_files)
            n_traj = len(room_trajs)
            traj_doas = get_traj_doas(room_trajs, n_traj)
            sample_counter = 0

            # generate each mixture
            for nmix in range(n_mixtures_per_fold):
                print('Room {}, generating mixture {}'.format(room_name, nmix+1))

                nth_mixture = initialize_mixture_metadata(room_name, snr_range)

                nth_mixture = populate_mixture(cfg, sample_counter, fold_event_durs, fold_event_filenames, n_traj, traj_doas, max_polyphony, nth_mixture)

                print(nth_mixture['event_onoffsets'])
                print(nth_mixture['sample_onoffsets'])
                input()
                fold_mixture['mixture'].append(nth_mixture)
            room_mixtures.append(fold_mixture)
        mixtures.append(room_mixtures)


if __name__ == "__main__":
    main()
