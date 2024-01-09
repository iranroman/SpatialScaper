from room_scaper.utils.parser import parse_args, load_config
from room_scaper.data.utils import get_path_to_room_files
from room_scaper.prepare_fsd50k import prepare_fsd50k
import yaml
import pickle
import os
import numpy as np
import librosa
import random

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

def get_filename_class_duration(fold_event_filenames, path_to_dataset, mixture_dur, class_dict=None):

    filename,filepath = random.choice(list(fold_event_filenames.items()))
    #filepath = os.path.join(path_to_dataset, filename.split('/')[-1]) # HACKY, BAD, Will improve with Adrian's contrib
    classid = filename.split('/')[0] # HACKY, BAD, Will improve with Adrian's contrib
    classid = class_dict[classid]
    filedur = librosa.get_duration(path=filepath)
    if filedur >= mixture_dur/4:
        filedur = mixture_dur//4 -1
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
    print(0, max_time, dt)
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
    nth_mixture['files'] = np.array([event['filepath'] for event in all_events_meta])
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

def prepare_metadata_and_stats(mixtures,
        fold_names,
        fold_rooms,
        class_dict,
        nb_frames,
        nb_mixtures_per_fold,
        max_polyphony,
    ):
    print('Calculate statistics and prepate metadata')
    #stats = []
    metadata = []
    stats = {}
    stats['nFrames_total'] = len(fold_names) * nb_mixtures_per_fold * nb_frames if np.isscalar(nb_mixtures_per_fold) else np.sum(nb_mixtures_per_fold) * nb_frames
    stats['class_multi_instance'] = np.zeros(len(class_dict))
    stats['class_instances'] = np.zeros(len(class_dict))
    stats['class_nEvents'] = np.zeros(len(class_dict))
    stats['class_presence'] = np.zeros(len(class_dict))
    
    stats['polyphony'] = np.zeros(max_polyphony+1)
    stats['event_presence'] = 0
    stats['nEvents_total'] = 0
    stats['nEvents_static'] = 0
    stats['nEvents_moving'] = 0
    
    # iterate over fold names
    for nfold, fold in enumerate(fold_names):
        print('Statistics and metadata for fold {}'.format(fold))
        rooms = fold_rooms[fold]
        nb_rooms = len(rooms)
        room_mixtures=[]
        for nr in range(nb_rooms):
            nb_mixtures = len(mixtures[nfold][nr]['mixture'])
            per_room_mixtures = []
            for nmix in range(nb_mixtures):
                mixture = {'classid': np.array([]), 'trackid': np.array([]), 'eventtimetracks': np.array([]), 'eventdoatimetracks': np.array([])}
                mixture_nm = mixtures[nfold][nr]['mixture'][nmix]
                event_classes = mixture_nm['class']
                event_states = mixture_nm['isMoving']
                
                #idx of events and interferers
                nb_events = len(event_classes)
                nb_events_moving = np.sum(event_states)
                stats['nEvents_total'] += nb_events
                stats['nEvents_static'] += nb_events - nb_events_moving
                stats['nEvents_moving'] += nb_events_moving

                # number of events per class
                for nc in range(len(class_dict)):
                    nb_class_events = np.sum(event_classes == nc)
                    stats['class_nEvents'][nc] += nb_class_events
                
                # store a timeline for each event
                eventtimetracks = np.zeros((nb_frames, nb_events))
                eventdoatimetracks = np.nan*np.ones((nb_frames, 2, nb_events))
                print(eventtimetracks.shape)
                print(eventdoatimetracks.shape)
                input()

                #prepare metadata for synthesis
                for nev in range(nb_events):
                    event_onoffset = mixture_nm['event_onoffsets'][nev,:]*10
                    doa_azel = np.round(mixture_nm['doa_azel'][nev])
                    #zero the activity according to perceptual onsets/offsets
                    sample_onoffsets = mixture_nm['sample_onoffsets'][nev]
                    ev_idx = np.arange(event_onoffset[0], event_onoffset[1]+0.1,dtype=int)
                    activity_mask = np.zeros(len(ev_idx),dtype=int)
                    sample_shape = np.shape(sample_onoffsets)
                    if len(sample_shape) == 1:
                        activity_mask[np.arange(int(np.round(sample_onoffsets[0]*10)),int(np.round(sample_onoffsets[1]*10)))] = 1
                    else:
                        for nseg in range(sample_shape[0]):
                            ran = np.arange(int(np.round(sample_onoffsets[nseg,0]*10)),int(np.round((sample_onoffsets[nseg,1])*10)))
                            activity_mask[ran] = 1
                    
                    if len(activity_mask) > len(ev_idx):
                        activity_mask = activity_mask[0:len(ev_idx)]

                    if np.shape(doa_azel)[0] == 1:
                        # static event
                        try:
                            eventtimetracks[ev_idx, nev] = activity_mask
                            print(ev_idx)
                            print(ev_idx.shape)
                            print(activity_mask.astype(bool))
                            print(activity_mask.astype(bool).shape)
                            print(nev)
                            print(activity_mask)
                            print(np.sum(activity_mask==1))
                            print(np.sum(activity_mask.astype(bool)))
                            print(activity_mask.shape)
                            print(doa_azel.shape)
                            print(eventdoatimetracks[ev_idx[activity_mask.astype(bool)],0,nev].shape) 
                            print((np.ones(np.sum(activity_mask==1))*doa_azel[0,0]).shape)
                            input()
                            eventdoatimetracks[ev_idx[activity_mask.astype(bool)],0,nev] = np.ones(np.sum(activity_mask==1))*doa_azel[0,0]
                            eventdoatimetracks[ev_idx[activity_mask.astype(bool)],1,nev] = np.ones(np.sum(activity_mask==1))*doa_azel[0,1]
                        except IndexError:
                             excess_idx = len(np.argwhere(ev_idx >= nb_frames))
                             ev_idx = ev_idx[:-excess_idx]
                             if len(activity_mask) > len(ev_idx):
                                 activity_mask = activity_mask[0:len(ev_idx)]
                             eventtimetracks[ev_idx, nev] = activity_mask
                             eventdoatimetracks[ev_idx[activity_mask.astype(bool)],0,nev] = np.ones(np.sum(activity_mask==1))*doa_azel[0,0]
                             eventdoatimetracks[ev_idx[activity_mask.astype(bool)],1,nev] = np.ones(np.sum(activity_mask==1))*doa_azel[0,1]

                    else:
                        # moving event
                        nb_doas = np.shape(doa_azel)[0]
                        ev_idx = ev_idx[:nb_doas]
                        activity_mask = activity_mask[:nb_doas]
                        try:
                            eventtimetracks[ev_idx,nev] = activity_mask
                            eventdoatimetracks[ev_idx[activity_mask.astype(bool)],:,nev] = doa_azel[activity_mask.astype(bool),:]
                        except IndexError:
                            excess_idx = len(np.argwhere(ev_idx >= nb_frames))
                            ev_idx = ev_idx[:-excess_idx]
                            if len(activity_mask) > len(ev_idx):
                                activity_mask = activity_mask[0:len(ev_idx)]
                            eventtimetracks[ev_idx,nev] = activity_mask
                            eventdoatimetracks[ev_idx[activity_mask.astype(bool)],:,nev] = doa_azel[activity_mask.astype(bool),:]

                mixture['classid'] = event_classes
                mixture['trackid'] = np.arange(0,nb_events)
                mixture['eventtimetracks'] = eventtimetracks
                mixture['eventdoatimetracks'] = eventdoatimetracks
                
                for nf in range(nb_frames):
                    # find active events
                    active_events = np.argwhere(eventtimetracks[nf,:] > 0)
                    # find the classes of the active events
                    active_classes = event_classes[active_events]
                    
                    if not active_classes.ndim and active_classes.size:
                        # add to zero polyphony
                        stats['polyphony'][0] += 1
                    else:
                        # add to general event presence
                        stats['event_presence'] += 1
                        # number of simultaneous events
                        nb_active = len(active_events)

                        # add to respective polyphony
                        try:
                            stats['polyphony'][nb_active] += 1
                        except IndexError:
                            pass #TODO: this is a workaround for less than 1% border cases, needs to be fixed although not very relevant
                        
                        # presence, instances and multi-instance for each class
                        
                        for nc in range(len(class_dict)):
                            nb_instances = np.sum(active_classes == nc)
                            if nb_instances > 0:
                                stats['class_presence'][nc] += 1
                            if nb_instances > 1:
                                stats['class_multi_instance'][nc] += 1
                            stats['class_instances'][nc] += nb_instances
                per_room_mixtures.append(mixture)
            room_mixtures.append(per_room_mixtures)
        metadata.append(room_mixtures)
    print('here!')
    input()
     
    # compute average polyphony
    weighted_polyphony_sum = 0
    for nn in range(self._mixture_setup['nOverlap']):
        weighted_polyphony_sum += nn * stats['polyphony'][nn+1]
    
    stats['avg_polyphony'] = weighted_polyphony_sum / stats['event_presence']
    
    #event percentages
    stats['class_event_pc'] = np.round(stats['class_nEvents']*1000./stats['nEvents_total'])/10.
    stats['event_presence_pc'] = np.round(stats['event_presence']*1000./stats['nFrames_total'])/10.
    stats['class_presence_pc'] = np.round(stats['class_presence']*1000./stats['nFrames_total'])/10.
    # percentage of frames with same-class instances
    stats['multi_class_pc'] = np.round(np.sum(stats['class_multi_instance']*1000./stats['nFrames_total']))/10.


    return self._metadata, stats

def write_metadata(self):
    if not os.path.isdir(self._metadata_path):
        os.makedirs(self._metadata_path)
    
    for nfold in range(self._mixture_setup['nb_folds']):
        print('Writing metadata files for fold {}'.format(nfold+1))
        nb_rooms = len(self._metadata[nfold])
        for nr in range(nb_rooms):
            nb_mixtures = len(self._metadata[nfold][nr])
            for nmix in range(nb_mixtures):
                print('Mixture {}'.format(nmix))
                metadata_nm = self._metadata[nfold][nr][nmix]
                
                # write to filename, omitting non-active frames
                mixture_filename = 'fold{}_room{}_mix{:03}.csv'.format(nfold+1, nr+1, nmix+1)
                file_id = open(self._metadata_path + '/' + mixture_filename, 'w', newline="")
                metadata_writer = csv.writer(file_id,delimiter=',',quoting = csv.QUOTE_NONE)
                for nf in range(self._nb_frames):
                    # find active events
                    active_events = np.argwhere(metadata_nm['eventtimetracks'][nf, :]>0)
                    nb_active = len(active_events)
                    
                    if nb_active > 0:
                        # find the classes of active events
                        active_classes = metadata_nm['classid'][active_events]
                        active_tracks = metadata_nm['trackid'][active_events]
                        
                        # write to file
                        for na in range(nb_active):
                            classidx = int(active_classes[na][0]) #additional zero index since it's packed in an array
                            trackidx = int(active_tracks[na][0])
                            
                            azim = int(metadata_nm['eventdoatimetracks'][nf,0,active_events][na][0])
                            elev = int(metadata_nm['eventdoatimetracks'][nf,1,active_events][na][0])
                            metadata_writer.writerow([nf,classidx,trackidx,azim,elev])
                file_id.close()

def DCASE_main():

    # 0. parse the yaml with the config. Define if you are generating data in 
    # the "training" or "testing" fold then, for each audio mixture you generate:
    # parse config arguments
    args = parse_args()
    cfg = load_config(args, args.path_to_config)
    print('generating data using parameters:')
    print(yaml.dump(dict(cfg), allow_unicode=True, default_flow_style=False))

    
    event_filenames = prepare_fsd50k()

    fold_names = cfg.FOLD_NAMES
    snr_range = cfg.SNR_RANGE
    max_polyphony = cfg.MAX_POLYPHONY
    mixtures = []

    # iterate over fold names
    for ifold, fold in enumerate(fold_names):
        
        fold_event_filenames = event_filenames.get_filenames(fold)
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

            fold_mixture = {'mixture': []}
            fold_mixture['roomidx'] = room_name

            # get the room's relevant info
            path_to_room_files = get_path_to_room_files(room_name)
            room_trajs = get_room_trajectories(path_to_room_files)
            n_traj = len(room_trajs) 
            traj_doas = get_traj_doas(room_trajs, n_traj) # TO BE SUBSTITUTED 

            # 2. add sound events to the mixture by determininig:
            for imixture in range(n_mixtures_per_fold//len(fold_rooms)):

                # this can also be probabilistic
                nevents = cfg.NUM_EVENTS_IN_MIXTURE

                track_events = [] # to ensure MAX_POLYPHONY
                all_events_meta = []
                for ievent in range(nevents):
                    # a. determine the label and the specific source file. The source time will
                    # be from 0 to its total duration. Map the label to the DCASE index
                    source_file_metadata = get_filename_class_duration(fold_event_filenames, cfg.PATH_TO_DATASET, cfg.MIXTURE_DUR, cfg.CLASS_DICT)

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
				#accumulate mixtures for each room
                fold_mixture['mixture'].append(nth_mixture)
            #accumulate rooms
            room_mixtures.append(fold_mixture)
        #accumulate mixtures per fold
        mixtures.append(room_mixtures)
        

    prepare_metadata_and_stats(mixtures,fold_names, cfg.FOLD_ROOMS, cfg.CLASS_DICT, cfg.METADATA_SR * cfg.MIXTURE_DUR, cfg.N_MIX_PER_FOLD, cfg.MAX_POLYPHONY)

    # 3. determine whether the mixture will undergo "channel_swap" augmentations
    # (channel_swap only applies to RIRs from real rooms) 

    # 4. dump resulting file with annotations



if __name__ == "__main__":
    DCASE_main()
