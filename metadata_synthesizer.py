import numpy as np
from utils import cart2sph
import os
import csv

class MetadataSynthesizer(object):
    def __init__(
            self, db_config, params, scenario_name
            ):
        self._db_config = db_config
        self._db = db_config._db_name
        self._metadata_path = params['mixturepath'] + '/' + 'metadata'
        self._classnames = db_config._classes
        self._active_classes = np.sort(params['active_classes'])
        self._nb_active_classes = len(self._active_classes)
        self._class2activeClassmap = []
        for cl in range(len(self._db_config._classes)):
            if cl in self._active_classes:
                self._class2activeClassmap.append(cl)
            else:
                self._class2activeClassmap.append(0)
        
        self._class_mobility = db_config._class_mobility
        self._mixture_setup = {}
        self._mixture_setup['scenario'] = scenario_name
        self._mixture_setup['nb_folds'] = db_config._nb_folds
        self._mixture_setup['rooms2folds'] = params['rooms2fold']
        self._mixture_setup['classnames'] = []
        for cl in self._classnames:
            self._mixture_setup['classnames'].append(cl)
        self._mixture_setup['nb_classes'] = len(self._active_classes)
        self._mixture_setup['fs_mix'] = 24000 #fs of RIRs
        self._mixture_setup['mixture_duration'] = params['mixture_duration']
        self._nb_mixtures_per_fold = params['nb_mixtures_per_fold']
        self._nb_mixtures = self._mixture_setup['nb_folds'] * self._nb_mixtures_per_fold if np.isscalar(self._nb_mixtures_per_fold) else np.sum(self._nb_mixtures_per_fold)
        self._mixture_setup['total_duration'] = self._nb_mixtures * self._mixture_setup['mixture_duration']
        self._mixture_setup['speed_set'] =  [10., 20., 40.]
        self._mixture_setup['snr_set'] = np.arange(6.,31.)
        self._mixture_setup['time_idx_100ms'] = np.arange(0.,self._mixture_setup['mixture_duration'],0.1)
        self._mixture_setup['nOverlap'] = params['max_polyphony']
        self._nb_frames = len(self._mixture_setup['time_idx_100ms'])
        self._rnd_generator = np.random.default_rng()
        
        self._rirdata = db_config._rirdata
        self._nb_classes = len(self._classnames)
        self._nb_speeds = len(self._mixture_setup['speed_set'])
        self._nb_snrs = len(self._mixture_setup['snr_set'])
        self._total_event_time_per_layer = params['event_time_per_layer']
        self._total_silence_time_per_layer = self._mixture_setup['mixture_duration'] - self._total_event_time_per_layer
        self._min_gap_len = 1. # in seconds, minimum length of gaps between samples
        self._trim_threshold = 3. #in seconds, minimum length under which a trimmed event at end is discarded
        self._move_threshold = 3. #in seconds, minimum length over which events can be moving

    def create_mixtures(self):
        self._mixtures = []
        foldlist = []

        for nfold in range(self._mixture_setup['nb_folds']):
            print('Generating metadata for fold {}'.format(str(nfold+1)))
            
            foldlist_nff = {}
            rooms_nf = self._mixture_setup['rooms2folds'][nfold]
            nb_rooms_nf = len(rooms_nf)
            
            idx_active = np.array([])
            for na in range(self._nb_active_classes):
                idx_active = np.append(idx_active, np.nonzero(self._db_config._samplelist[nfold]['class'] == self._active_classes[na]))
            idx_active = idx_active.astype('int')

            foldlist_nff['class'] = self._db_config._samplelist[nfold]['class'][idx_active]
            foldlist_nff['audiofile'] = self._db_config._samplelist[nfold]['audiofile'][idx_active]
            foldlist_nff['duration'] = self._db_config._samplelist[nfold]['duration'][idx_active]
            foldlist_nff['onoffset'] = self._db_config._samplelist[nfold]['onoffset'][idx_active]
            nb_samples_nf = len(foldlist_nff['duration'])
            
            # shuffle randomly the samples in the target list to avoid samples of the same class coming consecutively

            if len(np.shape(foldlist_nff['onoffset'])) == 1:
                foldlist_nff['onoffset'] = np.expand_dims(foldlist_nff['onoffset'],axis=1)
            foldlist_nf = foldlist_nff
            foldlist.append(foldlist_nf)
            sampleperm = self._rnd_generator.permutation(nb_samples_nf)
            foldlist_nf['class'] = foldlist_nf['class'][sampleperm]
            foldlist_nf['audiofile'] = foldlist_nf['audiofile'][sampleperm]
            foldlist_nf['duration'] = foldlist_nf['duration'][sampleperm]
            foldlist_nf['onoffset'] = foldlist_nf['onoffset'][sampleperm]
            room_mixtures = []
            for nr in rooms_nf:
                fold_mixture = {'mixture': []}
                fold_mixture['roomidx'] = rooms_nf
                nroom = nr
                print('Room {} \n'.format(nroom)) 
                n_traj = len(self._rirdata[nroom]['doa_xyz'])
                traj_doas = []
                
                for ntraj in self._rirdata[nroom]['doa_xyz']:
                    n_rirs = np.sum([len(subtr) for subtr in ntraj])
                    n_heights = len(ntraj)
                    all_doas = np.zeros((n_rirs, 3))
                    n_rirs_accum = 0
                    flip = 0
                    
                    for nheight in range(n_heights):
                        n_rirs_nh = len(ntraj[nheight])
                        doa_xyz = ntraj[nheight]
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
                        
                    traj_doas.append(all_doas)
            
                # start layering the mixtures for the specific room
                sample_counter = 0
                if np.isscalar(self._nb_mixtures_per_fold):
                    nb_mixtures_per_fold_per_room = int(np.round(self._nb_mixtures_per_fold / float(nb_rooms_nf)))
                else:
                    nb_mixtures_per_fold_per_room = int(np.round(self._nb_mixtures_per_fold[nfold] / float(nb_rooms_nf)))
                
                for nmix in range(nb_mixtures_per_fold_per_room):
                    print('Room {}, mixture {}'.format(nroom, nmix+1))

                    event_counter = 0
                    nth_mixture = {'files': np.array([]), 'class': np.array([]), 'event_onoffsets': np.array([]),
                                   'sample_onoffsets': np.array([]), 'trajectory': np.array([]), 'isMoving': np.array([]), 'isFlippedMoving': np.array([]),
                                   'speed': np.array([]), 'rirs': [], 'doa_azel': np.array([],dtype=object)}
                    nth_mixture['room'] = nroom
                    nth_mixture['snr'] = self._mixture_setup['snr_set'][self._rnd_generator.integers(0,self._nb_snrs)]
                    
                    for layer in range(self._mixture_setup['nOverlap']):
                        print('Layer {}'.format(layer))                        
                        #zero this flag (explained later)
                        TRIMMED_SAMPLE_AT_END = 0
                        
                        #fetch event samples till they add up to the target event time per layer
                        event_time_in_layer = 0
                        event_idx_in_layer = []
                        
                        while event_time_in_layer < self._total_event_time_per_layer:
                            #get event duration
                            ev_duration = np.ceil(foldlist_nf['duration'][sample_counter]*10.)/10.
                            event_time_in_layer += ev_duration
                            event_idx_in_layer.append(sample_counter)
                            
                            event_counter += 1
                            sample_counter += 1
  
                            if sample_counter == nb_samples_nf:
                                sample_counter = 0

                        # the last sample is going to be trimmed to fit the desired
                        # time, or omit if it is less than X sec, and occurs later than that time
                        trimmed_event_length = self._total_event_time_per_layer - (event_time_in_layer - ev_duration)
                        #Temporary workaround - for some reason for interference classes the dict is packed with an additional dimension - check it
                        if len(foldlist_nf['onoffset'][event_idx_in_layer[-1]]) == 1:
                            ons = foldlist_nf['onoffset'][event_idx_in_layer[-1]][0][0,0] if self._db == 'nigens' else foldlist_nf['onoffset'][event_idx_in_layer[-1]][0][0]
                        else:
                            ons = foldlist_nf['onoffset'][event_idx_in_layer[-1]][0,0] if self._db == 'nigens' else foldlist_nf['onoffset'][event_idx_in_layer[-1]][0]
                        if (trimmed_event_length > self._trim_threshold) and (trimmed_event_length > np.floor(ons*10.)/10.):
                            TRIMMED_SAMPLE_AT_END = 1
                        else:
                            if len(event_idx_in_layer) == 1:
                                raise ValueError("STOP, we will get stuck here forever")

                            #remove from sample list
                            event_idx_in_layer = event_idx_in_layer[:-1]
                            # reduce sample count and events-in-recording by 1
                            event_counter -= 1
                            if sample_counter != 0:
                                sample_counter -= 1
                            else:
                                # move sample counter to end of the list to re-use sample
                                sample_counter = nb_samples_nf-1
                            
                        nb_samples_in_layer = len(event_idx_in_layer)
                        # split silences between events
                        # randomize N split points uniformly for N events (in
                        # steps of 100msec)
                        mult_silence = np.round(self._total_silence_time_per_layer*10.)
                        
                        mult_min_gap_len = np.round(self._min_gap_len*10.)
                        if nb_samples_in_layer > 1:
                            
                            silence_splits = np.sort(self._rnd_generator.integers(1, mult_silence,nb_samples_in_layer-1))
                            #force gaps smaller then _min_gap_len to it
                            gaps = np.diff(np.concatenate(([0],silence_splits,[mult_silence])))
                            smallgaps_idx = np.argwhere(gaps[:(nb_samples_in_layer-1)] < mult_min_gap_len)
                            while np.any(smallgaps_idx):
                                temp = np.concatenate(([0], silence_splits))
                                silence_splits[smallgaps_idx] = temp[smallgaps_idx] + mult_min_gap_len
                                gaps = np.diff(np.concatenate(([0],silence_splits,[mult_silence])))
                                smallgaps_idx = np.argwhere(gaps[:(nb_samples_in_layer-1)] < mult_min_gap_len)
                            if np.any(gaps < mult_min_gap_len):
                                min_idx = np.argwhere(gaps < mult_min_gap_len)
                                gaps[min_idx] = mult_min_gap_len
                            # if gaps[nb_samples_in_layer-1] < mult_min_gap_len:
                            #     gaps[nb_samples_in_layer-1] = mult_min_gap_len
                            
                        else:
                            gaps = np.array([mult_silence])
                        
                        while np.sum(gaps) > self._total_silence_time_per_layer*10.:
                            silence_diff = np.sum(gaps) - self._total_silence_time_per_layer*10.
                            picked_gaps = np.argwhere(gaps > np.mean(gaps))
                            eq_subtract = silence_diff / len(picked_gaps)
                            picked_gaps = np.argwhere((gaps - eq_subtract) > mult_min_gap_len)
                            gaps[picked_gaps] -= eq_subtract
                            
                        # distribute events in timeline
                        time_idx = 0
                        for nl in range(nb_samples_in_layer):
                            #print('Sample {} in layer {}'.format(nl, layer))
                            # event offset (quantized to 100ms)
                            gap_nl = gaps[nl]
                            time_idx += gap_nl
                            event_nl = event_idx_in_layer[nl]
                            event_duration_nl = np.ceil(foldlist_nf['duration'][event_nl]*10.)
                            event_class_nl = int(foldlist_nf['class'][event_nl])
                            if len(foldlist_nf['onoffset'][event_nl]) == 1:
                                onoffsets = foldlist_nf['onoffset'][event_nl][0]
                            else:
                                onoffsets = foldlist_nf['onoffset'][event_nl]
                                
                            sample_onoffsets = np.zeros_like(onoffsets)
                            if self._db == 'nigens':
                                sample_onoffsets[:, 0] = np.floor(onoffsets[:,0]*10.)/10.
                                sample_onoffsets[:, 1] = np.floor(onoffsets[:,1]*10.)/10.
                                #trim event duration if it's the trimmed sample
                                if (nl == nb_samples_in_layer-1) and TRIMMED_SAMPLE_AT_END:
                                    event_duration_nl = len(self._mixture_setup['time_idx_100ms']) - time_idx - 1
                                    # keep only onset/offsets in the trimmed region
                                    find_last_offset_mtx = (event_duration_nl/10.) > sample_onoffsets
                                    sample_onoffsets = sample_onoffsets[:np.sum(find_last_offset_mtx[:,0]),:]
                                    if sample_onoffsets[-1, 1] > event_duration_nl/10.:
                                        sample_onoffsets[-1, 1] = event_duration_nl/10.
                            else:
                                sample_onoffsets = np.floor(onoffsets*10.)/10.
                                #trim event duration if it's the trimmed sample
                                if (nl == nb_samples_in_layer-1) and TRIMMED_SAMPLE_AT_END:
                                    event_duration_nl = len(self._mixture_setup['time_idx_100ms']) - time_idx - 1
                                    # keep only onset/offsets in the trimmed region
                                    if sample_onoffsets[1] > event_duration_nl/10.:
                                        sample_onoffsets[1] = event_duration_nl/10.
                                    
                            # trajectory
                            ev_traj = self._rnd_generator.integers(0, n_traj)
                            nRirs = np.sum([len(subtr) for subtr in self._rirdata[nroom]['doa_xyz'][ev_traj]])
                            
                            #if event is less than move_threshold long, make it static by default
                            if event_duration_nl <= self._move_threshold*10:
                                is_moving = 0 
                            else:
                                if self._class_mobility[event_class_nl] == 2:
                                    # randomly moving or static
                                    is_moving = self._rnd_generator.integers(0,2)
                                else:
                                    # only static or moving depending on class
                                    is_moving = self._class_mobility[event_class_nl]
                                                    
                            if is_moving:
                                ev_nspeed = self._rnd_generator.integers(0,self._nb_speeds)
                                ev_speed = self._mixture_setup['speed_set'][ev_nspeed]
                                # check if with the current speed there are enough
                                # RIRs in the trajectory to move through the full
                                # duration of the event, otherwise, lower speed
                                while len(np.arange(0,nRirs,ev_speed/10)) <= event_duration_nl:
                                    ev_nspeed = ev_nspeed-1
                                    if ev_nspeed == -1:
                                        break

                                    ev_speed = self._mixture_setup['speed_set'][ev_nspeed]
                                
                                is_flipped_moving = self._rnd_generator.integers(0,2)
                                event_span_nl = event_duration_nl * ev_speed / 10.
                                    
                                if is_flipped_moving:
                                    # sample length is shorter than all the RIRs
                                    # in the moving trajectory
                                    if ev_nspeed+1:
                                        end_idx = event_span_nl + self._rnd_generator.integers(0, nRirs-event_span_nl+1)
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
                                        start_idx = self._rnd_generator.integers(0, nRirs-event_span_nl+1)
                                        riridx = start_idx + np.arange(0,event_span_nl,dtype=int) - 1
                                        riridx = riridx[np.arange(0,len(riridx),ev_speed/10,dtype=int)]
                                    else:
                                        riridx = np.arange(0,event_span_nl)
                                        riridx = riridx[np.arange(0,len(riridx),ev_speed/10,dtype=int)]
                                        riridx[riridx>nRirs-1] = nRirs-1
                            else:
                                is_flipped_moving = 0
                                ev_speed = 0
                                riridx = np.array([self._rnd_generator.integers(0,nRirs)])
                            riridx = riridx.astype('int')

                            if nl == 0 and layer==0:
                                nth_mixture['event_onoffsets'] = np.array([[time_idx/10., (time_idx+event_duration_nl)/10.]])
                                nth_mixture['doa_azel'] = [cart2sph(traj_doas[ev_traj][riridx,:])]
                                nth_mixture['sample_onoffsets'] = [sample_onoffsets]
                            else:
                                nth_mixture['event_onoffsets'] = np.vstack((nth_mixture['event_onoffsets'], np.array([time_idx/10., (time_idx+event_duration_nl)/10.])))
                                nth_mixture['doa_azel'].append(cart2sph(traj_doas[ev_traj][riridx,:]))
                                nth_mixture['sample_onoffsets'].append(sample_onoffsets)
                                         
                            nth_mixture['files'] = np.append(nth_mixture['files'], foldlist_nf['audiofile'][event_nl])
                            nth_mixture['class'] = np.append(nth_mixture['class'], self._class2activeClassmap[int(foldlist_nf['class'][event_nl])])
                            nth_mixture['trajectory'] = np.append(nth_mixture['trajectory'], ev_traj)
                            nth_mixture['isMoving'] = np.append(nth_mixture['isMoving'], is_moving)
                            nth_mixture['isFlippedMoving'] = np.append(nth_mixture['isFlippedMoving'], is_flipped_moving)
                            nth_mixture['speed'] = np.append(nth_mixture['speed'], ev_speed)
                            nth_mixture['rirs'].append(riridx)

                            
                            time_idx += event_duration_nl
                        
                        # sort overlapped events by temporal appearance
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
                
                    #accumulate mixtures for each room
                    fold_mixture['mixture'].append(nth_mixture)
                #accumulate rooms
                room_mixtures.append(fold_mixture)
            #accumulate mixtures per fold
            self._mixtures.append(room_mixtures)
            

        return self._mixtures, self._mixture_setup, foldlist
    
    def prepare_metadata_and_stats(self):
        print('Calculate statistics and prepate metadata')
        stats = []
        self._metadata = []
        stats = {}
        stats['nFrames_total'] = self._mixture_setup['nb_folds'] * self._nb_mixtures_per_fold * self._nb_frames if np.isscalar(self._nb_mixtures_per_fold) else np.sum(self._nb_mixtures_per_fold) * self._nb_frames
        stats['class_multi_instance'] = np.zeros(self._nb_classes)
        stats['class_instances'] = np.zeros(self._nb_classes)
        stats['class_nEvents'] = np.zeros(self._nb_classes)
        stats['class_presence'] = np.zeros(self._nb_classes)
        
        stats['polyphony'] = np.zeros(self._mixture_setup['nOverlap']+1)
        stats['event_presence'] = 0
        stats['nEvents_total'] = 0
        stats['nEvents_static'] = 0
        stats['nEvents_moving'] = 0
        
        for nfold in range(self._mixture_setup['nb_folds']):
            print('Statistics and metadata for fold {}'.format(nfold+1))
            rooms = self._mixtures[nfold][0]['roomidx']
            nb_rooms = len(rooms)
            room_mixtures=[]
            for nr in range(nb_rooms):
                nb_mixtures = len(self._mixtures[nfold][nr]['mixture'])
                per_room_mixtures = []
                for nmix in range(nb_mixtures):
                    mixture = {'classid': np.array([]), 'trackid': np.array([]), 'eventtimetracks': np.array([]), 'eventdoatimetracks': np.array([])}
                    mixture_nm = self._mixtures[nfold][nr]['mixture'][nmix]
                    event_classes = mixture_nm['class']
                    event_states = mixture_nm['isMoving']
                    
                    #idx of events and interferers
                    nb_events = len(event_classes)
                    nb_events_moving = np.sum(event_states)
                    stats['nEvents_total'] += nb_events
                    stats['nEvents_static'] += nb_events - nb_events_moving
                    stats['nEvents_moving'] += nb_events_moving

                    # number of events per class
                    for nc in range(self._mixture_setup['nb_classes']):
                        nb_class_events = np.sum(event_classes == nc)
                        stats['class_nEvents'][nc] += nb_class_events
                    
                    # store a timeline for each event
                    eventtimetracks = np.zeros((self._nb_frames, nb_events))
                    eventdoatimetracks = np.nan*np.ones((self._nb_frames, 2, nb_events))

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
                                eventdoatimetracks[ev_idx[activity_mask.astype(bool)],0,nev] = np.ones(np.sum(activity_mask==1))*doa_azel[0,0]
                                eventdoatimetracks[ev_idx[activity_mask.astype(bool)],1,nev] = np.ones(np.sum(activity_mask==1))*doa_azel[0,1]
                            except IndexError:
                                 excess_idx = len(np.argwhere(ev_idx >= self._nb_frames))
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
                                excess_idx = len(np.argwhere(ev_idx >= self._nb_frames))
                                ev_idx = ev_idx[:-excess_idx]
                                if len(activity_mask) > len(ev_idx):
                                    activity_mask = activity_mask[0:len(ev_idx)]
                                eventtimetracks[ev_idx,nev] = activity_mask
                                eventdoatimetracks[ev_idx[activity_mask.astype(bool)],:,nev] = doa_azel[activity_mask.astype(bool),:]

                    mixture['classid'] = event_classes
                    mixture['trackid'] = np.arange(0,nb_events)
                    mixture['eventtimetracks'] = eventtimetracks
                    mixture['eventdoatimetracks'] = eventdoatimetracks
                    
                    for nf in range(self._nb_frames):
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
                            
                            for nc in range(self._mixture_setup['nb_classes']):
                                nb_instances = np.sum(active_classes == nc)
                                if nb_instances > 0:
                                    stats['class_presence'][nc] += 1
                                if nb_instances > 1:
                                    stats['class_multi_instance'][nc] += 1
                                stats['class_instances'][nc] += nb_instances
                    per_room_mixtures.append(mixture)
                room_mixtures.append(per_room_mixtures)
            self._metadata.append(room_mixtures)
         
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
                                
                                
                                
                            
        
        
