import numpy as np
import scipy.io
import utils
import os
import mat73
import scipy.signal as signal
import soundfile
import pickle

import librosa # used since we allow .wav and .mp3 loading 

class AudioSynthesizer(object):
    def __init__(
            self, params, mixtures, mixture_setup, db_config, audio_format
            ):
        self._mixtures = mixtures
        self._rirpath = params['rirpath']
        self._db_path = params['db_path']
        self._audio_format = audio_format
        self._outpath = params['mixturepath'] + '/' + mixture_setup['scenario'] + '/' + self._audio_format
        self._rirdata = db_config._rirdata
        self._nb_rooms = len(self._rirdata)
        self._room_names = list(self._rirdata.keys())
        self._classnames = mixture_setup['classnames']
        self._fs_mix = mixture_setup['fs_mix']
        self._t_mix = mixture_setup['mixture_duration']
        self._l_mix = int(np.round(self._fs_mix * self._t_mix))
        self._time_idx100 = np.arange(0., self._t_mix, 0.1)
        self._stft_winsize_moving = 0.1*self._fs_mix//2
        self._nb_folds = len(mixtures)
        self._apply_event_gains = db_config._apply_class_gains
        self._db_name = params['db_name']
        self._fs = params['fs']
        if self._apply_event_gains:
            self._class_gains = db_config._class_gains
        
        
    def synthesize_mixtures(self):
        rirdata2room_idx = {'bomb_shelter':1, 'gym':2, 'pb132':3, 'pc226':4, 'sa203':5, 'sc203':6, 'se203':8, 'tb103':9, 'tc352':10}
        # create path if doesn't exist
        if not os.path.isdir(self._outpath):
            os.makedirs(self._outpath)
        
        for nfold in range(self._nb_folds):
            print('Generating scene audio for fold {}'.format(nfold+1))

            rooms = self._mixtures[nfold][0]['roomidx']
            nb_rooms_in_fold = len(rooms)
            for nr in range(nb_rooms_in_fold):

                nroom = rooms[nr]
                nb_mixtures = len(self._mixtures[nfold][nr]['mixture'])
                print('Loading RIRs for room {}'.format(nroom))
                
                room_idx = rirdata2room_idx[nroom]
                if room_idx > 9:
                    struct_name = 'rirs_{}_{}'.format(room_idx,nroom)
                else:
                    struct_name = 'rirs_0{}_{}'.format(room_idx,nroom)
                path = self._rirpath + '/' + struct_name + '.pkl'
                with open(path,'rb') as f:
                    rirs = pickle.load(f)[self._audio_format]
                # stack all the RIRs for all heights to make one large trajectory
                print('Stacking same trajectory RIRs')
                lRir = len(rirs[0][0])
                nCh = len(rirs[0][0][0])
                
                n_traj = len(self._rirdata[nroom]['doa_xyz'])
                n_rirs_max = 0
                for trj in self._rirdata[nroom]['doa_xyz']:
                    n_rirs_ = np.sum([len(hei) for hei in trj])
                    if n_rirs_ > n_rirs_max:
                        n_rirs_max = n_rirs_
                channel_rirs = np.zeros((lRir, nCh, n_rirs_max, n_traj))
                for ntraj in range(n_traj):
                    nHeights = len(self._rirdata[nroom]['doa_xyz'][ntraj])
                    nRirs_accum = 0
                    
                    # flip the direction of each second height, so that a
                    # movement can jump from the lower to the higher smoothly and
                    # continue moving the opposite direction
                    flip = False
                    for nheight in range(nHeights):
                        nRirs_nh = len(self._rirdata[nroom]['doa_xyz'][ntraj][nheight])
                        rir_l = len(rirs[ntraj][nheight][0,0,:])
                        if flip:
                            channel_rirs[:, :, nRirs_accum + np.arange(0,nRirs_nh),ntraj] = rirs[ntraj][nheight][:,:,np.arange(rir_l-1,-1,-1)]
                        else:
                            channel_rirs[:, :, nRirs_accum + np.arange(0,nRirs_nh),ntraj] = rirs[ntraj][nheight]
                            
                        nRirs_accum += nRirs_nh
                        flip = not flip
                
                del rirs #clear some memory
                
                for nmix in range(nb_mixtures):
                    print('Writing mixture {}/{}'.format(nmix+1,nb_mixtures))

                    ### WRITE TARGETS EVENTS
                    mixture_nm = self._mixtures[nfold][nr]['mixture'][nmix]
                    try:
                        nb_events = len(mixture_nm['class'])
                    except TypeError:
                        nb_events = 1
                    
                    mixsig = np.zeros((self._l_mix, 4))
                    for nev in range(nb_events):
                        if not nb_events == 1:
                            classidx = int(mixture_nm['class'][nev])
                            onoffset = mixture_nm['event_onoffsets'][nev,:]
                            filename = mixture_nm['files'][nev]
                            ntraj = int(mixture_nm['trajectory'][nev])
                        
                        else:
                            classidx = int(mixture_nm['class'])
                            onoffset = mixture_nm['event_onoffsets']
                            filename = mixture_nm['files']
                            ntraj = int(mixture_nm['trajectory'])
                            
                        # load event audio and resample to match RIR sampling
                        
                        if self._db_name == 'nigens':
                            eventsig, fs_db = soundfile.read(self._db_path + '/' + filename)   
                        elif self._db_name == 'fsd50k':
                            eventsig, fs_db = librosa.load(filename, sr=self._fs, duration=onoffset[1]/10) # here we need librosa since we are loading .mp3 
                        else:
                            raise Exception(f"Unknown event database: {self._db_name}")
                            
                

                        
                        if len(np.shape(eventsig)) > 1:
                            eventsig = eventsig[:,0]
                        eventsig = signal.resample_poly(eventsig, self._fs_mix, fs_db)
                        
                        #spatialize audio
                        riridx = mixture_nm['rirs'][nev] if nb_events > 1 else mixture_nm['rirs']
                        
                        
                        moving_condition = mixture_nm['isMoving'][nev] if nb_events > 1 else mixture_nm['isMoving']
                        if nb_events > 1 and not moving_condition:
                            riridx = int(riridx[0]) if len(riridx)==1 else riridx.astype('int')
                        if nb_events == 1 and type(riridx) != int:
                            riridx = riridx[0]
                            
                        if moving_condition:
                            nRirs_moving = len(riridx) if np.shape(riridx) else 1
                            ir_times = self._time_idx100[np.arange(0,nRirs_moving)]
                            mixeventsig = 481.6989*utils.ctf_ltv_direct(eventsig, channel_rirs[:, :, riridx, ntraj], ir_times, self._fs_mix, self._stft_winsize_moving) / float(len(eventsig))
                        else:
                            mixeventsig0 = scipy.signal.convolve(eventsig, np.squeeze(channel_rirs[:, 0, riridx, ntraj]), mode='full', method='fft')
                            mixeventsig1 = scipy.signal.convolve(eventsig, np.squeeze(channel_rirs[:, 1, riridx, ntraj]), mode='full', method='fft')
                            mixeventsig2 = scipy.signal.convolve(eventsig, np.squeeze(channel_rirs[:, 2, riridx, ntraj]), mode='full', method='fft')
                            mixeventsig3 = scipy.signal.convolve(eventsig, np.squeeze(channel_rirs[:, 3, riridx, ntraj]), mode='full', method='fft')

                            mixeventsig = np.stack((mixeventsig0,mixeventsig1,mixeventsig2,mixeventsig3),axis=1)
                        if self._apply_event_gains:
                            # apply random gain to each event based on class gain, distribution given externally
                            K=1000
                            rand_energies_per_spec = utils.sample_from_quartiles(K, self._class_gains[classidx])
                            intr_quart_energies_per_sec = rand_energies_per_spec[K + np.arange(3*(K+1))]
                            rand_energy_per_spec = intr_quart_energies_per_sec[np.random.randint(len(intr_quart_energies_per_sec))]
                            sample_onoffsets = mixture_nm['sample_onoffsets'][nev]
                            sample_active_time = sample_onoffsets[1] - sample_onoffsets[0]
                            target_energy = rand_energy_per_spec*sample_active_time
                            if self._audio_format == 'mic':
                                event_omni_energy = np.sum(np.sum(mixeventsig,axis=1)**2)
                            elif self._audio_format == 'foa':
                                event_omni_energy = np.sum(mixeventsig[:,0]**2)
                                
                            norm_gain = np.sqrt(target_energy / event_omni_energy)
                            mixeventsig = norm_gain * mixeventsig

                        lMixeventsig = np.shape(mixeventsig)[0]
                        if np.round(onoffset[0]*self._fs_mix) + lMixeventsig <= self._t_mix * self._fs_mix:
                            mixsig[int(np.round(onoffset[0]*self._fs_mix)) + np.arange(0,lMixeventsig,dtype=int), :] += mixeventsig
                        else:
                            lMixeventsig_trunc = int(self._t_mix * self._fs_mix - int(np.round(onoffset[0]*self._fs_mix)))
                            mixsig[int(np.round(onoffset[0]*self._fs_mix)) + np.arange(0,lMixeventsig_trunc,dtype=int), :] += mixeventsig[np.arange(0,lMixeventsig_trunc,dtype=int), :]

                    # normalize
                    gnorm = 0.5/np.max(np.max(np.abs(mixsig)))

                    mixsig = gnorm*mixsig
                    mixture_filename = 'fold{}_room{}_mix{:03}.wav'.format(nfold+1, nr+1, nmix+1)
                    soundfile.write(self._outpath + '/' + mixture_filename, mixsig, self._fs_mix)


                




