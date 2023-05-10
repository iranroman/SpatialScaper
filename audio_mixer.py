import numpy as np
import os
import soundfile

class AudioMixer(object):
    def __init__(
            self, params, db_config, mixtures, mixture_setup, audio_format, scenario_out, scenario_interf='target_interf'
            ):
        self._recpath2020 = params['noisepath']
        self._rooms_paths2020 = ['01_bomb_center','02_gym','03_pb132_paatalo_classroom2','04_pc226_paatalo_office',
                                  '05_sa203_sahkotalo_lecturehall','06_sc203_sahkotalo_classroom2','07_se201_sahkotalo_classroom',
                                  '08_se203_sahkotalo_classroom','09_tb103_tietotalo_lecturehall',
                                  '10_tc352_tietotalo_meetingroom']
        self._nb_rooms2020 = len(self._rooms_paths2020)
        self._recpath2019 = params['noisepath']
        self._rooms_paths2019 = ['11_language_center','12_tietotalo','13_reaktori','14_sahkotalo','15_festia']
        self._nb_rooms2019 = len(self._rooms_paths2019)        
        self._mixturepath = params['mixturepath']
        self._mixtures = mixtures
        self._targetpath = self._mixturepath + '/' + mixture_setup['scenario']
        if scenario_out == 'target_noisy':
            self._scenarios = [1, 0, 1]
        elif scenario_out == 'target_interf_noiseless':
            self._scenarios = [1, 1, 0]
        elif scenario_out == 'target_interf_noisy':
            self._scenarios = [1, 1, 1]
        else:
            raise ValueError('Incorrect scenario specified')
        
        self._scenariopath = self._mixturepath + '/' + scenario_out
        self._audio_format = audio_format
        if self._audio_format == 'mic':
            self._mic_format = 'tetra'
        elif self._audio_format == 'foa':
            self._mic_format = 'foa_sn3d'

        if self._scenarios[1]:
            self._interfpath = self._mixturepath + '/' + scenario_interf
        self._fs_mix = mixture_setup['fs_mix']
        self._tMix = mixture_setup['mixture_duration']
        self._lMix = int(np.round(self._fs_mix*self._tMix))
        # target signal-to-interference power ratio
        #  set at 3 now of all targets, so that the total interference power is 
        #approximately 0dB wrt. to a single layer of targets (for 3 target layers)
        self._sir = 3.
        self._nb_folds = mixture_setup['nb_folds']
        self._rnd_generator = np.random.default_rng(2024)

    def mix_audio(self):
        if not os.path.isdir(self._scenariopath + '/' + self._audio_format):
            os.makedirs(self._scenariopath + '/' + self._audio_format)
        # start creating the mixtures description structure
        for nfold in range(self._nb_folds):
            print('Adding noise for fold {}'.format(nfold+1))
            rooms = self._mixtures[nfold][0]['roomidx']
            nb_rooms = len(rooms)
            for nr in range(nb_rooms):
                nroom = rooms[nr]
                
                if self._scenarios[2]:
                    print('Loading ambience')
                    recpath = self._recpath2020 if nroom <=10 else self._recpath2019
                    roompath = self._rooms_paths2020 if nroom <= 10 else self._rooms_paths2019
                    roomidx = nroom if nroom <= 10 else nroom-10
                    ambience, _ = soundfile.read(recpath  + '/' + roompath[roomidx-1] + '/ambience_' + self._mic_format + '_24k_edited.wav')
                    lSig = np.shape(ambience)[0]
                    nSegs = np.floor(lSig/self._lMix)
                
                nb_mixtures = len(self._mixtures[nfold][nr]['mixture'])
                for nmix in range(nb_mixtures):
                    print('Loading target mixture {}/{} \n'.format(nmix+1,nb_mixtures))
                    mixture_filename = 'fold{}_room{}_mix{:03}.wav'.format(nfold+1,nr+1,nmix+1)
                    snr = self._mixtures[nfold][nr]['mixture'][nmix]['snr']
                    
                    target_sig, _ = soundfile.read(self._targetpath + '/' + self._audio_format + '/' + mixture_filename)
                    target_omni_energy = np.sum(np.mean(target_sig,axis=1)**2) if self._audio_format == 'mic' else np.sum(target_sig[:,0]**2)
                    
                    if self._scenarios[1]:
                        print('Loading interferer mixture {}/{} \n'.format(nmix+1, nb_mixtures))
                        interf_sig, _ = soundfile.read(self._interfpath + '/' + self._audio_format + '/' + mixture_filename)
                        inter_omni_energy = np.sum(np.mean(interf_sig,axis=1)**2) if self._audio_format == 'mic' else np.sum(interf_sig[:,0]**2)
                        interf_norm = np.sqrt(inter_omni_energy/(self._sir * inter_omni_energy))
                        ## ADD INTERFERENCE 
                        target_sig += interf_norm * interf_sig
                    
                    if self._scenarios[2]:
                        # check if the number of mixture is lower than the duration of the noise recordings
                        idx_range = np.arange(0,self._lMix,dtype=int) # computed here for convenience
                        if nmix < nSegs:
                            ambient_sig = ambience[nmix*self._lMix+idx_range, :]
                        else:
                            # else just mix randomly two segments
                            rand_idx = self._rnd_generator.integers(0,nSegs,2)
                            ambient_sig = ambience[rand_idx[0]*self._lMix+ idx_range, :]/np.sqrt(2) 
                            + ambience[rand_idx[1]*self._lMix + idx_range, :]/np.sqrt(2)
                            
                        ambi_energy = np.sum(np.mean(ambient_sig,axis=1)**2) if self._audio_format == 'mic' else np.sum(ambient_sig[:,0]**2)
                        ambi_norm = np.sqrt(target_omni_energy * 10.**(-snr/10.) / ambi_energy)
                        
                        ## ADD NOISE 
                        target_sig += ambi_norm * ambient_sig
                    
                    
                    soundfile.write(self._scenariopath+'/'+self._audio_format+'/'+mixture_filename, target_sig, self._fs_mix)
                    
                    
