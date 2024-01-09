import numpy as np
import scipy
import scipy.io
import scipy.signal as signal
import scaper_utils
import os
import mat73
import soundfile
import pickle
from spatial_scaper.data import sofa_utils

import librosa # used since we allow .wav and .mp3 loading 

def sample_from_quartiles(K, stats):
    minn = stats[0]
    maxx = stats[4]
    quart1 = stats[1]
    mediann = stats[2]
    quart3 = stats[3]
    samples = minn + (quart1 - minn)*np.random.rand(K, 1)
    samples = np.append(samples,quart1)
    samples = np.append(samples, quart1 + (mediann-quart1)*np.random.rand(K,1))
    samples = np.append(samples,mediann)
    samples = np.append(samples, mediann + (quart3-mediann)*np.random.rand(K,1))
    samples = np.append(samples, quart3)
    samples = np.append(samples, quart3 + (maxx-quart3)*np.random.rand(K,1))
    
    return samples
    

def stft_ham(insig, winsize=256, fftsize=512, hopsize=128):
    nb_dim = len(np.shape(insig))
    lSig = int(np.shape(insig)[0])
    nCHin = int(np.shape(insig)[1]) if nb_dim > 1 else 1
    x = np.arange(0,winsize)
    nBins = int(fftsize/2 + 1)
    nWindows = int(np.ceil(lSig/(2.*hopsize)))
    nFrames = int(2*nWindows+1)
    
    winvec = np.zeros((len(x),nCHin))
    for i in range(nCHin):
        winvec[:,i] = np.sin(x*(np.pi/winsize))**2
    
    frontpad = winsize-hopsize
    backpad = nFrames*hopsize-lSig

    if nb_dim > 1:
        insig_pad = np.pad(insig,((frontpad,backpad),(0,0)),'constant')
        spectrum = np.zeros((nBins, nFrames, nCHin),dtype='complex')
    else:
        insig_pad = np.pad(insig,((frontpad,backpad)),'constant')
        spectrum = np.zeros((nBins, nFrames),dtype='complex')

    idx=0
    nf=0
    if nb_dim > 1:
        while nf <= nFrames-1:
            insig_win = np.multiply(winvec, insig_pad[idx+np.arange(0,winsize),:])
            inspec = scipy.fft.fft(insig_win,n=fftsize,norm='backward',axis=0)
            #inspec = scipy.fft.fft(insig_win,n=fftsize,axis=0)
            inspec=inspec[:nBins,:]
            spectrum[:,nf,:] = inspec
            idx += hopsize
            nf += 1
    else:
        while nf <= nFrames-1:
            insig_win = np.multiply(winvec[:,0], insig_pad[idx+np.arange(0,winsize)])
            inspec = scipy.fft.fft(insig_win,n=fftsize,norm='backward',axis=0)
            #inspec = scipy.fft.fft(insig_win,n=fftsize,axis=0)
            inspec=inspec[:nBins]
            spectrum[:,nf] = inspec
            idx += hopsize
            nf += 1
    
    return spectrum
    
    
def ctf_ltv_direct(sig, irs, ir_times, fs, win_size):
    convsig = []
    win_size = int(win_size)
    hop_size = int(win_size / 2)
    fft_size = win_size*2
    nBins = int(fft_size/2)+1
    
    # IRs
    ir_shape = np.shape(irs)
    sig_shape = np.shape(sig)
    
    lIr = ir_shape[0]

    if len(ir_shape) == 2:
        nIrs = ir_shape[1]
        nCHir = 1
    elif len(ir_shape) == 3:
        nIrs = ir_shape[2]
        nCHir = ir_shape[1]
    
    if nIrs != len(ir_times):
        return ValueError('Bad ir times')
    
    # number of STFT frames for the IRs (half-window hopsize)
    
    nIrWindows = int(np.ceil(lIr/win_size))
    nIrFrames = 2*nIrWindows+1
    # number of STFT frames for the signal (half-window hopsize)
    lSig = sig_shape[0]
    nSigWindows = np.ceil(lSig/win_size)
    nSigFrames = 2*nSigWindows+1
    
    # quantize the timestamps of each IR to multiples of STFT frames (hopsizes)
    tStamps = np.round((ir_times*fs+hop_size)/hop_size)
    
    # create the two linear interpolator tracks, for the pairs of IRs between timestamps
    nIntFrames = int(tStamps[-1])
    Gint = np.zeros((nIntFrames, nIrs))
    for ni in range(nIrs-1):
        tpts = np.arange(tStamps[ni],tStamps[ni+1]+1,dtype=int)-1
        ntpts = len(tpts)
        ntpts_ratio = np.arange(0,ntpts)/(ntpts-1)
        Gint[tpts,ni] = 1-ntpts_ratio
        Gint[tpts,ni+1] = ntpts_ratio
    
    # compute spectra of irs
    
    if nCHir == 1:
        irspec = np.zeros((nBins, nIrFrames, nIrs),dtype=complex)
    else:
        temp_spec = stft_ham(irs[:, :, 0], winsize=win_size, fftsize=2*win_size,hopsize=win_size//2)
        irspec = np.zeros((nBins, np.shape(temp_spec)[1], nCHir, nIrs),dtype=complex)
    
    for ni in range(nIrs):
        if nCHir == 1:
            irspec[:, :, ni] = stft_ham(irs[:, ni], winsize=win_size, fftsize=2*win_size,hopsize=win_size//2)
        else:
            spec = stft_ham(irs[:, :, ni], winsize=win_size, fftsize=2*win_size,hopsize=win_size//2)
            irspec[:, :, :, ni] = spec#np.transpose(spec, (0, 2, 1))
    
    #compute input signal spectra
    sigspec = stft_ham(sig, winsize=win_size,fftsize=2*win_size,hopsize=win_size//2)
    #initialize interpolated time-variant ctf
    Gbuf = np.zeros((nIrFrames, nIrs))
    if nCHir == 1:
        ctf_ltv = np.zeros((nBins, nIrFrames),dtype=complex)
    else:
        ctf_ltv = np.zeros((nBins,nIrFrames,nCHir),dtype=complex)
    
    S = np.zeros((nBins, nIrFrames),dtype=complex)
    
    #processing loop
    idx = 0
    nf = 0
    inspec_pad = sigspec
    nFrames = int(np.min([np.shape(inspec_pad)[1], nIntFrames]))
    
    convsig = np.zeros((win_size//2 + nFrames*win_size//2 + fft_size-win_size, nCHir))
    
    while nf <= nFrames-1:
        #compute interpolated ctf
        Gbuf[1:, :] = Gbuf[:-1, :]
        Gbuf[0, :] = Gint[nf, :]
        if nCHir == 1:
            for nif in range(nIrFrames):
                ctf_ltv[:, nif] = np.matmul(irspec[:,nif,:], Gbuf[nif,:].astype(complex))
        else:
            for nch in range(nCHir):
                for nif in range(nIrFrames):
                    ctf_ltv[:,nif,nch] = np.matmul(irspec[:,nif,nch,:],Gbuf[nif,:].astype(complex))
        inspec_nf = inspec_pad[:, nf]
        S[:,1:nIrFrames] = S[:, :nIrFrames-1]
        S[:, 0] = inspec_nf
        
        repS = np.tile(np.expand_dims(S,axis=2), [1, 1, nCHir])
        convspec_nf = np.squeeze(np.sum(repS * ctf_ltv,axis=1))
        first_dim = np.shape(convspec_nf)[0]
        convspec_nf = np.vstack((convspec_nf, np.conj(convspec_nf[np.arange(first_dim-1, 1, -1)-1,:])))
        convsig_nf = np.real(scipy.fft.ifft(convspec_nf, fft_size, norm='forward', axis=0)) ## get rid of the imaginary numerical error remain
        # convsig_nf = np.real(scipy.fft.ifft(convspec_nf, fft_size, axis=0))
        #overlap-add synthesis
        convsig[idx+np.arange(0,fft_size),:] += convsig_nf
        #advance sample pointer
        idx += hop_size
        nf += 1
    
    convsig = convsig[(win_size):(nFrames*win_size)//2,:]
    
    return convsig

class AudioSynthesizer(object):
    def __init__(
            self, params, mixtures, mixture_setup, db_config, audio_format
            ):
        self._mixtures = mixtures
        self._rirpath_sofa = params['rirpath_sofa']
        self._db_path = params['db_path']
        self._audio_format = audio_format
        self._outpath = params['mixturepath'] + '/' + mixture_setup['scenario'] + '/' + self._audio_format
        self._rirdata = db_config._rirdata
        self._nb_rooms = sum(os.path.isdir(self._rirpath_sofa+f'/{self._audio_format}/'+i) for i in os.listdir(self._rirpath_sofa+f'/{self._audio_format}/'))
        self._room_names = sorted([d for d in os.listdir(self._rirpath_sofa+f'/{self._audio_format}/') if os.path.isdir(self._rirpath_sofa+f'/{self._audio_format}/'+d)])
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
                
                sofas = sorted(os.listdir(self._rirpath_sofa+f'/{self._audio_format}/'+nroom))
                channel_rirs_sofa = []
                max_long = 0
                for sofa in sofas:
                    rirs = sofa_utils.load_rir(self._rirpath_sofa + f'/{self._audio_format}/'+nroom+f'/{sofa}')
                    if len(rirs) > max_long:
                        max_long = len(rirs)
                    channel_rirs_sofa.append(np.transpose(rirs,(2,1,0))[...,np.newaxis])
                for i in range(len(channel_rirs_sofa)):
                    diff_ = max_long - len(channel_rirs_sofa[i][0,0])
                    channel_rirs_sofa[i] = np.pad(channel_rirs_sofa[i], ((0,0),(0,0),(0,diff_),(0,0)))
                channel_rirs = np.concatenate(channel_rirs_sofa,axis=-1)
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
                            eventsig, fs_db = librosa.load(filename, sr=self._fs) # here we need librosa since we are loading .mp3 
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
                            mixeventsig = 481.6989*scaper_utils.ctf_ltv_direct(eventsig, channel_rirs[:, :, riridx, ntraj], ir_times, self._fs_mix, self._stft_winsize_moving) / float(len(eventsig))
                        else:
                            mixeventsig0 = scipy.signal.convolve(eventsig, np.squeeze(channel_rirs[:, 0, riridx, ntraj]), mode='full', method='fft')
                            mixeventsig1 = scipy.signal.convolve(eventsig, np.squeeze(channel_rirs[:, 1, riridx, ntraj]), mode='full', method='fft')
                            mixeventsig2 = scipy.signal.convolve(eventsig, np.squeeze(channel_rirs[:, 2, riridx, ntraj]), mode='full', method='fft')
                            mixeventsig3 = scipy.signal.convolve(eventsig, np.squeeze(channel_rirs[:, 3, riridx, ntraj]), mode='full', method='fft')

                            mixeventsig = np.stack((mixeventsig0,mixeventsig1,mixeventsig2,mixeventsig3),axis=1)
                        if self._apply_event_gains:
                            # apply random gain to each event based on class gain, distribution given externally
                            K=1000
                            rand_energies_per_spec = scaper_utils.sample_from_quartiles(K, self._class_gains[classidx])
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


                




