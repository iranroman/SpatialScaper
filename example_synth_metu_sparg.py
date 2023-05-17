import sys
import os
import glob
import numpy as np
# from db_config import DBConfig
from metadata_synthesizer import MetadataSynthesizer
from audio_synthesizer import AudioSynthesizer
from audio_mixer import AudioMixer
import pickle
from generation_parameters import get_params

'''
This script is the first instance of DCASE SELD style synthesis using 32ch from the METU-Sparg dataset
'''

# use parameter set defined by user
task_id = '2' # NOTE: to be deprecated soon given the introduction of .yaml files instead

params = get_params(task_id)
    
### Create database config based on params (e.g. filelist name etc.)
# db_config = DBConfig(params)
    
# # LOAD DB-config which is already done
db_handler = open('db_config_fsd.obj','rb')
db_config = pickle.load(db_handler)
db_handler.close()

file = open("rirdata_dict.pkl",'rb')
db_config._rirdata = pickle.load(file)
file.close()

### add the METU-SPARG room
file = open("/scratch/data/SELD-data-generator/metusparg/doa_xyz_mic.pkl",'rb') # need a better name
db_config._rirdata['metu'] = {} 
db_config._rirdata['metu']['doa_xyz'] = pickle.load(file)
file.close()

music_dir = os.path.join(params['db_path'], "music")
# list of train music wave file paths
tr_wave_files = []
# Use glob to recursively search for all wave files in the directory and its subdirectories
for file in glob.glob(os.path.join(music_dir, "train", '**/*.wav'), recursive=True):
    tr_wave_files.append(file)

# list of test music wave file paths
ts_wave_files = []
# Use glob to recursively search for all wave files in the directory and its subdirectories
for file in glob.glob(os.path.join(music_dir, "test", '**/*.wav'), recursive=True):
    ts_wave_files.append(file)

tr_samplelist = db_config._samplelist[0]
i = 0
sample_list = []
for sample in tr_samplelist['audiofile']:
    # print(sample)
    if 'music/' in sample:
        sample = '/{}'.format(tr_wave_files[i%len(tr_wave_files)])
        print(sample)
        i += 1
    sample_list.append(sample)
db_config._samplelist[0]['audiofile'] = np.array(sample_list)

ts_samplelist = db_config._samplelist[1]
i = 0
sample_list = []
for sample in ts_samplelist['audiofile']:
    print(sample)
    if 'music/' in sample:
        sample = '/{}'.format(ts_wave_files[i%len(ts_wave_files)])
        i += 1
    sample_list.append(sample)
db_config._samplelist[1]['audiofile'] = np.array(sample_list)

#create mixture synthesizer class
noiselessSynth = MetadataSynthesizer(db_config, params, 'target_noiseless')
    
#create mixture targets
mixtures_target, mixture_setup_target, foldlist_target = noiselessSynth.create_mixtures()
    
#calculate statistics and create metadata structure
metadata, stats = noiselessSynth.prepare_metadata_and_stats()
    
#write metadata to text files
noiselessSynth.write_metadata()
    

if not params['audio_format'] == 'both': # create a dataset of only one data format (FOA or MIC)
    #create audio synthesis class and synthesize audio files for given mixtures
    noiselessAudioSynth = AudioSynthesizer(params, mixtures_target, mixture_setup_target, db_config, params['audio_format'])
    noiselessAudioSynth.synthesize_mixtures()
        
    #mix the created audio mixtures with background noise
    audioMixer = AudioMixer(params, db_config, mixtures_target, mixture_setup_target, params['audio_format'], 'target_noisy')
    audioMixer.mix_audio()
else:
    #create audio synthesis class and synthesize audio files for given mixtures
    noiselessAudioSynth = AudioSynthesizer(params, mixtures_target, mixture_setup_target, db_config, 'foa')
    noiselessAudioSynth.synthesize_mixtures()
    noiselessAudioSynth2 = AudioSynthesizer(params, mixtures_target, mixture_setup_target, db_config, 'mic')
    noiselessAudioSynth2.synthesize_mixtures()
        
    #mix the created audio mixtures with background noise
    audioMixer = AudioMixer(params, db_config, mixtures_target, mixture_setup_target, 'foa', 'target_noisy')
    audioMixer.mix_audio()
    audioMixer2 = AudioMixer(params, db_config, mixtures_target, mixture_setup_target, 'mic', 'target_noisy')
    audioMixer2.mix_audio()
