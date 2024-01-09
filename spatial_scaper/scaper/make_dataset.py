import sys
import os
import numpy as np
from metadata_synthesizer import MetadataSynthesizer
from audio_synthesizer import AudioSynthesizer
from audio_mixer import AudioMixer
import pickle
from spatial_scaper.generation_parameters import get_params




def main(argv):
    """
    Main wrapper for the whole data generation framework.
    
    :param argv: expects two optional inputs. 
        first input: task_id - (optional) To chose the generation parameters from generation_parameters.py.
                                (default) 1 - uses default parameters (FSD50K data)
    """
    print(argv)
    if len(argv) != 2:
        print('\n\n')
        print('The code expected an optional input')
        print('\t>> python make_dataset.py <task-id>')
        print('\t\t<task-id> is used to choose the user-defined parameter set from generation_parameters.py')
        print('Using default inputs for now')
        print('\n\n')

    # use parameter set defined by user
    task_id = '2' if len(argv) < 2 else argv[1]

    params = get_params(task_id)
    
    
    #import database config file if provided, sles
    if len(argv) >= 3:
        print(f"Loading db_config {argv[2]}")
        with open(os.path.join('db_configs', argv[2]), 'rb') as f:
            db_config = pickle.load(f)
    else:
        ### Create database config based on params (e.g. filelist name etc.)
        print("No DBConfig!")
    
    #create mixture synthesizer class
    noiselessSynth = MetadataSynthesizer(db_config, params, 'target_noiseless')
    
    #create mixture targets
    mixtures_target, mixture_setup_target, foldlist_target = noiselessSynth.create_mixtures()
    
    #calculate statistics and create metadata structure
    metadata, stats = noiselessSynth.prepare_metadata_and_stats()
    
    #write metadata to text files
    noiselessSynth.write_metadata()
    
    #create audio synthesis class and synthesize audio files for given mixtures
    if not params['audio_format'] == 'both': # create a dataset of only one data format (FOA or MIC)
        noiselessAudioSynth = AudioSynthesizer(params, mixtures_target, mixture_setup_target, db_config, params['audio_format'])
        noiselessAudioSynth.synthesize_mixtures()
        
        #mix the created audio mixtures with background noise
        audioMixer = AudioMixer(params, db_config, mixtures_target, mixture_setup_target, params['audio_format'], 'target_noisy')
        audioMixer.mix_audio()
    else:
        noiselessAudioSynth = AudioSynthesizer(params, mixtures_target, mixture_setup_target, db_config, 'foa')
        noiselessAudioSynth.synthesize_mixtures()
        noiselessAudioSynth2 = AudioSynthesizer(params, mixtures_target, mixture_setup_target, db_config, 'mic')
        noiselessAudioSynth2.synthesize_mixtures()
        
        #mix the created audio mixtures with background noise
        audioMixer = AudioMixer(params, db_config, mixtures_target, mixture_setup_target, 'foa', 'target_noisy')
        audioMixer.mix_audio()
        audioMixer2 = AudioMixer(params, db_config, mixtures_target, mixture_setup_target, 'mic', 'target_noisy')
        audioMixer2.mix_audio()
    

    
if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

