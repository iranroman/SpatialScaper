# DCASE2022-data-generator
Data generator for creating synthetic audio mixtures suitable for DCASE Challenge 2022 Task 3

### Prerequisites

The provided code was tested with Python 3.8 and the following libraries:
SoundFile 0.10.3, mat73 0.58, numpy 1.20.1, scipy 1.6.2, librosa 0.8.1. 

Must also download these RIR databases:
* TAU SRIR DB: https://zenodo.org/record/6408611

Also download
* FSD50K https://zenodo.org/record/4060432
* orchset (could be done using `mirdata`)

```
pip install mirdata
```

```
>>> import mirdata
>>> orchset = mirdata.initialize('orchset')
>>> orchset = mirdata.initialize('orchset', data_home='/home/iran/datasets/orchset')
>>> orchset.download()
```

```
cp /path/to/orchset/audio/mono/* /path/to/FSD50K/
```

## Getting Started

This repository contains several Python file, which in total create a complete data generation framework.
* The `generation_parameters.py` is a separate script used for setting the parameters for the data generation process, including things such as audio dataset, number of folds, mixuture length, etc.
* The `db_config.py` is a class for containing audio filelists and data parameters from different audio datasets used for the mixture generation.
* The `metadata_synthesizer.py` is a class for generating the mixture target labels, along with the corresponding metadata and statistics. Information from this class can be further used for synthesizing the final audios.
* The `audio_synthesizer.py` is a class for synthesizing noiseless audio files containing the simulated mixtures.
* The `audio_mixer.py` is a class for mixing the generated audio mixtures with background noise and/or interference mixtures.
* The `make_dataset.py` is the main script in which the whole framework is used to perform the full data generation process.
* The `utils.py` is an additional file containing complementary functions for other scripts.

Moreover, an object file is included in case the database configuration via `db_config.py` takes too much time:
* The `db_config_fsd.obj` is a DBConfig class containing information about the database and files for the FSD50K audioset.
* you will need to run the `mat2dict.py` script to convert `matlab` files with RIR data into python pickles. 

```
python mat2dict.py /path/to/TAU_DB/TAU-SRIR_DB/
``` 


The exemplary script is:
* The `example_script_DCASE2022.py` is a script showing a pipeline to generate data similar to the current DCASE2022 dataset.

### Using the generated dataset to train the DCASE 2023 Task 3 audio-only baseline you should get results similar to these:


| Dataset | ER<sub>20°</sub> | F<sub>20°</sub> | LE<sub>CD</sub> | LR<sub>CD</sub> |
| ----| --- | --- | --- | --- |
| Ambisonic (FOA + Multi-ACCDOA) | 0.60 | 28.7 % | 23.2&deg; | 48.8 % |
| Microphone Array (MIC-GCC + Multi-ACCDOA) | 0.64 | 26.9 % | 23.8&deg; | 46.2 % |
