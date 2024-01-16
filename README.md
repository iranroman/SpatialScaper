# SpatialScaper

A python library to create synthetic audio mixtures suitable for DCASE Challenge Task 3

Install SpatialScaper
```
pip install -e .
```

To start using SpatialScaper using the default FSD50K dataset used by the [DCASE data generator](https://github.com/danielkrause/DCASE2022-data-generator):

1. download and unzip the FSD50K dataset
```
zenodo_get -r 4060432 -o path/to/datasets
zip -s 0 path/to/datasets/FSD50K.dev_audio.zip --out path/to/datasets/unsplit.zip
unzip path/to/datasets/unsplit.zip -d path/to/datasets
zip -s 0 path/to/datasets/FSD50K.eval_audio.zip --out path/to/datasets/unsplit.zip
unzip path/to/datasets/unsplit.zip -d path/to/datasets
```

2. re-structure FSD50K to be consistent with the structure expected by the [DCASE data generator](https://github.com/danielkrause/DCASE2022-data-generator)
```
python scripts/fsd50k_to_dcase_format.py path/to/datasets/FSD50K path/to/datasets/FSD50K_DCASE
```
[comment]: <> (TODO: add steps to clean up redundant files after preparing FSD50K)

### METU ROOM
Download and prepare
```
zenodo_get -r 2635758 -o path/to/datasets
unzip path/to/datasets/spargair.zip -d path/to/datasets
rm path/to/datasets/spargair.zip
rm -r path/to/datasets/__MACOSX
python scripts/prepare_metu.py
```

### Example script to generate soundscapes
Spatialize
```
python mvp.py
```

```
requirements.txt
    zenodo_get
    librosa
    soundfile
    pysofaconventions
    netCDF4
    scipy
    matplotlib
    scaper
```













### Prerequisites

The provided code was tested with Python 3.10 and the following libraries:
SoundFile 0.10.3, mat73 0.58, numpy 1.20.1, scipy 1.6.2, librosa 0.8.1. 

Must download these RIR databases:
* TAU SRIR DB: https://zenodo.org/record/6408611

### One-time setup

This data synthesizer uses sound events from the [FSD50K](https://zenodo.org/record/4060432#.ZE7ely2B0Ts) dataset. To download the dataset for first time and to also include music tracks (from the the [FMA dataset](https://github.com/mdeff/fma) dataset) as part of the data synthesizer sound events, do the following under `prepare_fsd50k.py`:

- Change the dataset parameter configuration paths and select `"download": True` to download the FSD50K dataset along with the music FMA dataset:

```
PARAM_CONFIG = {
    "dataset_home": "/datasets/FSD50K", # add /path/to (not path/to/dir)
    "metadata_path": "dcase_datagen/metadata", # add /path/to (not path/to/dir)
    "dcase_sound_events_txt": "dcase_datagen/metadata/sound_event_fsd50k_filenames.txt",
    "download": True,
    "music_home": "/datasets/fma", # add /path/to (not path/to/dir)
    "music_metadata": "dcase_datagen/metadata", # add /path/to (not path/to/dir)
    "ntracks_genre": 40,
    "split_prob": 0.6
}
```
*Note:* if you already have the FSD50K dataset downloaded simply update the `/path/to` the dataset (the same applies for the FMA dataset) and set `"download": False`.

Execute the script by:

```
python prepare_fsd50k.py
```

Also trim the fma files to be 10 seconds long using the provided script (otherwise model performance will suffer for transient sound events)
```
python scripts/trim_fma.py
```

In practice, however, to generate data all you need to do is run the exemplary script is:
* The `example_script_DCASE2022.py` is a script showing a pipeline to generate data.

* you will need to run the `mat2dict.py` script to convert `matlab` files with RIR data into python pickles. 

```
python mat2dict.py /path/to/TAU_DB/TAU-SRIR_DB/
``` 

### Using the generated dataset to train the DCASE 2023 Task 3 audio-only baseline you should get results similar to these:

| Dataset | ER<sub>20°</sub> | F<sub>20°</sub> | LE<sub>CD</sub> | LR<sub>CD</sub> |
| ----| --- | --- | --- | --- |
| Ambisonic (FOA + Multi-ACCDOA) | 0.60 | 28.7 % | 23.2&deg; | 48.8 % |
| Microphone Array (MIC-GCC + Multi-ACCDOA) | 0.64 | 26.9 % | 23.8&deg; | 46.2 % |

## Other info:

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

under the hood, the data synthesizer will load all the necessary data and metada by simply loading `prepare_fsd50k` as a python module.

```
from prepare_fsd50k import prepare_fsd50k

fsd50k = prepare_fsd50k() # object embedding data and metadata paths

# e.g.: to retrive 'train' DCASE filenames into FSD50K filepaths
filenames = fsd50k.get_filenames('train')
```
