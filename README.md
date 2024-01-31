# SpatialScaper

A python library to create synthetic audio mixtures suitable for DCASE Challenge Task 3

Install SpatialScaper
```
pip install -e .
```

### Prerequisites

The provided code was tested with Python 3.8 and the following libraries:
SoundFile 0.10.3, mat73 0.58, numpy 1.20.1, scipy 1.6.2, librosa 0.8.1. 

### Prepare sound event files for soundscape synthesis

SpatialScaper works with any sound files that you wish to spatialize. You can get started using sound events from the [FSD50K](https://zenodo.org/record/4060432#.ZE7ely2B0Ts) and [FMA](https://github.com/mdeff/fma) (music) dataset by using.

```
python scripts/prepare_fsd50k_fma.py --download_FSD --download_FMA --data_dir datasets
```

This creates a `datasets/sound_event_datasets/FSD50K_FMA` directory with a structure of sound event categories and files. 

### Prepare RIR databases

```
python scripts/prepare_metu.py
```

### Example script to generate soundscapes
```
python example_generation.py
```

NOTE: this version is functional but still has many rough edges. Please contribute by commenting on existing issues or creating new ones as appropriate.
