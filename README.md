<!-- omit in toc -->
# SpatialScaper: a library to simulate and augment soundscapes for sound event localization and detection in realistic rooms.
[![Platform](https://img.shields.io/badge/Platform-linux-lightgrey?logo=linux)](https://www.linux.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-orange?logo=python)](https://www.python.org/)	
[![arXiv](https://img.shields.io/badge/Arxiv-2401.03497-blueviolet?logo=arxiv)](https://arxiv.org/abs/2401.12238)
[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> [!WARNING]
> SpatialScaper is still undergoing active development. We have done our due diligence to test that  `example_generation.py` works as expected. However, please open an issue and describe any errors you encounter. Also, make sure to pull often, as we are actively adding more features. Note: You'll need 100GB of storage space to comfortably setup and run the DCASE Task 3 data generation pipeline. 

**Guides**
- [Requirements and Installation](#requirements-and-installation)
- [Preparing Sound Event Assets](#preparing-sound-event-assets)
- [Preparing RIR Datasets](#preparing-rir-datasets)
- [Example data generation (for DCASE Task 3)](#quick-examples-for-new-users)

<!-- omit in toc -->
## Introduction
SpatialScaper is a python library to create synthetic audio mixtures suitable for DCASE Challenge Task 3.

## Requirements and Installation
To run the SpatialScaper library, manually setup your environment as follows.

<!-- omit in toc -->
#### Manual Environment Setup
The minimum environment requirements are `Python >= 3.8`. You could find the versions of other dependencies we use in `setup.py`. 
```shell 
git clone https://github.com/iranroman/SpatialScaper.git
cd SpatialScaper
pip install --editable ./
```
<details>
<summary>Click for more details</summary>

### Conda Enviroment with Python==3.8

```
conda create -n "ssenv" python=3.8
```

### Python Virtual Enviroment with Python==3.8

```
python3.8 -m venv "ssenv"
```

</details>

## Preparing Sound Event Assets

First we need to prepare sound event assets for soundscape synthesis. SpatialScaper works with any sound files that you wish to spatialize. You can get started using sound events from the [FSD50K](https://zenodo.org/record/4060432#.ZE7ely2B0Ts) and [FMA](https://github.com/mdeff/fma) (music) dataset by using.
```shell
python scripts/prepare_fsd50k_fma.py --download_FSD --download_FMA --data_dir datasets
```

This creates a `datasets/sound_event_datasets/FSD50K_FMA` directory with a structure of sound event categories and files. 

**Attention:** the first time setup takes some time ⏳, we recommend running under a `screen` or `tmux` session.

## Preparing RIR Datasets

```
python scripts/prepare_rirs.py
```

**Attention:** the first time setup takes some time ⏳, we recommend running under a `screen` or `tmux` session.

Note: stay tuned as we will soon release our A2B ambisonics encoder. In the meantime, download the METU FOA sofa file from this [google drive link](https://drive.google.com/file/d/1zamCd6OR6Tr5M40RdDhswYbT1wbGo2ZO/view?usp=sharing). Place alongside all other sofa files that `prepare_rirs.py` generates under `SpatialScaper/datasets/rir_datasets/spatialscaper_RIRs`. 

<details>
<summary>Full descriptions of available rooms </summary>

The available rooms for soundscape generation are as follows:

| Room Name     | Description                                                                                     | Trajectory type | URL                                 |
|---------------|-------------------------------------------------------------------------------------------------|-----------------|-------------------------------------|
| metu          | Classroom S05 at the METU Graduate School of Informatics on 23 January 2018.                   | Square          | [Link](https://zenodo.org/records/2635758) |
| arni          | Arni variable acoustics room at the Acoustics Lab, Aalto University, Espoo, Finland.           | Linear          | [Link](https://zenodo.org/records/5720724) |
| bomb_shelter  | Large open space in underground bomb shelter, with plastic-coated floor and rock walls. Ventilation noise. | Circular  | [Link](https://zenodo.org/records/6408611) |
| gym           | Large open gym space. Ambience of people using weights and gym equipment in adjacent rooms.     | Circular        | [Link](https://zenodo.org/records/6408611) |
| pb132         | Small classroom with group work tables and carpet flooring. Ventilation noise.                 | Circular        | [Link](https://zenodo.org/records/6408611) |
| pc226         | Meeting room with hard floor and partially glass walls. Ventilation noise.                     | Circular        | [Link](https://zenodo.org/records/6408611) |
| sa203         | Lecture hall with inclined floor and rows of desks. Ventilation noise.                         | Linear          | [Link](https://zenodo.org/records/6408611) |
| sc203         | Small classroom with group work tables and carpet flooring. Ventilation noise.                 | Linear          | [Link](https://zenodo.org/records/6408611) |
| se203         | Large classroom with hard floor and rows of desks. Ventilation noise.                          | Linear          | [Link](https://zenodo.org/records/6408611) |
| tb103         | Lecture hall with inclined floor and rows of desks. Ventilation noise.                          | Linear          | [Link](https://zenodo.org/records/6408611) |
| tc352         | Meeting room with hard floor and partially glass walls. Ventilation noise.                     | Circular        | [Link](https://zenodo.org/records/6408611) |

Note that SRIR directions and distances differ with the room. Possible azimuths span the whole range of $\phi\in[-180,180)$, while the elevations span approximately a range between $\theta\in[-50,50]$ degrees.

</details>

## Quick Examples for New Users

Below we present the [example_generation.py](example_generation.py). The example generates 20 soundscapes, 1 minute long each, using audio clips from FSD50K, spatialized in the `gym` room. These soundscapes are consistent with the DCASE Task 3 format. 

Execute as:

```shell
python example_generation.py
```

```python
import numpy as np
import spatialscaper as ss
import os

# Constants
NSCAPES = 20  # Number of soundscapes to generate
FOREGROUND_DIR = "datasets/sound_event_datasets/FSD50K_FMA"  # Directory with FSD50K foreground sound files
RIR_DIR = (
    "datasets/rir_datasets"  # Directory containing Room Impulse Response (RIR) files
)
ROOM = "bomb_shelter"  # Initial room setting, change according to available rooms listed below
FORMAT = "mic"  # Output format specifier
N_EVENTS_MEAN = 15  # Mean number of foreground events in a soundscape
N_EVENTS_STD = 6  # Standard deviation of the number of foreground events
DURATION = 60.0  # Duration in seconds of each soundscape, customizable by the user
SR = 24000  # SpatialScaper default sampling rate for the audio files
OUTPUT_DIR = "output"  # Directory to store the generated soundscapes
REF_DB = -65  # Reference decibel level for the background ambient noise. Try making this random too!

# List of possible rooms to use for soundscape generation. Change 'ROOM' variable to one of these:
# "metu", "arni","bomb_shelter", "gym", "pb132", "pc226", "sa203", "sc203", "se203", "tb103", "tc352"
# Each room has a different Room Impulse Response (RIR) file associated with it, affecting the acoustic properties.

# FSD50K sound classes that will be spatialized include:
# 'femaleSpeech', 'maleSpeech', 'clapping', 'telephone', 'laughter',
# 'domesticSounds', 'footsteps', 'doorCupboard', 'music',
# 'musicInstrument', 'waterTap', 'bell', 'knock'.
# These classes are sourced from the FSD50K dataset, and
# are consistent with the DCASE SELD challenge classes.


# Function to generate a soundscape
def generate_soundscape(index):
    track_name = f"fold5_room1_mix{index+1:03d}"
    # Initialize Scaper. 'max_event_overlap' controls the maximum number of overlapping sound events.
    ssc = ss.Scaper(
        DURATION,
        FOREGROUND_DIR,
        RIR_DIR,
        FORMAT,
        ROOM,
        max_event_overlap=2,
        speed_limit=2.0,  # in meters per second
    )
    ssc.ref_db = REF_DB

    # static ambient noise
    ssc.add_background()

    # Add a random number of foreground events, based on the specified mean and standard deviation.
    n_events = int(np.random.normal(N_EVENTS_MEAN, N_EVENTS_STD))
    n_events = n_events if n_events > 0 else 1  # n_events should be greater than zero

    for _ in range(n_events):
        ssc.add_event()  # randomly choosing and spatializing an FSD50K sound event

    audiofile = os.path.join(OUTPUT_DIR, FORMAT, track_name)
    labelfile = os.path.join(OUTPUT_DIR, "labels", track_name)

    ssc.generate(audiofile, labelfile)


# Main loop for generating soundscapes
for iscape in range(NSCAPES):
    print(f"Generating soundscape: {iscape + 1}/{NSCAPES}")
    generate_soundscape(iscape)
```

<!-- omit in toc -->
## Citation
If you find our SpatialScaper library useful, please cite the following paper:
```
@inproceedings{roman2024spatial,
  title={Spatial Scaper: a library to simulate and augment soundscapes for sound event localization and detection in realistic rooms},
  author={Roman, Iran R and Ick, Christopher and Ding, Sivan and Roman, Adrian S and McFee, Brian and Bello, Juan P},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2024},
  organization={IEEE}
}
```

Also cite the RIR and sound event databases that SpatialScaper uses.

```
@dataset{politis_2022_6408611,
  author       = {Politis, Archontis and
                  Adavanne, Sharath and
                  Virtanen, Tuomas},
  title        = {{TAU Spatial Room Impulse Response Database (TAU- 
                   SRIR DB)}},
  month        = apr,
  year         = 2022,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.6408611},
  url          = {https://doi.org/10.5281/zenodo.6408611}
}

@dataset{orhun_olgun_2019_2635758,
  author       = {Orhun Olgun and
                  Huseyin Hacihabiboglu},
  title        = {{METU SPARG Eigenmike em32 Acoustic Impulse 
                   Response Dataset v0.1.0}},
  month        = apr,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {0.1.0},
  doi          = {10.5281/zenodo.2635758},
  url          = {https://doi.org/10.5281/zenodo.2635758}
}

@article{mckenzie2021dataset,
  title={Dataset of spatial room impulse responses in a variable acoustics room for six degrees-of-freedom rendering and analysis},
  author={McKenzie, Thomas and McCormack, Leo and Hold, Christoph},
  journal={arXiv preprint arXiv:2111.11882},
  year={2021}
}

@article{fonseca2021fsd50k,
  title={Fsd50k: an open dataset of human-labeled sound events},
  author={Fonseca, Eduardo and Favory, Xavier and Pons, Jordi and Font, Frederic and Serra, Xavier},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={30},
  pages={829--852},
  year={2021},
  publisher={IEEE}
}

@article{defferrard2016fma,
  title={FMA: A dataset for music analysis},
  author={Defferrard, Micha{\"e}l and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  journal={arXiv preprint arXiv:1612.01840},
  year={2016}
}
```
