<!-- omit in toc -->
# SpatialScaper: A python library to create synthetic audio mixtures suitable for DCASE Challenge Task 3
[![Platform](https://img.shields.io/badge/Platform-linux-lightgrey?logo=linux)](https://www.linux.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-orange?logo=python)](https://www.python.org/)	
[![arXiv](https://img.shields.io/badge/Arxiv-2401.03497-blueviolet?logo=arxiv)](https://arxiv.org/abs/2401.12238)
[![Contributing][contributing-image]][contributing-url]

**Guides**
- [Requirements and Installation](#requirements-and-installation)
- [Preparing Sound Event Assets](#preparing-sound-event-assets)
- [Preparing RIR Datasets](#preparing-rir-datasets)
- [Quick Examples for New Users](#quick-examples-for-new-users)

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

## Preparing Sound Event Assets

First we need to prepare sound event assets for soundscape synthesis. SpatialScaper works with any sound files that you wish to spatialize. You can get started using sound events from the [FSD50K](https://zenodo.org/record/4060432#.ZE7ely2B0Ts) and [FMA](https://github.com/mdeff/fma) (music) dataset by using.
```shell
python scripts/prepare_fsd50k_fma.py --download_FSD --download_FMA --data_dir datasets
```

This creates a `datasets/sound_event_datasets/FSD50K_FMA` directory with a structure of sound event categories and files. 

## Preparing RIR Datasets

```
python scripts/prepare_rirs.py
```

## Quick Examples for New Users
```shell
# TBD
```

<!-- omit in toc -->
## Citation
If you find our SpatialScaper library useful, please cite the following paper:
```
@article{roman2024spatial,
  title={Spatial scaper: a library to simulate and augment soundscapes for sound event localization and detection in realistic rooms},
  author={Roman, Iran R and Ick, Christopher and Ding, Sivan and Roman, Adrian S and McFee, Brian and Bello, Juan P},
  journal={arXiv preprint arXiv:2401.12238},
  year={2024}
}
```
