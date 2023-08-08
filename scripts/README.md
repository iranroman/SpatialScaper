# Dataset preparation scripts

## `prepare_fsd50k`

The class `FSD50KDataLoad` is in charge of the FSD50K dataset preparation. It can download or simply load the dataset and its metadata into a specific format needed for the SELD data generator.

### Download
Set `"download": True` within the configuration dictionary `PARAM_CONFIG` in `prepare_fsd50k.py`

### Usage and Load only

If the dataset already exists and all you need is to load the fsd50k dictionary:

```
PARAM_CONFIG = {
    "dataset_home": "/path/to" # add only /path/to the FSD50K location
    "metadata_path": "/path/to" # add only /path/to the metadata location,
    "dcase_sound_events_txt": os.path.join(PARENT_DIR,"dcase_datagen/metadata/sound_event_fsd50k_filenames.txt"), # I recommend leaving this file path as is
    "download": False,
    "music_home": "/path/to" # add only /path/to the FMA location
    "music_metadata": os.path.join(PARENT_DIR, "dcase_datagen/metadata"),  # I recommend leaving this file path as is
    "ntracks_genre": 10,
    "split_prob": 0.6
}
```

#### Get fsd50k dataset object
```
import prepare_fsd50k
fsd50k = prepare_fsd50k()
```

#### Use dictionary to find DCASE filenames to FSD50K path names
```
# retrive dcase filenames to FSD50K filepaths
filenames = fsd50k.get_filenames('train')
# From a filename, get the its file path:
dcase_filename = 'waterTap/train/train/371615.wav'
fsd_filepath = filenames[dcase_filename]
```

## More info

'FSD50KDataLoad' dictionary organizes all tracks by folds (`train`, `test`) and maps filenames to their path in the correspnding dataset. Example dictionary when calling `fsd50k.save_fsd_to_dcase_json()`

```
"train": {
    "music/train/Hip-Hop/000002.mp3": "../dcase_datagen/data/fma_small/000/000002.mp3",
    ...
"test": {
    "music/test/Hip-Hop/004684.mp3": "../dcase_datagen/data/fma_small/004/004684.mp3",
    ...
}
```

## `prepare_tau_srir_db`

TBD