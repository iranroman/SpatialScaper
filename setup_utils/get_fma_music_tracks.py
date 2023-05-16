import os
import random
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

FMA_FS = 44100 # FMA sampling rate
NEW_FS = 24000 # DCASE synth sampling rate
ntracks = 50 # No. tracks to be taken from each genre
train_test_split = 0.6 # train vs. test split probability
# Set relevant paths
train_dir = '/path/to/music/train' # destination
test_dir = '/path/to/music/test' # destination
FMA_PATH = "/path/to/fma_small/" # source
# Load the FMA metadata
tracks = pd.read_csv(os.path.join(FMA_PATH,'tracks.csv'), header=[0, 1], index_col=0)
# Create directories for the train and test sets
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get genres available from FMA dataset
genres = tracks['track']['genre_top'].unique()

# Loop through the genre 8 classes
for genre in genres:
    # Get tracks by genre, consider only set from "small" tracks
    genre_tracks = tracks[(tracks['track', 'genre_top'] == genre) & (tracks['set', 'subset'] == 'small')]
    # Get ntracks from the current genre
    train_tracks = genre_tracks[:ntracks]
    for track_id, track in train_tracks.iterrows():
        # get track name by id
        audio_path = os.path.join(FMA_PATH,f'{track_id:06}.mp3')
        # Load the audio data
        y, sr = librosa.load(audio_path, sr=FMA_FS, mono=True)
        y = librosa.resample(y, orig_sr=FMA_FS, target_sr=NEW_FS)
        # Based on prob decide test vs. train split
        save_dir = None
        if np.random.rand() < train_test_split:
            save_dir = train_dir
        else:
            save_dir = test_dir
        output_path = os.path.join(save_dir, str(genre), f'{track_id:06}.wav')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, y, NEW_FS, 'PCM_16')

