import numpy as np
import spatialscaper as ss
import os

# currently only DCASE Task 3 (SELD) format supported

# Constants
NSCAPES = 25
FOREGROUND_DIR = "datasets/sound_event_datasets/FSD50K_FMA"
BACKGROUND_DIR = ""
RIR_DIR = "datasets/rir_datasets"
ROOM = "metu"
FORMAT = "mic" 
N_EVENTS_MEAN = 15
N_EVENTS_STD = 6
DURATION = 60.0  # Duration in seconds
SR = 24000  # Sampling rate
OUTPUT_DIR = "output"
REF_DB = -65  # Reference decibel level


# Function to generate a soundscape
def generate_soundscape(index):
    track_name = f"fold5_room1_mix00{index + 1}"
    ssc = ss.Scaper(
        DURATION, FOREGROUND_DIR, BACKGROUND_DIR, RIR_DIR, ROOM, FORMAT, SR
    )
    ssc.ref_db = REF_DB

    # Add background (static white noise for now)
    ssc.add_background()

    # Add a random number of foreground events
    n_events = int(np.random.normal(N_EVENTS_MEAN, N_EVENTS_STD))
    # for this duration and n_events distribution, more than 25 breaks recursion
    n_events = n_events if 0 < n_events < 25 else N_EVENTS_MEAN
    for _ in range(n_events):
        ssc.add_event()

    audiofile = os.path.join(OUTPUT_DIR, FORMAT, track_name)
    labelfile = os.path.join(OUTPUT_DIR, "labels", track_name)

    ssc.generate(audiofile, labelfile)


# Main loop for generating soundscapes
for iscape in range(NSCAPES):
    print(f"Generating soundscape: {iscape + 1}/{NSCAPES}")
    generate_soundscape(iscape)
