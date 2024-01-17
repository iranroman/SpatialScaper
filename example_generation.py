import numpy as np
import spatialscaper as ss
import os

# Constants
NSCAPES = 10
FOREGROUND_DIR = "datasets/FSD50K_DCASE"
BACKGROUND_DIR = ""
SOFA_DIR = "datasets"
ROOM = "metu"
FORMAT = "mic"
MIN_EVENTS = 3
MAX_EVENTS = 8
DURATION = 60.0  # Duration in seconds
SR = 24000  # Sampling rate
OUTPUT_DIR = "output"
REF_DB = -65  # Reference decibel level


# Function to generate a soundscape
def generate_soundscape(index):
    track_name = f"fold5_room1_mix00{index + 1}"
    ssc = ss.Scaper(
        DURATION, FOREGROUND_DIR, BACKGROUND_DIR, SOFA_DIR, ROOM, FORMAT, SR
    )
    ssc.ref_db = REF_DB

    # Add background
    ssc.add_background()

    # Add a random number of foreground events
    n_events = np.random.randint(MIN_EVENTS, MAX_EVENTS + 1)
    for _ in range(n_events):
        ssc.add_event(event_position=("moving", ("uniform", None, None)))

    audiofile = os.path.join(OUTPUT_DIR, FORMAT, track_name)
    labelfile = os.path.join(OUTPUT_DIR, "labels", track_name)

    ssc.generate(audiofile, labelfile)


# Main loop for generating soundscapes
for iscape in range(NSCAPES):
    print(f"Generating soundscape: {iscape + 1}/{NSCAPES}")
    generate_soundscape(iscape)
