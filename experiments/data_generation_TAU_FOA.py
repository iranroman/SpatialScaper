import numpy as np
import spatialscaper as ss
import os

# Constants
NSCAPES_PER_ROOM = 133  # Number of soundscapes to generate per room
FOREGROUND_DIR = "datasets/sound_event_datasets/FSD50K_FMA"  # Directory with FSD50K foreground sound files
RIR_DIR = "datasets/rir_datasets"  # Directory containing Room Impulse Response (RIR) files
FORMAT = "foa"  # Output format specifier
N_EVENTS_MEAN = 15  # Mean number of foreground events in a soundscape
N_EVENTS_STD = 6  # Standard deviation of the number of foreground events
DURATION = 60.0  # Duration in seconds of each soundscape, customizable by the user
SR = 24000  # SpatialScaper default sampling rate for the audio files
OUTPUT_DIR = "FOA_TAU"  # Directory to store the generated soundscapes

ROOMS = ["bomb_shelter", "gym", "pb132", "pc226", "sa203", "sc203", "se203", "tb103", "tc352"]

def generate_soundscape(room, index, room_number):
    ref_db = np.random.uniform(-70, -50)  # Randomly determine the ref_db
    track_name = f"fold1_room{room_number}_mix{index+1:03d}"
    # Initialize Scaper
    ssc = ss.Scaper(
        DURATION,
        FOREGROUND_DIR,
        RIR_DIR,
        FORMAT,
        room,
        max_event_overlap=2,
        speed_limit=2.0,
    )
    ssc.ref_db = ref_db

    # static white noise in this example
    ssc.add_background()

    # Add a random number of foreground events
    n_events = int(np.random.normal(N_EVENTS_MEAN, N_EVENTS_STD))
    n_events = n_events if n_events > 0 else 1

    for _ in range(n_events):
        ssc.add_event()

    audiofile = os.path.join(OUTPUT_DIR, FORMAT, track_name)
    labelfile = os.path.join(OUTPUT_DIR, "labels", track_name)

    ssc.generate(audiofile, labelfile)

# Main loop for generating soundscapes for each room
for room_number, room in enumerate(ROOMS, start=1):
    for iscape in range(NSCAPES_PER_ROOM):
        print(f"Generating soundscape: {iscape + 1}/{NSCAPES_PER_ROOM} for room {room}")
        generate_soundscape(room, iscape, room_number)
