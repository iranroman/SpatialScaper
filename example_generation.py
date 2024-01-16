import numpy as np
import scipy
import librosa
import os
import spatialscaper as ss
import matplotlib.pyplot as plt

NSCAPES = 10
FOREGROUND_DIR = "datasets/FSD50K_DCASE"
BACKGROUND_DIR = ""  # TODO: define a set of default background tracks
ROOM = "metu"
FORMAT = "mic"
MIN_EVENTS = 3
MAX_EVENTS = 8
DURATION = 60.0  # seconds
SR = 24000
OUTPUT_DIR = "output"


for iscape in range(NSCAPES):
    print("Generating soundscape: {:d}/{:d}".format(iscape + 1, NSCAPES))

    TRACK_NAME = f"fold5_room1_mix00{iscape+1}"
    # create a spatial scaper
    ssc = ss.Scaper(DURATION, FOREGROUND_DIR, BACKGROUND_DIR, ROOM, FORMAT, SR)

    # add a random number of foreground events
    n_events = np.random.randint(MIN_EVENTS, MAX_EVENTS + 1)
    for _ in range(n_events):
        ssc.add_event()

    audiofile = os.path.join(OUTPUT_DIR, FORMAT, TRACK_NAME)
    labelfile = os.path.join(OUTPUT_DIR, 'labels', TRACK_NAME)

    ssc.generate(audiofile,labelfile)
