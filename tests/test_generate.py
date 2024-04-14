import os
import pathlib
import warnings
import numpy as np
import spatialscaper as ss
import librosa
import pytest

HERE = pathlib.Path(__file__).parent
DATA_DIR = HERE/"data"
OUTPUT_DIR = HERE/"output"


def compare_audio(gen_file, ref_file):
    if not os.path.isfile(ref_file):
        warnings.warn(f"reference {os.path.basename(ref_file)} does not exist.")
        return
    yp, srp = librosa.load(gen_file, sr=None, mono=False)
    yt, srt = librosa.load(ref_file, sr=None, mono=False)
    assert srp == srt, "sample rates do not match"
    assert yp.shape == yt.shape, "audio shapes do not match"
    assert np.allclose(yp, yt), "audio content does not match"


@pytest.mark.parametrize("room", ["bomb_shelter"])
@pytest.mark.parametrize("fmt", ["mic", "foa"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_end2end(room, fmt, seed):
    ss.utils.set_seed(seed)
    
    ssc = ss.Scaper(
        duration=30,
        foreground_dir="datasets/sound_event_datasets/FSD50K_FMA",
        rir_dir="datasets/rir_datasets",
        room=room,
        fmt=fmt,
        max_event_overlap=2,
        speed_limit=2.0,  # in meters per second
        ref_db=-65,
    )

    ssc.add_background()
    for _ in range(max(int(np.random.normal(4, 5)), 1)):
        ssc.add_event()

    track_name = f"room-{room}_fmt-{fmt}_seed-{seed:03d}"
    audiofile = OUTPUT_DIR / "end2end" / fmt / track_name
    labelfile = OUTPUT_DIR / "end2end" / f"{fmt}_labels" / track_name
    ssc.generate(audiofile, labelfile)

    refaudiofile = DATA_DIR / "end2end" / fmt / track_name
    compare_audio(f'{audiofile}.wav', f'{refaudiofile}.wav')
