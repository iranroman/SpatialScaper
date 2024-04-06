import os
import math
import random
import glob
from collections import namedtuple

import librosa
import scipy
import numpy as np
import warnings

# Local application/library specific imports
from .utils import (
    get_label_list,
    get_files_list,
    new_event_exceeds_max_overlap,
    count_leading_zeros_in_period,
    generate_trajectory,
    db2multiplier,
    traj_2_ir_idx,
    find_indices_of_change,
    IR_normalizer,
    spatialize,
    get_timegrid,
    get_labels,
    save_output,
    sort_matrix_by_columns,
)
from .sofa_utils import load_rir_pos, load_pos


# Paths for room SOFA files
__SPATIAL_SCAPER_RIRS_DIR__ = "spatialscaper_RIRs"
__PATH_TO_AMBIENT_NOISE_FILES__ = os.path.join("source_data", "TAU-SNoise_DB")
__ROOM_RIR_FILE__ = {
    "metu": "metu_sparg_em32.sofa",
    "arni": "arni_mic.sofa",
    "bomb_shelter": "bomb_shelter_mic.sofa",
    "gym": "gym_mic.sofa",
    "pb132": "pb132_mic.sofa",
    "pc226": "pc226_mic.sofa",
    "sa203": "sa203_mic.sofa",
    "sc203": "sc203_mic.sofa",
    "se203": "se203_mic.sofa",
    "tb103": "tb103_mic.sofa",
    "tc352": "tc352_mic.sofa",
}


class BaseRoom:
    def __init__(self) -> None:
        pass

    def get_room_ambient_noise(self):
        raise NotImplementedError

    def get_positions(self):
        """
        Retrieves the XYZ coordinates of impulse response positions in the room.

        Returns:
            numpy.ndarray: An array of XYZ coordinates for the impulse response positions.
        """
        raise NotImplementedError

    def get_irs(self, format=True):
        """
        Retrieves impulse responses and their positions for the room.

        Args:
            wav (bool): Whether to include the waveforms of the impulse responses.
            pos (bool): Whether to include the positions of the impulse responses.

        Returns:
            tuple: A tuple containing the impulse responses, their sampling rate, and their XYZ positions.
        """
        raise NotImplementedError

    def get_boundaries(self):
        """
        Determines the minimum and maximum XYZ coordinates for the current room setup.

        Returns:
            tuple: A tuple containing the minimum and maximum XYZ coordinates for the room.
        """
        all_xyz = self.get_positions()
        xyz_min = all_xyz.min(axis=0)
        xyz_max = all_xyz.max(axis=0)
        return xyz_min, xyz_max


class SOFARoom(BaseRoom):
    def __init__(self, rir_dir, room, fmt):
        self.rir_dir = rir_dir
        self.room = room
        self.format = fmt

    @property
    def sofa_path(self):
        return os.path.join(
            self.rir_dir, __SPATIAL_SCAPER_RIRS_DIR__, __ROOM_RIR_FILE__[self.room]
        )

    def get_ambient_noise_paths(self):
        path_to_ambient_noise_files = os.path.join(
            self.rir_dir, __PATH_TO_AMBIENT_NOISE_FILES__
        )
        ambient_noise_format_files = glob.glob(
            os.path.join(path_to_ambient_noise_files, "*", "*")
        )
        if self.format == "mic":
            ambient_noise_format_files = [
                f for f in ambient_noise_format_files if "tetra" in f
            ]
        elif self.format == "foa":
            ambient_noise_format_files = [
                f for f in ambient_noise_format_files if "foa" in f
            ]
        if self.room == "bomb_shelter":
            room_ambient_noise_file = [
                f for f in ambient_noise_format_files if "bomb_center" in f
            ]
        else:
            room_ambient_noise_file = [
                f for f in ambient_noise_format_files if self.room in f
            ]
        assert len(room_ambient_noise_file) < 2
        if room_ambient_noise_file:
            return room_ambient_noise_file
        return ambient_noise_format_files
        
    def get_positions(self):
        return load_pos(self.sofa_path, doas=False)
    
    def get_irs(self, format=True):
        all_irs, ir_sr, all_ir_xyzs = load_rir_pos(self.sofa_path, doas=False)
        ir_sr = ir_sr.data[0]
        all_irs = all_irs.data
        all_ir_xyzs = all_ir_xyzs.data
        if ir_sr != self.sr:
            all_irs = librosa.resample(all_irs, orig_sr=ir_sr, target_sr=self.sr)
            ir_sr = self.sr
        if format:
            self._format_irs(all_irs)
        return all_irs, ir_sr, all_ir_xyzs
    
    def _format_irs(self, all_irs, fmt="mic"):
        if fmt == "mic" and self.room == "metu":
            return all_irs[:, [5, 9, 25, 21], :]
        return all_irs





def get_room(rir_dir, *a, **kw):
    if isinstance(rir_dir, BaseRoom):
        return rir_dir
    return SOFARoom(rir_dir, *a, **kw)