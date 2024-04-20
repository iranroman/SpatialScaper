import os
import glob

import librosa

# Local application/library specific imports
from .sofa_utils import load_rir_pos, load_pos


# Paths for room SOFA files
__SPATIAL_SCAPER_RIRS_DIR__ = "spatialscaper_RIRs"
__PATH_TO_AMBIENT_NOISE_FILES__ = os.path.join("source_data", "TAU-SNoise_DB")
__ROOM_RIR_FILE__ = {
    "metu": "metu_sparg_{fmt}.sofa",
    "arni": "arni_{fmt}.sofa",
    "bomb_shelter": "bomb_shelter_{fmt}.sofa",
    "gym": "gym_{fmt}.sofa",
    "pb132": "pb132_{fmt}.sofa",
    "pc226": "pc226_{fmt}.sofa",
    "sa203": "sa203_{fmt}.sofa",
    "sc203": "sc203_{fmt}.sofa",
    "se203": "se203_{fmt}.sofa",
    "tb103": "tb103_{fmt}.sofa",
    "tc352": "tc352_{fmt}.sofa",
}


class BaseRoom:
    """
    Initialize a Room object.

    A Room encapsulates the spatial and acoustic characteristics available of a physical room.
    This includes a collection of impulse response measurements taken at different positions in
    the room.
    """

    def __init__(self) -> None:
        pass

    def get_ambient_noise_paths(self):
        """
        Retrieves paths to ambient noise audio files specific to this room.

        Returns:
            list[str]: A list of audio paths.
        """
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
        """Path to the SOFA file for this room."""
        return os.path.join(
            self.rir_dir,
            __SPATIAL_SCAPER_RIRS_DIR__,
            __ROOM_RIR_FILE__[self.room].format(fmt=self.format),
        )

    def get_ambient_noise_paths(self):
        # list files from disk
        path_to_ambient_noise_files = os.path.join(
            self.rir_dir, __PATH_TO_AMBIENT_NOISE_FILES__
        )
        ambient_noise_format_files = glob.glob(
            os.path.join(path_to_ambient_noise_files, "*", "*")
        )

        # translate format
        fmt = self.format
        if self.format == "mic":
            fmt = "tetra"
        room = self.room
        if self.room == "bomb_shelter":
            room = "bomb_center"

        # filter files
        ambient_noise_format_files = [
            f for f in ambient_noise_format_files if fmt in os.path.basename(f)
        ]
        room_ambient_noise_files = [
            f for f in ambient_noise_format_files if room in os.path.basename(f)
        ]

        # assert len(room_ambient_noise_files) < 2
        return room_ambient_noise_files or ambient_noise_format_files

    def get_positions(self):
        return load_pos(self.sofa_path, doas=False)

    def get_irs(self, sr=None, format=True):
        all_irs, ir_sr, all_ir_xyzs = load_rir_pos(self.sofa_path, doas=False)
        ir_sr = ir_sr.data[0]
        all_irs = all_irs.data
        all_ir_xyzs = all_ir_xyzs.data
        if sr is not None and ir_sr != sr:
            all_irs = librosa.resample(all_irs, orig_sr=ir_sr, target_sr=sr)
            ir_sr = sr
        return all_irs, ir_sr, all_ir_xyzs


def get_room(rir_dir, *a, **kw):
    if isinstance(rir_dir, BaseRoom):
        return rir_dir
    return SOFARoom(rir_dir, *a, **kw)