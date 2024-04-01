import os
import math
import random
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
    db2scale,
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


# Sound event classes for DCASE Challenge
__DCASE_SOUND_EVENT_CLASSES__ = {
    "femaleSpeech": 0,
    "maleSpeech": 1,
    "clapping": 2,
    "telephone": 3,
    "laughter": 4,
    "domesticSounds": 5,
    "footsteps": 6,
    "doorCupboard": 7,
    "music": 8,
    "musicInstrument": 9,
    "waterTap": 10,
    "bell": 11,
    "knock": 12,
}
__DCASE_LABEL_RATE__ = 10

Event = namedtuple(
    "Event",
    [
        "label",
        "source_file",
        "source_time",
        "event_time",
        "event_duration",
        "snr",
        "role",
        "pitch_shift",
        "time_stretch",
        "event_position",
    ],
)

# Paths for room SOFA files
__SPATIAL_SCAPER_RIRS_DIR__ = "spatialscaper_RIRs"
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


class Scaper:
    def __init__(
        self,
        duration=60,
        foreground_dir="",
        background_dir="",
        rir_dir="",
        room="metu",
        fmt="mic",
        sr=24000,
        DCASE_format=True,
        max_event_overlap=2,
        max_event_dur=10.0,
        ref_db=-60,
        speed_limit=1.5,
        max_sample_attempts=100,
    ):
        """
        Initializes a SpatialScaper object.

        Soundscapes are synthesized audio scenes with user-defined foreground and background sounds.
        TODO: support for user-defined background sounds still in progress.
        This class allows for detailed configuration of the soundscape's auditory scene, including
        spatial properties, acoustic characteristics, and compliance with DCASE challenge formats.

        Args:
            duration (float): The duration of the soundscape in seconds. Default is 60 seconds.
            foreground_dir (str): Directory path containing foreground sound files. Default is an empty string.
            background_dir (str): Directory path containing background sound files. Default is an empty string.
            rir_dir (str): Directory path containing Room Impulse Response (RIR) files to spatialize sound
                events in rooms. Default is an empty string.
            room (str): Identifier for the room where the scape will be simulated. Default is 'metu'.
            fmt (str): Output format specification, e.g., 'mic' for tetrahedral microphone. Default is 'mic'.
            sr (int): Sampling rate of the output audio in Hertz. Default is 24000 Hz.
            DCASE_format (bool): Flag to enable formatting of output labels for DCASE challenges compatibility.
                Default is True.
            max_event_overlap (int): Maximum number of events allowed to overlap at any point in time. Default is 2.
            max_event_dur (float): Maximum duration of any single sound event in seconds. Default is 10.0 seconds.
            ref_db (float): Reference level in decibels for the soundscape's overall loudness normalization.
                Default is -60 dB.
            speed_limit (float): Approximates the average speed at which a moving sound can travel in the room
                from a starting to an end point. Default is 1.5.
            max_sample_attempts (int): Maximum attempts to place a sound event at a specific point in time
                without exceeding max_event_overlap, before giving up . Default is 100.

        Attributes:
            fg_events (list): Initialized as an empty list to hold foreground event specifications.
            bg_events (list): Initialized as an empty list to hold background event specifications.
            fg_labels (dict): Maps foreground event labels to indices or DCASE class labels, based on DCASE_format.
            label_rate (int): The label sampling rate, defined only if DCASE_format is True.
        """

        self.duration = duration
        self.foreground_dir = foreground_dir
        self.background_dir = background_dir
        self.rir_dir = rir_dir
        self.room = room
        self.format = fmt
        self.sr = sr
        self.DCASE_format = DCASE_format
        if self.DCASE_format:
            self.label_rate = __DCASE_LABEL_RATE__
        self.max_event_overlap = max_event_overlap
        self.max_event_dur = max_event_dur
        self.ref_db = ref_db

        self.fg_events = []
        self.bg_events = []

        fg_label_list = get_label_list(self.foreground_dir)
        if self.DCASE_format:
            self.fg_labels = {
                l: __DCASE_SOUND_EVENT_CLASSES__[l] for l in fg_label_list
            }
        else:
            self.fg_labels = {l: i for i, l in enumerate(fg_label_list)}

        self.speed_limit = speed_limit

        self.max_sample_attempts = max_sample_attempts

    def add_background(self):
        """
        Adds a background event to the soundscape.
        This method sets fixed values for event time, duration, and
        SNR, and adds the event to the background events list.
        """
        event_time = ("const", 0)
        event_duration = ("const", self.duration)
        snr = ("const", 0)
        role = "background"
        pitch_shift = None
        time_stretch = None

        self.bg_events.append(
            Event(
                label=None,
                source_file=None,
                source_time=None,
                event_time=event_time[1],
                event_duration=event_duration[1],
                event_position=None,
                snr=snr[1],
                role="background",
                pitch_shift=pitch_shift,
                time_stretch=time_stretch,
            )
        )

    def add_event(
        self,
        label=("choose", []),
        source_file=("choose", []),
        source_time=("const", 0),
        event_time=None,
        event_position=("choose", ("uniform", None, None)),
        snr=("uniform"),
        split=None,
    ):
        """
        Adds a foreground event to the soundscape.

        Args:
            label (tuple): Specification for selecting the label of the event.
            source_file (tuple): Specification for selecting the source file of the event.
            source_time (tuple): Starting time of the event in the source file.
            event_time (tuple/None): Start time of the event in the soundscape.
            event_position (tuple): Specification for the position of the event in space.
            snr (tuple): Specification for the signal-to-noise ratio of the event.
            split (str/None): Specification for the split of the dataset.

        Handles random selection and validation of event parameters, including label, source file, and event time.

        Returns:
            None
        """
        # TODO: pitch_shift=(pitch_dist, pitch_min, pitch_max),
        # TODO: time_stretch=(time_stretch_dist, time_stretch_min, time_stretch_max))
        _DEFAULT_SNR_RANGE = (5, 30)

        if event_time is None:
            event_time = ("uniform", 0, self.duration)

        if label[0] == "choose" and label[1]:
            label_ = random.choice(label[1])
        elif label[0] == "choose":
            label_ = random.choice(list(self.fg_labels.keys()))
        elif label[0] == "const":
            label_ = label[1]

        if source_file[0] == "choose" and source_file[1]:
            source_file_ = random.choice(source_file[1])
        elif source_file[0] == "choose":
            source_file_ = random.choice(
                get_files_list(os.path.join(self.foreground_dir, label_), split)
            )

        if source_time[0] == "const":
            source_time_ = source_time[1]

        event_duration_ = librosa.get_duration(path=source_file_)
        if event_duration_ - source_time_ > self.max_event_dur:
            event_duration_ = self.max_event_dur
        event_time_ = self.define_event_onset_time(
            event_time,
            event_duration_,
            self.fg_events,
            self.max_event_overlap,
            1 / self.label_rate,
        )
        if event_time_ is None:
            warnings.warn(
                f'Could not find a start time for sound event "{source_file_}" that satisfies max_event_overlap = {self.max_event_overlap}. If this continues happening, you may want to consider adding less sound events to the scape or increasing max_event_overlap.'
            )
            if source_file[0] == "choose":
                # TODO: why does this warning only print once?
                warnings.warn("Randomly choosing a new sound event to try again.")
                self.add_event(
                    label,
                    source_file,
                    source_time,
                    event_time,
                    event_position,
                    snr,
                    split,
                )
            return None
        if self.DCASE_format:
            # round down to one decimal value
            event_time_ = (self.label_rate * event_time_ // 1) / self.label_rate

        if event_position[0] == "choose":
            moving = bool(random.getrandbits(1))
        else:
            moving = True if event_position[0] == "moving" else False
        if moving:  # currently the trajectory shape is randomly selected
            shape = "circular" if bool(random.getrandbits(1)) else "linear"
            if event_position[1][0] == "uniform" and moving:
                event_position_ = self.define_trajectory(
                    event_position[1],
                    int(event_duration_ / (1 / self.label_rate)),
                    shape,
                    event_duration_,
                    self.speed_limit,
                )
        else:
            xyz_min, xyz_max = self._get_room_min_max()
            event_position_ = [self._gen_xyz(xyz_min, xyz_max)]

        if snr[0] == "uniform" and len(snr) == 3:
            snr_ = random.uniform(*snr[1:])
        else:
            snr_ = random.uniform(*_DEFAULT_SNR_RANGE)

        self.fg_events.append(
            Event(
                label=label_,
                source_file=source_file_,
                source_time=source_time_,
                event_time=event_time_,
                event_duration=event_duration_,
                event_position=event_position_,
                snr=snr_,
                role="foreground",
                pitch_shift=None,
                time_stretch=None,
            )
        )

    def define_event_onset_time(
        self,
        event_time,
        event_duration,
        other_events,
        max_overlap,
        increment,
    ):
        """
        Finds a start time for an event ensuring it doesn't exceed a specified maximum overlap
        with other events in the soundscape.

        This method attempts to find an onset time for a new event, given its duration, such that the total overlap
        with existing events does not surpass a predefined maximum. It utilizes a specified increment to adjust
        the search granularity and a maximum number of attempts to find a suitable start time.

        Args:
            event_time (tuple): Specifies the method and range for selecting the event's start time.
                                Format is ("uniform", start_range, end_range) for a uniformly distributed
                                random selection, or ("const", value) for a fixed start time.
            event_duration (float): The duration of the event in seconds.
            other_events (list): A list of existing events in the soundscape, against which overlap is calculated.
            max_overlap (int): The maximum number of other events that the new event is allowed to overlap
                               with simultaneously.
            increment (float): The step size in seconds used to incrementally check for potential start times
                               within the specified range.

        Returns:
            float or None: The calculated start time for the event that meets the overlap criteria, or None
            if no suitable time could be found within the maximum number of attempts.
        """

        # Select a random start time within the range
        if event_time[0] == "uniform":
            _, start_range, end_range = event_time
            for _ in range(self.max_sample_attempts):
                random_start_time = random.uniform(
                    start_range, end_range - event_duration
                )
                if not new_event_exceeds_max_overlap(
                    random_start_time,
                    event_duration,
                    other_events,
                    max_overlap,
                    increment,
                ):
                    return random_start_time
            return None
        elif event_time[0] == "const":
            return event_time[1]

    def _gen_xyz(self, xyz_min, xyz_max):
        """
        Generates a random XYZ coordinate within specified bounds.

        Args:
            xyz_min (list/tuple): Minimum XYZ coordinates.
            xyz_max (list/tuple): Maximum XYZ coordinates.

        Returns:
            list: A randomly generated XYZ coordinate within the given bounds.
        """
        xyz = []
        for i in range(3):  # xyz
            xyz.append(random.uniform(xyz_min[i], xyz_max[i]))
        return xyz

    def _get_room_min_max(self):
        """
        Determines the minimum and maximum XYZ coordinates for the current room setup.

        Returns:
            tuple: A tuple containing the minimum and maximum XYZ coordinates for the room.
        """
        all_xyz = self.get_room_irs_xyz()
        xyz_min = all_xyz.min(axis=0)
        xyz_max = all_xyz.max(axis=0)
        return xyz_min, xyz_max

    def generate_end_point(
        self, xyz_start, xyz_min, xyz_max, speed_limit, event_duration
    ):
        """
        Generates a random end point for a moving sound event within specified spatial bounds,
        ensuring the movement complies with a given speed limit.

        This method calculates a random end point for an event, given its start point and duration,
        ensuring that the distance between the start and end points does not imply a speed exceeding
        the specified limit. The method accounts for three-dimensional space constraints by adhering
        to minimum and maximum bounds for each coordinate.

        Args:
                xyz_start (list or tuple): The starting coordinates (x, y, z) of the event.
                xyz_min (list or tuple): The minimum allowable coordinates (x, y, z) for the event's end point.
                xyz_max (list or tuple): The maximum allowable coordinates (x, y, z) for the event's end point.
                speed_limit (float): The maximum speed at which the event can move, in units per second.
                event_duration (float): The duration of the event, in seconds.

        Returns:
                list: The calculated end coordinates (x, y, z) of the event that adhere to the specified speed
          limit and spatial bounds.

        """
        # Calculate the maximum distance possible
        max_distance = speed_limit * event_duration

        # Helper function to calculate distance
        def distance(point1, point2):
            return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))

        # Generate a random end point within bounds that also complies with the speed limit
        while True:
            xyz_end = [
                random.uniform(min_val, max_val)
                for min_val, max_val in zip(xyz_min, xyz_max)
            ]
            if distance(xyz_start, xyz_end) <= max_distance:
                return xyz_end

    def define_trajectory(
        self, trajectory_params, npoints, shape, event_duration, speed_limit=1.5
    ):
        """
        Defines a trajectory for a moving sound event within specified spatial bounds,
        adhering to a given shape and speed limit.

        This method calculates a series of XYZ coordinates that outline the path of a
        sound event, based on the specified trajectory shape, the number of points to
        define the trajectory, and the bounds within which the trajectory must lie. It
        generates a starting point and an end point that comply with the specified speed
        limit over the event's duration, and then interpolates between these points
        according to the trajectory's shape.

        Args:
            trajectory_params (tuple): Parameters defining the trajectory's spatial bounds,
                                       structured as (param_description, xyz_min, xyz_max).
                                       'param_description' is not directly used but indicates
                                       the nature of the bounds.
            npoints (int): The number of points to be used in defining the trajectory,
                           influencing the granularity of the path.
            shape (str): The shape of the trajectory. Supported shapes include 'circular',
                         'linear', etc., which determine how the path is interpolated between
                         start and end points.
            event_duration (float): The duration of the event, in seconds, which, along with
                                    the speed limit, influences the maximum allowable distance
                                    between the start and end points.
            speed_limit (float): The maximum speed at which the event can move, in units per
                                 second. Default is 1.5. This parameter, combined with the event
                                 duration, constrains the end point generation.

        Returns:
            list: A list of XYZ coordinates defining the trajectory. Each element of the list
                  is a tuple or list representing the XYZ coordinates at a point along the trajectory.
        """

        if all(trajectory_params[1:]):
            xyz_min, xyz_max = trajectory_params[1:]
        else:
            xyz_min, xyz_max = self._get_room_min_max()
        xyz_start = self._gen_xyz(xyz_min, xyz_max)
        xyz_end = self.generate_end_point(
            xyz_start, xyz_min, xyz_max, speed_limit, event_duration
        )
        return generate_trajectory(xyz_start, xyz_end, npoints, shape)

    def define_position(self, position_params):
        """
        Defines a position for a sound event.

        Args:
            position_params (tuple/list): Parameters defining the position bounds.

        Returns:
            list: A list containing the XYZ coordinates of the defined position.
        """
        # TODO: make this work for other distributions
        if position_params:
            xyz_min, xyz_max = position_params
        else:
            xyz_min, xyz_max = self._get_room_min_max()
        return [self._gen_xyz(xyz_min, xyz_max)]

    def get_room_irs_xyz(self):
        """
        Retrieves the XYZ coordinates of impulse response positions in the room.

        Returns:
            numpy.ndarray: An array of XYZ coordinates for the impulse response positions.
        """
        room_sofa_path = os.path.join(
            self.rir_dir, __SPATIAL_SCAPER_RIRS_DIR__, __ROOM_RIR_FILE__[self.room]
        )
        return load_pos(room_sofa_path, doas=False)

    def get_room_irs_wav_xyz(self, wav=True, pos=True):
        """
        Retrieves impulse responses and their positions for the room.

        Args:
            wav (bool): Whether to include the waveforms of the impulse responses.
            pos (bool): Whether to include the positions of the impulse responses.

        Returns:
            tuple: A tuple containing the impulse responses, their sampling rate, and their XYZ positions.
        """
        room_sofa_path = os.path.join(
            self.rir_dir, __SPATIAL_SCAPER_RIRS_DIR__, __ROOM_RIR_FILE__[self.room]
        )
        all_irs, ir_sr, all_ir_xyzs = load_rir_pos(room_sofa_path, doas=False)
        ir_sr = ir_sr.data[0]
        all_irs = all_irs.data
        all_ir_xyzs = all_ir_xyzs.data
        if ir_sr != self.sr:
            all_irs = librosa.resample(all_irs, orig_sr=ir_sr, target_sr=self.sr)
            ir_sr = self.sr
        return all_irs, ir_sr, all_ir_xyzs

    def get_format_irs(self, all_irs, fmt="mic"):
        """
        Retrieves impulse responses according to the specified format.

        Args:
            all_irs (numpy.ndarray): Array of all impulse responses.
            fmt (str): The format for retrieving impulse responses (e.g., 'mic').

        Returns:
            numpy.ndarray: An array of impulse responses formatted according to the specified format.
        """
        if fmt == "mic" and self.room == "metu":
            return all_irs[:, [5, 9, 25, 21], :]
        else:
            return all_irs

    def generate_noise(self, audio):
        """
        Generates and adds noise to the provided audio based on the background events.

        Args:
            audio (numpy.ndarray): The audio to which noise will be added.

        Returns:
            numpy.ndarray: The audio with added noise.
        """
        for event in self.bg_events:
            if not event.source_file:
                audio += db2scale(self.ref_db + event.snr) * np.random.normal(
                    0, 1, (int(event.event_duration * self.sr), self.nchans)
                )
        return audio

    def synthesize_events_and_labels(self, all_irs, all_ir_xyzs, out_audio):
        """
        Synthesizes audio events based on foreground events and their spatial trajectories,
        and generates corresponding labels for each event.

        This method processes each foreground event to spatialize its audio according to
        the impulse response (IR) corresponding to its trajectory. It then normalizes and
        blends these spatialized audio snippets into the output audio. Additionally, it
        generates precise labels for each event indicating its time, location, class,
        and source ID.

        Args:
            all_irs (numpy.ndarray): An array of impulse responses for the room.
            all_ir_xyzs (numpy.ndarray): An array of XYZ coordinates corresponding to each impulse response.
            out_audio (numpy.ndarray): The initial audio array to which the spatialized event audio will be added.

        Returns:
            tuple: A tuple containing the synthesized audio with all events and a matrix of labels for each event.
                   The labels include time, spatial coordinates, class ID, and source ID for each audio event.

        Detailed Process:
            1. Iterates over each foreground event.
            2. Retrieves the impulse response indices and their corresponding XYZ coordinates based on the event's trajectory.
            3. Normalizes the impulse responses and the event's audio signal.
            4. Spatializes the event's audio using the normalized impulse responses.
            5. Scales the spatialized audio based on the event's signal-to-noise ratio (SNR).
            6. Adds the scaled, spatialized audio to the output audio at the correct onset time.
            7. Generates a time grid and labels for the spatialized audio, indicating the event's location and class at each time point.
            8. Trims the spatialized audio to match the length of the labels.
            9. Aggregates all labels and sorts them chronologically.

        The method ensures that the spatialized audio of each event is correctly aligned in time and space within the soundscape,
        and that the labels accurately reflect the temporal and spatial characteristics of each event.
        """

        all_labels = []
        for ievent, event in enumerate(self.fg_events):
            # fetch trajectory from irs
            ir_idx = traj_2_ir_idx(all_ir_xyzs, event.event_position)
            irs = all_irs[ir_idx]
            ir_xyzs = all_ir_xyzs[ir_idx]
            # remove repeated positions
            ir_idx = find_indices_of_change(ir_xyzs)
            irs = irs[ir_idx]
            ir_xyzs = ir_xyzs[ir_idx]

            # load and normalize audio signal by its norm
            x, _ = librosa.load(event.source_file, sr=self.sr)
            x = x[: int(event.event_duration * self.sr)]
            x = x / np.max(np.abs(x))

            # normalize irs to have unit energy
            norm_irs = IR_normalizer(irs)

            # SPATIALIZE
            norm_irs = np.transpose(norm_irs, (2, 1, 0))
            if len(irs) > 1:
                ir_times = np.linspace(0, event.event_duration, len(irs))
                xS = spatialize(x, norm_irs, ir_times, sr=self.sr, s=event.snr)
            else:
                ir_times = np.linspace(0, event.event_duration, len(irs) + 1)
                ir_xyzs = np.concatenate([ir_xyzs, ir_xyzs])
                xS = []
                for i in range(norm_irs.shape[1]):
                    _x = scipy.signal.convolve(
                        x, np.squeeze(norm_irs[:, i]), mode="full", method="fft"
                    )
                    xS.append(_x)
                xS = np.array(xS).T
                xS = xS[: len(x)]

            # standardize the spatialized audio
            event_scale = db2scale(self.ref_db + event.snr)
            xS = (event_scale / np.std(xS)) * xS

            # add to out_audio
            onsamp = int(event.event_time * self.sr)
            out_audio[onsamp : onsamp + len(xS)] += xS

            # generate ground truth
            time_grid = get_timegrid(
                len(xS),
                self.sr,
                ir_times,
                time_grid_resolution=round(1 / self.label_rate, 1),
            )
            labels = get_labels(
                ir_times,
                time_grid,
                ir_xyzs,
                class_id=self.fg_labels[event.label],
                source_id=ievent,
            )
            labels[:, 0] = labels[:, 0] + int(event.event_time * self.label_rate)
            xS = xS[
                : int(time_grid[-1] * self.sr)
            ]  # trim audio signal to exactly match labels
            all_labels.append(labels)

        labels = sort_matrix_by_columns(np.vstack(all_labels))

        return out_audio, labels

    def generate(self, audiopath, labelpath):
        """
        Generates the final soundscape audio and corresponding labels, then saves them to specified paths.

        This method combines all background and foreground events, spatializes them according to the room's impulse responses,
        and creates a final audio mix. It also generates a comprehensive set of labels for all events in the soundscape.
        The final audio and labels are then saved to the given paths.

        Args:
            audiopath (str): File path where the synthesized soundscape audio will be saved.
            labelpath (str): File path where the labels for the soundscape will be saved.

        Process:
            1. Fetches and formats the room impulse responses (IRs) and their XYZ coordinates.
            2. Initializes an empty audio array for the output soundscape.
            3. Adds background noise to the output audio.
            4. Sorts the foreground events by their onset time.
            5. Calls `synthesize_events_and_labels` to process each foreground event,
               spatialize its audio, and generate labels.
            6. Saves the synthesized soundscape audio and the labels to the specified paths.

        The method ensures that all components of the soundscape are correctly synthesized and spatialized,
        and that the output audio and labels are accurately saved for further use or analysis.
        """

        all_irs, ir_sr, all_ir_xyzs = self.get_room_irs_wav_xyz()
        all_irs = self.get_format_irs(all_irs)
        self.nchans = all_irs.shape[1]  # a bit ugly but works for now

        # initialize output audio array
        out_audio = np.zeros((int(self.duration * self.sr), self.nchans))

        # add background noise
        out_audio = self.generate_noise(out_audio)

        # sort foreground events by onset time
        self.fg_events = sorted(self.fg_events, key=lambda x: x.event_time)

        out_audio, labels = self.synthesize_events_and_labels(
            all_irs, all_ir_xyzs, out_audio
        )

        # save output
        save_output(audiopath, labelpath, out_audio, self.sr, labels)
