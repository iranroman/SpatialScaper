import os
import random
from collections import namedtuple

import librosa
import numpy as np

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
__ROOM_RIR_FILE__ = {"metu": "metu_sparg.sofa"}

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
        label_rate=10,
        max_event_overlap=2,
        ref_db=-12,
    ):
        """
        Initializes a Scaper object for generating soundscapes.

        Args:
            duration (float): The duration of the soundscape in seconds.
            foreground_dir (str): Path to the directory containing foreground sound files.
            background_dir (str): Path to the directory containing background sound files.
            sofa_dir (str): Path to the directory containing SOFA files for room impulse responses.
            room (str): The name of the room for which the soundscape is being generated.
            fmt (str): Format of the output (e.g., 'mic' for microphone format).
            sr (int): Sampling rate for the audio.
            DCASE_format (bool): Whether to format output for DCASE challenges.
            label_rate (int): The rate at which labels are generated.
            max_event_overlap (int): Maximum allowed overlap between events.
            ref_db (float): Reference decibel level.

        """

        self.duration = duration
        self.foreground_dir = foreground_dir
        self.background_dir = background_dir
        self.rir_dir = rir_dir
        self.room = room
        self.format = fmt
        self.sr = sr
        self.DCASE_format = DCASE_format
        self.label_rate = label_rate
        self.max_event_overlap = max_event_overlap
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
        event_duration=None,
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
            event_duration (float/None): Duration of the event.
            event_position (tuple): Specification for the position of the event in space.
            snr (tuple): Specification for the signal-to-noise ratio of the event.
            split (str/None): Specification for the split of the dataset.

        Handles random selection and validation of event parameters, including label, source file, and event time.
        """
        # TODO: pitch_shift=(pitch_dist, pitch_min, pitch_max),
        # TODO: time_stretch=(time_stretch_dist, time_stretch_min, time_stretch_max))
        DEFAULT_SNR_RANGE = (5, 30)

        if event_time is None:
            event_time = ("uniform", 0, self.duration)

        if label[0] == "choose" and label[1]:
            label = random.choice(label[1])
        elif label[0] == "choose":
            label = random.choice(list(self.fg_labels.keys()))

        if source_file[0] == "choose" and source_file[1]:
            source_file = random.choice(source_file[1])
        elif source_file[0] == "choose":
            source_file = random.choice(
                get_files_list(os.path.join(self.foreground_dir, label), split)
            )

        if source_time[0] == "const":
            source_time = source_time[1]

        event_duration = librosa.get_duration(filename=source_file)
        event_time = self.define_event_onset_time(
            event_time,
            event_duration,
            self.fg_events,
            self.max_event_overlap,
            1 / self.label_rate,
        )
        if self.DCASE_format:
            event_time = round(
                event_time, count_leading_zeros_in_period(self.label_rate) + 1
            )

        if event_position[0] == "choose":
            moving = bool(random.getrandbits(1))
        else:
            moving = True if event_position[0] == "moving" else False
        if moving:
            shape = "circular" if bool(random.getrandbits(1)) else "linear"
        if event_position[1][0] == "uniform" and moving:
            event_position = self.define_trajectory(
                event_position[1], int(event_duration / (1 / self.label_rate)), shape
            )

        if snr[0] == "uniform" and len(snr) == 3:
            snr = random.uniform(*snr[1:])
        else:
            snr = random.uniform(*DEFAULT_SNR_RANGE)  # default SNR range

        self.fg_events.append(
            Event(
                label=label,
                source_file=source_file,
                source_time=source_time,
                event_time=event_time,
                event_duration=event_duration,
                event_position=event_position,
                snr=snr,
                role="foreground",
                pitch_shift=None,
                time_stretch=None,
            )
        )

    def define_event_onset_time(
        self, event_time, event_duration, other_events, max_overlap, increment
    ):
        """
        Recursively finds a start time for an event that doesn't exceed the maximum overlap with other events.

        Args:
            event_time (tuple): Specification of the event's start time.
            event_duration (float): Duration of the event.
            other_events (list): List of other events in the soundscape.
            max_overlap (int): Maximum allowed overlap with other events.
            increment (float): Incremental step for checking overlap.

        Returns:
            float: A start time for the event that satisfies the overlap constraint.
        """

        # Select a random start time within the range
        if event_time[0] == "uniform":
            _, start_range, end_range = event_time
            random_start_time = random.uniform(start_range, end_range - event_duration)

        # Check if the selected time overlaps with more than max_overlap events
        if new_event_exceeds_max_overlap(
            random_start_time, event_duration, other_events, max_overlap, increment
        ):
            # If it does overlap, recursively try again
            return self.define_event_onset_time(
                event_time, event_duration, other_events, max_overlap, increment
            )
        else:
            # If it doesn't overlap, return the selected start time
            return random_start_time

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

    def define_trajectory(self, trajectory_params, npoints, shape):
        """
        Defines a trajectory for a moving sound event.

        Args:
            trajectory_params (tuple): Parameters defining the trajectory bounds.
            npoints (int): Number of points to define the trajectory.
            shape (str): The shape of the trajectory (e.g., 'circular', 'linear').

        Returns:
            list: A list of XYZ coordinates defining the trajectory.
        """

        if all(trajectory_params[1:]):
            xyz_min, xyz_max = trajectory_params[1:]
        else:
            xyz_min, xyz_max = self._get_room_min_max()
        xyz_start = self._gen_xyz(xyz_min, xyz_max)
        xyz_end = self._gen_xyz(xyz_min, xyz_max)
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
        room_sofa_path = os.path.join(self.rir_dir, __ROOM_RIR_FILE__[self.room])
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
        room_sofa_path = os.path.join(self.rir_dir, __ROOM_RIR_FILE__[self.room])
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
            x = x / np.max(np.abs(x))

            # normalize irs to have unit energy
            norm_irs = IR_normalizer(irs)

            # SPATIALIZE
            norm_irs = np.transpose(norm_irs, (2, 1, 0))
            if len(irs) > 1:
                ir_times = np.linspace(0, event.event_duration, len(irs))
                xS = spatialize(x, norm_irs, ir_times, sr=self.sr, s=event.snr)
            else:
                # TODO: implement static
                continue

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