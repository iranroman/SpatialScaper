import random
import os
import librosa

from .utils import get_label_list
from .utils import get_files_list
from .utils import new_event_exceeds_max_overlap
from .utils import count_leading_zeros_in_period

# dcase.community/challenge2023/task-sound-event-localization-and-detection-evaluated-in-real-spatial-sound-scenes#sound-event-classes        
__DCASE_SOUND_EVENT_CLASSES__ = {
    'femaleSpeech': 0,
    'maleSpeech':1,
    'clapping':2,
    'telephone':3,
    'laughter':4,
    'domesticSounds':5,
    'footsteps':6,
    'doorCupboard':7,
    'music':8,
    'musicInstrument':9,
    'waterTap':10,
    'bell':11,
    'knock':12,
}

class Scaper:
    def __init__(self,
        duration=60,
        foreground_dir="datasets/FSD50K_DCASE",
        background_dir="",
        room="metu",
        fmt="mic",
        sr=24000,
        DCASE_format=True,
        label_rate = 10,
        max_event_overlap = 2
    ):
        self.duration = duration
        self.foreground_dir = foreground_dir
        self.background_dir = background_dir
        self.room = room
        self.format = fmt
        self.sr = sr
        self.DCASE_format=DCASE_format
        self.label_rate = label_rate
        self.max_event_overlap = max_event_overlap

        self.fg_events = []

        fg_label_list = get_label_list(self.foreground_dir)
        if self.DCASE_format:
            self.fg_labels = {l:__DCASE_SOUND_EVENT_CLASSES__[l] for l in fg_label_list}
        else:
            self.fg_labels = {l:i for i,l in enumerate(fg_label_list)}

    def add_event(self,
            label=('choose',[]),
            source_file=('choose',[]),
            source_time=('const',0),
            event_time=None,
            event_duration=None,
            event_position=('uniform',[]),
            snr=('uniform',0.5,1),
            split=None,
            ):
        # TODO: make snr be actual snr. Currently a linear scale factor
        # TODO: pitch_shift=(pitch_dist, pitch_min, pitch_max),
        # TODO: time_stretch=(time_stretch_dist, time_stretch_min, time_stretch_max))
        if event_time is None:
            event_time = ('uniform',0,self.duration)

        if label[0] == 'choose' and label[1]:
            label = random.choice(label[1])
        elif label[0] == 'choose':
            label = random.choice(list(self.fg_labels.keys()))

        if source_file[0] == 'choose' and source_file[1]:
            source_file = random.choice(source_file[1])
        elif source_file[0] == 'choose':
            source_file = random.choice(get_files_list(os.path.join(self.foreground_dir,label),split))

        if source_time[0] == 'const':
            source_time = source_time[1]

        event_duration = librosa.get_duration(path=source_file)
        event_time = self.define_event_onset_time(event_time,event_duration,self.fg_events,self.max_event_overlap,1/self.label_rate)
        if self.DCASE_format:
            event_time=round(event_time,count_leading_zeros_in_period(self.label_rate)+1)

        if len(event_position) == 2:
            if not event_position[1]:
                moving = True 
                # TODO: implement bool(random.getrandbits(1))
            if event_position[0] == 'choose' and event_position[1]:
                event_position = random.choice(event_position[1])
            elif event_position[0] == 'uniform' and moving:
                event_position = self.define_trajectory([],int(event_duration/(1/self.label_rate)))
            elif event_position[0] == 'uniform':
                event_position = self.define_position([])

        if snr[0] == 'uniform':
            snr = random.uniform(*snr[1:])

        self.fg_events.append(Event(label=label,
								source_file=source_file,
								source_time=source_time,
								event_time=event_time,
								event_duration=event_duration,
                                event_position=event_position,
								snr=snr,
                                role='foreground',
                                pitch_shift=None,
                                time_stretch=None))
    
    def define_event_onset_time(self,event_time, event_duration, other_events, max_overlap, increment):
        """ Recursively find a start time for the event that doesn't exceed max_overlap. """

        # Select a random start time within the range
        if event_time[0] == 'uniform':
            _, start_range, end_range = event_time
            random_start_time = random.uniform(start_range, end_range - event_duration)

        # Check if the selected time overlaps with more than max_overlap events
        if new_event_exceeds_max_overlap(random_start_time, event_duration, other_events, max_overlap, increment):
            # If it does overlap, recursively try again
            return define_event_onset_time(event_time, event_duration, other_events, max_overlap, increment)
        else:
            # If it doesn't overlap, return the selected start time
            return random_start_time

    def _gen_xyz(self,xyz_min,xyz_max):
        xyz = []
        for i in range(3): #xyz
            xyz.append(random.uniform(xyz_min[i],xyz_max[i]))
        return xyz

    def _get_room_min_max(self):
        all_xyz = self.get_room_raw_irs(wav=False)
        xyz_min = all_xyz.min(axis=0)
        xyz_max = all_xyz.max(axis=0)
        return xyz_min, xyz_max


    def define_trajectory(self,trajectory_params,npoints,shapes=['linear','circular']):
        # TODO: make this work for other distributions
        if len(trajectory_params)>1:
            xyz_min, xyz_max = trajectory_params[:2]
            shape = trajectory_params[2]
        else:
            xyz_min, xyz_max = self._get_room_min_max()
        if trajectory_params and len(trajectory_params) == 1:
            shape = trajectory_params[0]
        else:
            shape = random.choice(shapes)
        xyz_start = self._gen_xyz(xyz_min,xyz_max)
        xyz_end = self._gen_xyz(xyz_min,xyz_max)
        return generate_trajectory(xyz_start, xyz_end, 25, shape)

    def define_position(self,position_params):
        # TODO: make this work for other distributions
        if position_params:
            xyz_min, xyz_max = position_params
        else:
            xyz_min, xyz_max = self._get_room_min_max()
        return [self._gen_xyz(xyz_min,xyz_max)]

    def get_room_raw_irs(self,wav=True,pos=True):
        # fetch all irs and resample if needed
        if wav and pos:
            all_irs, ir_sr, all_ir_xyzs = load_rir_pos(GET_ROOM_SOFA_PATH[self.room])
            if ir_sr != self.sr:
                all_irs = librosa.resample(all_irs, orig_sr=ir_sr, target_sr=self.sr)
                ir_sr = self.sr
            return all_irs, ir_sr, all_ir_xyzs
        if pos:
            return load_pos(GET_ROOM_SOFA_PATH[self.room])

    def get_format_irs(self,all_irs, fmt='mic'):
        if fmt == 'mic' and self.room == 'metu':
            return all_irs[:,[5,9,25,21],:]
        else:
            return all_irs


    def generate(self, audiopath, labelpath):

        all_irs, ir_sr, all_ir_xyzs = self.get_room_raw_irs()
        all_irs = self.get_format_irs(all_irs)

        out_audio = 0.001*np.random.normal(0,1,(int(self.duration*self.sr),4))

        all_labels = []
        self.fg_events = sorted(self.fg_events, key=lambda x:x.event_time)
        for ievent, event in enumerate(self.fg_events):
            # load and normalize audio signal
            x,_ = librosa.load(event.source_file, sr=self.sr)
            x = x / np.max(np.abs(x))

            # fetch trajectory from irs
            ir_idx = trajectory2indices(all_ir_xyzs, event.event_position)
            irs = all_irs[ir_idx]
            ir_xyzs = all_ir_xyzs[ir_idx]
            # remove repeated positions for now
            ir_idx = find_indices_of_change(ir_xyzs)
            irs = irs[ir_idx]
            ir_xyzs = ir_xyzs[ir_idx]

            # normalize irs to have unit energy
            norm_irs = IR_normalizer(irs)

            # SPATIALIZE
            norm_irs = np.transpose(norm_irs, (2, 1, 0))
            if len(irs) > 1:
                ir_times = np.linspace(0, event.event_duration, len(irs))
                xS = spatialize(x, norm_irs, ir_times, sr=self.sr, s=event.snr)
            else:
                continue

            # generate ground truth
            time_grid = get_timegrid(len(xS), self.sr, ir_times, time_grid_resolution=round(1/self.label_rate,1))
            labels = get_labels(ir_times, time_grid, ir_xyzs, class_id=self.fg_labels[event.label], source_id=ievent)
            labels[:,0] = labels[:,0] + int(event.event_time * self.label_rate)
            xS = xS[: int(time_grid[-1] * self.sr)]  # trim audio signal to exactly match labels
            onsamp = int(event.event_time*self.sr)
            out_audio[onsamp:onsamp+len(xS)] += xS
            all_labels.append(labels)

        labels = sort_matrix_by_columns(np.vstack(all_labels))

        # save output
        save_output(audiopath, labelpath, out_audio, self.sr, labels)
