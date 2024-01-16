import os
import glob
import numpy as np

def get_label_list(folder_path):
    '''
    modified from
    github.com/justinsalamon/scaper/master/scaper/util.py
    '''

    label_list = []
    folder_names = os.listdir(folder_path)
    for fname in folder_names:
        if (os.path.isdir(os.path.join(folder_path, fname)) and
                fname[0] != '.'):
            label_list.append(fname)
    # ensure consistent ordering of labels
    label_list.sort()
    return label_list

def get_files_list(path,split):
    if split:
        subfiles = glob.glob(os.path.join(path,split, "**"),recursive=True)
    else:
        subfiles = glob.glob(os.path.join(path, "**"),recursive=True)
    subfiles = [f for f in subfiles if os.path.isfile(f)]
    return subfiles

def new_event_exceeds_max_overlap(new_event_time, new_event_duration, other_events, max_overlap, increment):
    """ Check if the new event overlaps with more than max_overlap events. """

    # Incrementally check each second of the new event's duration
    for t in np.arange(new_event_time, new_event_time + new_event_duration, increment):
        current_overlap = 0
        for event in other_events:
            if t >= event.event_time - increment and t <= event.event_time + event.event_duration + increment:
                current_overlap += 1
            if current_overlap > max_overlap:
                return True  # Overlaps with more than max_overlap events
    return False  # Suitable time found


def count_leading_zeros_in_period(frequency_hz):
    # Calculate the period
    period_seconds = 1 / frequency_hz

    # Convert the period to a string to find the leading zeros
    period_str = f"{period_seconds:.10f}"

    # Split the string at the decimal point and work with the fractional part
    fractional_part = period_str.split('.')[1]

    # Count leading zeros using a generator expression with a condition to stop after first non-zero digit
    return sum(1 for i, digit in enumerate(fractional_part) if digit == '0' and '1' not in fractional_part[:i+1])

