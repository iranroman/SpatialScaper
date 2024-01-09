#    'tc352','sc203','bomb_shelter','pc226','pb132','se203',
#    'tb103','sa203','gym'
import os
import glob 

CODE_DIR = os.path.dirname(__file__)
PATH_TO_RIRS = os.path.join(CODE_DIR, 'rir_databases')

def get_path_to_room_files(room_name):
    all_room_filenames = glob.glob(os.path.join(PATH_TO_RIRS, '*/*'), recursive=True)
    room_files_path = [rname for rname in all_room_filenames if room_name==rname.split(os.sep)[-1]]
    assert len(room_files_path) == 1
    return room_files_path[0]

