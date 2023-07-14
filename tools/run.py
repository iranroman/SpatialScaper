from room_scaper.utils.parser import parse_args, load_config
from room_scaper.data.utils import get_path_to_room_files
import yaml
import pickle
import os

def get_fold_files(foldname, filenames):
    """
    assuming the foldname is in the
    relevant filenames path
    """
    fold_files = [fname for fname in filenames if foldname in fname.split('/')]
    if fold_files:
        return fold_files
    else:
        import warnings
        warnings.warn(f'No files found for fold {foldname}')

def get_sound_event_filenames(path):
    if path.endswith('.txt'):
        filenames = []
        with open(path) as file:
            while line := file.readline():
                filenames.append(line.strip())
    # TODO: make this work with recursive listing of files in a directory
    return filenames
        
def load_pickle(filename):
    file = open(filename,'rb')
    object_file = pickle.load(file)
    file.close()
    return object_file
    


def get_room_trajectories(path_to_room_files):
    '''
    room_name: a string with the name
    '''
    room_trajs = load_pickle(os.path.join(path_to_room_files,'metadata','doa_xyz.pkl'))
    return room_trajs

def main():
    '''
    main function to trigger the data generation
    '''

    # parse config arguments
    args = parse_args()
    cfg = load_config(args, args.path_to_config)
    print('generating data using parameters:')
    print(yaml.dump(dict(cfg), allow_unicode=True, default_flow_style=False))

    
    event_filenames = get_sound_event_filenames(cfg.PATH_TO_SOUND_EVENT_FILES)


    fold_names = cfg.FOLD_NAMES

    # iterate over fold names
    for fold in fold_names:
        
        fold_event_files = get_fold_files(fold, event_filenames)

        # iterate over rooms
        fold_rooms = cfg.FOLD_ROOMS[fold]
        for room_name in fold_rooms:

            path_to_room_files = get_path_to_room_files(room_name)
            room_trajs = get_room_trajectories(path_to_room_files)
            print(room_trajs[0][0].shape)
        




if __name__ == "__main__":
    main()
