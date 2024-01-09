import os
import pickle
import scipy
import mat73
import numpy as np

from utils import parse_args, get_y
from pathlib import Path

ROOM_NAMES = ['bomb_shelter', 'gym', 'pb132', 'pc226', 'sa203', 'sc203', 'se203', 'tb103', 'tc352']
OUTPATH = 'room_scaper/data/rir_databases/TAU_SRIR_DB'

def main():
    args = parse_args()

    PATH_TO_RIRs = args.path

    measinfo = scipy.io.loadmat(os.path.join(PATH_TO_RIRs,'measinfo.mat'))['measinfo']
    rirdata = scipy.io.loadmat(os.path.join(PATH_TO_RIRs,'rirdata.mat'))['rirdata']['room'][0][0]

    # get the trajectories for each room 
    for iroom, room in enumerate(ROOM_NAMES):

        #################################
        #################################
        #################################
        # work first with the rir metadata
        #################################
        #################################
        #################################

        trajs_data = rirdata[iroom][0][2]

        trajs_list_doa = []
        for traj in trajs_data:

            heights_data = traj

            heights_list = []
            for height in heights_data:

                heights_list.append(height[0])

            trajs_list_doa.append(heights_list)

        room_out_path_meta = os.path.join(OUTPATH, room, 'metadata')
        Path(room_out_path_meta).mkdir(parents=True, exist_ok=True)

        with open(os.path.join(room_out_path_meta, 'doa_xyz.pkl'), 'wb') as outp:
            pickle.dump(trajs_list_doa, outp, pickle.HIGHEST_PROTOCOL)

        #################################
        #################################
        # now add the distance information
        #################################
        #################################
        room_name = measinfo['room'][iroom][0][0]
        assert room_name == room
        trajectories = [str(t[0]) for t in measinfo['trajectories'][iroom][0][0]] 
        trajectory_type = (measinfo['trajectoryType'][iroom][0][0]) 
        dists = np.array([t for t in measinfo['distances'][iroom][0]]) 
        heights = np.array([t for t in measinfo['heights'][iroom][0]])
        mic_position = np.array(measinfo['micPosition'][iroom][0])
        # compute distances
        trajs_list = []
        for ntraj in range(len(trajectories)):
        heights_list = []
        for nheight in range(len(heights[0])):
            #start
            path = trajs_list_doa[ntraj][nheight]
            ndoas = trajs_list_doa[ntraj][nheight].shape[0]
            height_delta = heights[0,nheight] - 1.2 # the microphone height is assumed to be 1.2
            
            if trajectory_type == 'circular':
                rad = dists[0,ntraj]
                distances = map_to_cylinder(path, rad, axis=2)
            else: # assuming it's a straight line
                rad = np.sqrt(dists[0][0][ntraj]**2 + (dists[0][2][ntraj]+height_delta)**2)
                distances = map_to_cylinder(path, rad, axis=1)
            heights_list.append(distances)            
        trajs_list.append(heights_list)

        with open(os.path.join(room_out_path_meta, 'dist.pkl'), 'wb') as outp:
            pickle.dump(trajs_list, outp, pickle.HIGHEST_PROTOCOL)
        #rirfile = '{}rirs_{:02d}_{}.mat'.format(PATH_TO_RIRs,rirdata2room_idx[iroom],room)
        #rirwavs = mat73.loadmat(rirfile)['rirs']['mic'] # just mic for now
        #with open('{}.pkl'.format(rirfile[:-4]), 'wb') as outp:
        #    pickle.dump(rirwavs, outp, pickle.HIGHEST_PROTOCOL)




if __name__ == "__main__":
    main()

'''
import pickle
import scipy.io
import mat73
import numpy as np

PATH_TO_RIRs = '/home/iran/datasets/TAU-SRIR_DB/'

db_handler = open('db_config_fsd.obj','rb')  
db_config = pickle.load(db_handler) 
db_handler.close()


rirdata2room_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 9: 7, 10: 8}
rirdata2room_idx = {v:k for k,v in rirdata2room_idx.items()}
'''
