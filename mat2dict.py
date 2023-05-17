import pickle
import sys
import scipy.io
import mat73
import numpy as np


def main(argv):

    PATH_TO_RIRs = argv[1]

    db_handler = open('db_config_fsd.obj','rb')  
    db_config = pickle.load(db_handler) 
    db_handler.close()

    measinfomat = scipy.io.loadmat('{}/measinfo.mat'.format(PATH_TO_RIRs))['measinfo']

    rirdata2room_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 9: 7, 10: 8}
    rirdata2room_idx = {v:k for k,v in rirdata2room_idx.items()}
    rirdata2room_measinfo = {1: 'bomb_shelter', 2: 'gym', 3: 'pb132', 4: 'pc226', 5: 'sa203', 6: 'sc203', 8: 'se203', 9: 'tb103', 10: 'tc352'}

    # 1. add room names to rirdata_dict
    rirdata_dict = {}
    for v in rirdata2room_measinfo.values():
        rirdata_dict[v] = {}
        rirdata_dict[v]['doa_xyz'] = None
        rirdata_dict[v]['dist'] = None

    # 2. add the trajectories to each room 
    for iroom, room in enumerate(rirdata_dict.keys()):

        #################################
        #################################
        #################################
        # work first with the rir metadata
        #################################
        #################################
        #################################

        trajs_data = db_config._rirdata[iroom][0][2]

        trajs_list = []
        for traj in trajs_data:

            heights_data = traj

            heights_list = []
            for height in heights_data:

                heights_list.append(height[0])

            trajs_list.append(heights_list)
        rirdata_dict[room]['doa_xyz'] = trajs_list

        #################################
        #################################
        #################################
        # now generate a dictionary with the actual rirs
        #################################
        #################################
        #################################

        rirfile = '{}rirs_{:02d}_{}.mat'.format(PATH_TO_RIRs,rirdata2room_idx[iroom],room)
        rirwavs_mic = mat73.loadmat(rirfile)['rirs']['mic']
        rirwavs_foa = mat73.loadmat(rirfile)['rirs']['foa']
        rirwavs = {'mic':rirwavs_mic, 'foa':rirwavs_foa}
        with open('{}.pkl'.format(rirfile[:-4]), 'wb') as outp:
            pickle.dump(rirwavs, outp, pickle.HIGHEST_PROTOCOL)


    with open('rirdata_dict.pkl', 'wb') as outp:
        pickle.dump(rirdata_dict, outp, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

