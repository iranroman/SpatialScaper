import os
import math
import pickle
import librosa
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def unit_vector(ref, source):
    """
    Calculates a unit vector between two points in 3D space
    """
    x = source[0] - ref[0]
    y = source[1] - ref[1]
    z = source[2] - ref[2]
    mag = math.sqrt(x**2 + y**2 + z**2)
    return [x/mag, y/mag, z/mag]

def points_distance(ref, source):
    """
    Calculates distance between two points in 3D space
    """
    return np.sqrt(np.sum((np.array(ref)-np.array(source))**2))

METU_PATH = "/mnt/ssdt7/RIR-datasets/spargair/em32"
mic_xyz = [(3 - 3) * 0.5, (3 - 2) * 0.3, (3 - 3) * 0.5 + 1.5]
mic_nch = 32

rir_meas_data = [
    '000', '012', '024', '041', '053', '100', '112', '124', '141', '153', '200', '212', '224', '241', '253', '300', '312', '324', '342', '354', '401', '413', '430', '442', '454', '501', '513', '530', '542', '554', '601', '613', '630', '642', '654',
    '001', '013', '030', '042', '054', '101', '113', '130', '142', '154', '201', '213', '230', '242', '254', '301', '313', '330', '343', '360', '402', '414', '431', '443', '460', '502', '514', '531', '543', '560', '602', '614', '631', '643', '660', 
    '002', '014', '031', '043', '060', '102', '114', '131', '143', '160', '202', '214', '231', '243', '260', '302', '314', '331', '344', '361', '403', '420', '432', '444', '461', '503', '520', '532', '544', '561', '603', '620', '632', '644', '661', 
    '003', '020', '032', '044', '061', '103', '120', '132', '144', '161', '203', '220', '232', '244', '261', '303', '320', '333', '350', '362', '404', '421', '433', '450', '462', '504', '521', '533', '550', '562', '604', '621', '633', '650', '662', 
    '004', '021', '033', '050', '062', '104', '121', '133', '150', '162', '204', '221', '233', '250', '262', '304', '321', '334', '351', '363', '410', '422', '434', '451', '463', '510', '522', '534', '551', '563', '610', '622', '634', '651', '663', 
    '010', '022', '034', '051', '063', '110', '122', '134', '151', '163', '210', '222', '234', '251', '263', '310', '322', '340', '352', '364', '411', '423', '440', '452', '464', '511', '523', '540', '552', '564', '611', '623', '640', '652', '664', 
    '011', '023', '040', '052', '064', '111', '123', '140', '152', '164', '211', '223', '240', '252', '264', '311', '323', '341', '353', '400', '412', '424', '441', '453', '500', '512', '524', '541', '553', '600', '612', '624', '641', '653'
    ]

outter_trayectory_top = ["062", "061", "060", "160", "260", "360", "460", "560", "660", "661", "662", "663", "664", "564", "464", "364", "264", "164", "064", "063"]
# inner_trayectory_top = [TBD]

rirdata_dict = {}
room = "metu"
rirdata_dict[room] = {}
rirdata_dict[room]['doa_xyz'] = [[]]
rirdata_dict[room]['dist'] = [[]]
rirdata_dict[room]['rir'] = [[]]

x_coords = []
y_coords = []
z_coords = []

for height in range(0, 7): # decrease heights
    heights_list = []
    doa_xyz = []
    hir_list = [] # impulse responses list per height
    for num in outter_trayectory_top:
        # Coords computed based on documentation.pdf from METU Sparg
        x = (3 - int(num[0])) * 0.5
        y = (3 - int(num[2])) * 0.3
        z = (3 - int(num[1])+height) * 0.5 + 1.5 # (1.5) mic height
        source_xyz = [x, y, z]
        doa_xyz.append(unit_vector(mic_xyz, source_xyz))
        distances = points_distance(mic_xyz, source_xyz)
        heights_list.append(distances) 
        rir_name = num[0] + str(int(num[1])-height) + num[2]
        '''
        # Note: uncomment code below to merge IRs into a multi-channel .wav
        # Load the 32 eigenmike IR wavefiles and concat in a single numpy array
        cmd_mix_all_ch = ""
        for mic_idx in range(1,mic_nch+1):
            cmd_mix_all_ch += f'{os.path.join(METU_PATH, rir_name)}/IR{mic_idx:05}.wav '
        # SoX -M to merge 32 into multi-channel signal
        ir32ch_path = os.path.join(METU_PATH, rir_name, "IR_32ch.wav")
        print("generating", ir32ch_path)
        os.system(f'sudo sox -M {cmd_mix_all_ch} {ir32ch_path}')
        '''
        ir32ch_path = os.path.join(METU_PATH, rir_name, "IR_32ch.wav")
        # ir4ch_path = os.path.join(METU_PATH, rir_name, "IR_MIC.wav")
        irdata, sr = librosa.load(ir32ch_path, mono=False, sr=48000)
        irdata_resamp = librosa.resample(irdata, orig_sr=sr, target_sr=24000)
        print("Adding RIR with shape", irdata_resamp.shape)
        hir_list.append(irdata_resamp)
        # # Used for plotting only
        # x_coords.append(x)
        # y_coords.append(y)
        # z_coords.append(z)
    rirdata_dict[room]['doa_xyz'][0].append(doa_xyz)
    rirdata_dict[room]['dist'][0].append(heights_list)
    rirdata_dict[room]['rir'][0].append(hir_list)

# print("doa_xyz info")
# print("Length list", len(rirdata_dict[room]['doa_xyz']))
# print("Length inner list", len(rirdata_dict[room]['doa_xyz'][0]))

with open('{}.pkl'.format("rir_MIC"), 'wb') as outp:
    pickle.dump(rirdata_dict[room]['rir'], outp, pickle.HIGHEST_PROTOCOL)

with open('{}.pkl'.format("doa_xyz_MIC"), 'wb') as outp:
    pickle.dump(rirdata_dict[room]['doa_xyz'], outp, pickle.HIGHEST_PROTOCOL)

# # Plotting measurements
# fig = plt.figure()
# ax = plt.axes(projection ='3d')
# # plotting
# ax.scatter(x_coords, y_coords,z_coords, 'blue')
# ax.scatter((3 - 3) * 0.5, (3 - 2) * 0.3, (3 - 3) * 0.5 + 1.5, 'red')
# ax.set_title('METU RIR measurements')
# plt.savefig("metu_meas.png")