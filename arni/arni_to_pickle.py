from pysofaconventions import *
import matplotlib.pyplot as plt
import librosa
import numpy as np
import math
import pickle


def unit_vector(azimuth, elevation):
    """
    Compute unit vector given the azimuth and elevetion of source in 3D space
    Args:
        azimuth (float)
        elevation (float)
    Returns:
        A list representing the coordinate points xyz in 3D space
    """
    x = math.cos(elevation) * math.cos(azimuth)
    y = math.cos(elevation) * math.sin(azimuth)
    z = math.sin(elevation)
    return [x, y, z]

def compute_azimuth_elevation(receiver_pos, source_pos):
    # Calculate the vector from the receiver to the source
    vector = [source_pos[0] - receiver_pos[0], source_pos[1] - receiver_pos[1], source_pos[2] - receiver_pos[2]]
    # Calculate the azimuth angle
    azimuth = math.atan2(vector[0], vector[1])
    # if azimuth < 0:
    #     azimuth += math.pi
    # Calculate the elevation angle
    distance = math.sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    elevation = math.asin(vector[2] / distance)
    return azimuth, elevation, distance

# Load the .sofa file
file_path = '/mnt/ssdt7/RIR-datasets/Arni/6dof_SRIRs_eigenmike_raw/6DoF_SRIRs_eigenmike_raw_0percent_absorbers_enabled.sofa'
sofa = SOFAFile(file_path,'r')

# File is actually not valid, but we can forgive them
if not sofa.isValid():
    print("Error: the file is invalid")

# Convention is SimpleFreeFieldHRIR
print("\n")
print("SOFA Convention:", sofa.getGlobalAttributeValue('SOFAConventions'))

# Let's see the dimensions:
print("\n")
print("Dimensions:")
sofa.printSOFADimensions()

# Let's see the variables as well
print("\n")
print("Variables")
sofa.printSOFAVariables()

# Let's check the position of the measurementa (Source position)
sourcePositions = sofa.getVariableValue('SourcePosition')
print("\n")
print("Source Positions")
print(sourcePositions)
# and the info (units, coordinates)
print(sofa.getPositionVariableInfo('SourcePosition'))

# Let's inspect the first measurement
m = 7
print("\n")
print("Source Position of measurement " + str(m))
print(sourcePositions[m])

# Let's inspect the first measurement
print("\n")
print("Listener Positions")
listenerPosition = sofa.getVariableValue('ListenerPosition')
print(listenerPosition)
print(sofa.getPositionVariableInfo('ListenerPosition'))
print("\n")
print("Listener Position of measurement " + str(m))
print(listenerPosition[m])
# which is at 82 degrees azimuth, -7 degrees elevation

# Read the data
rirdata = sofa.getDataIR()
num_meas, num_ch = rirdata.shape[0], rirdata.shape[1]
meas_per_mic = 3 # equal the number of meas per trajectory
num_mics_traj = 5 # equals the number of trajectories
FS = 48000 # original impulse reponse sampling rate
NEW_FS = 24000 # new sampling rate (same as DCASE Synth)
# Get mics coordinates
mic_xyz = [listenerPosition[mic] for mic in range(0,num_meas-6,3)]

print(mic_xyz)
print(listenerPosition[9])
rirdata_dict = {}
# Data structure (potentially to be used once this is fully integrated)
room = "arni"
rirdata_dict[room] = {}
rirdata_dict[room]['doa_xyz'] = [[] for i in range(num_mics_traj)]
rirdata_dict[room]['dist'] = [[] for i in range(num_mics_traj)]
rirdata_dict[room]['rir'] = {'mic':[[] for i in range(num_mics_traj)]}

meas_count = 0
for mic_traj in range(num_mics_traj):
    doa_xyz = [] # assume only one height
    dists = []
    hir_data = [] # assume only one height
    for meas in range(meas_per_mic):
        # add impulse response
        irdata = rirdata[meas_count, :, :]
        irdata_resamp = librosa.resample(irdata, orig_sr=FS, target_sr=NEW_FS)
        hir_data.append(irdata_resamp)
        print("mic and source", mic_xyz[mic_traj], sourcePositions[meas_count])
        azi, ele, dis = compute_azimuth_elevation(mic_xyz[mic_traj], sourcePositions[meas_count])
        print("Azi and ele", azi, ele)
        uvec_xyz = unit_vector(azi, ele)
        doa_xyz.append(uvec_xyz)
        dists.append(dis)
        meas_count += 1
    rirdata_dict[room]['doa_xyz'][mic_traj].append(np.array(doa_xyz))
    rirdata_dict[room]['rir']['mic'][mic_traj].append(np.transpose(np.array(hir_data),(2,1,0)))
    
with open('{}.pkl'.format(f"rirs_13_arni"), 'wb') as outp: # should go inside TAU_DB/TAU-SRIR_DB/rirs_13_arni.pkl
    pickle.dump(rirdata_dict[room]['rir'], outp, pickle.HIGHEST_PROTOCOL)

with open('{}.pkl'.format(f"doa_xyz_arni"), 'wb') as outp:
    pickle.dump(rirdata_dict[room]['doa_xyz'], outp, pickle.HIGHEST_PROTOCOL)

# (Nsamps, Nch, N_ir)



