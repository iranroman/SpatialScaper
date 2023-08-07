import os
import mat73
import sys
sys.path.append('../room_simulation') #we should move stuff around to avoid this
import tau_loading
import numpy as np
import pysofaconventions as pysofa

from netCDF4 import Dataset
import time

def load_flat_tau_srir(tau_db_dir, room_idx, aud_fmt='foa', flip=True):
    rooms = ['bomb_shelter', 'gym', 'pb132', 'pc226', 'sa203', 'sc203', 'se203', 'tb103', 'tc352']
    room = rooms[room_idx]
    rir_file = [file for file in os.listdir(tau_db_dir) if room in file][0]
    rirs = mat73.loadmat(os.path.join(tau_db_dir, rir_file))['rirs']
    output_paths, path_metadata, room_metadata = tau_loading.load_paths(room_idx, tau_db_dir)
    n_traj, n_heights = output_paths.shape
    N, R, _ = rirs[aud_fmt][0][0].shape
    path_stack = np.empty((0, 3))
    rir_stack = np.empty((N, R, 0))
    M = 0
    for i in range(n_traj):
        for j in range(n_heights):
            path_ij = output_paths[i,j]
            rir_ij = rirs[aud_fmt][i][j]
            if flip:
                if j%2==1:
                    #flip every other height, as in DCASE
                    path_ij = path_ij[::-1]
                    rir_ij = rir_ij[::-1]
            path_stack = np.concatenate((path_stack, path_ij), axis=0)
            rir_stack = np.concatenate((rir_stack, rir_ij), axis=2) 
            M += output_paths[i,j].shape[0]
            
    rirs = np.reshape(rir_stack, (M,R,N))
    source_pos = np.reshape(path_stack, (M,3))
    mic_pos = np.repeat([room_metadata['microphone_position']], M, axis=0) 

    return rirs, source_pos, mic_pos, room

def create_srir_sofa(filepath, rirs, source_pos, mic_pos, db_name="Default_db",\
                     room_name="Room_name", listener_name="foa", sr=24000, comment="na"):
    M = rirs.shape[0]
    R = rirs.shape[1]
    N = rirs.shape[2]
    E = 1
    I = 1
    C = 3
    
    assert rirs.shape == (M,R,N)
    assert source_pos.shape == (M,C)
    
    
    # Need to delete it first if file already exists
    if os.path.exists(filepath):
        print(f"Overwriting {filepath}")
        os.remove(filepath)
    rootgrp = Dataset(filepath, 'w', format='NETCDF4')

    #----------Required Attributes----------#

    rootgrp.Conventions = 'SOFA'
    rootgrp.Version = '2.1'
    rootgrp.SOFAConventions = 'SingleRoomSRIR'
    rootgrp.SOFAConventionsVersion = '1.0'
    rootgrp.APIName = 'pysofaconventions'
    rootgrp.APIVersion = '0.1.5'
    rootgrp.AuthorContact = 'chris.ick@nyu.edu'
    rootgrp.Organization = 'Music and Audio Research Lab - NYU'
    rootgrp.License = 'Use whatever you want'
    rootgrp.DataType = 'FIR'
    rootgrp.DateCreated = time.ctime(time.time())
    rootgrp.DateModified = time.ctime(time.time())
    rootgrp.Title = db_name + " - " + room_name
    rootgrp.RoomType = 'shoebox'
    rootgrp.DatabaseName = db_name
    rootgrp.ListenerShortName = listener_name
    rootgrp.RoomShortName = room_name
    rootgrp.Comment = comment

    #----------Required Dimensions----------#

    rootgrp.createDimension('M', M)
    rootgrp.createDimension('N', N)
    rootgrp.createDimension('E', E)
    rootgrp.createDimension('R', R)
    rootgrp.createDimension('I', I)
    rootgrp.createDimension('C', C)

    #----------Required Variables----------#
    listenerPositionVar = rootgrp.createVariable('ListenerPosition',    'f8',   ('M','C'))
    listenerPositionVar.Units   = 'metre'
    listenerPositionVar.Type    = 'cartesian'
    listenerPositionVar[:] = mic_pos

    listenerUpVar       = rootgrp.createVariable('ListenerUp',          'f8',   ('I','C'))
    listenerUpVar.Units         = 'metre'
    listenerUpVar.Type          = 'cartesian'
    listenerUpVar[:]    = np.asarray([0,0,1])

    # Listener looking forward (+x direction)
    listenerViewVar     = rootgrp.createVariable('ListenerView',        'f8',   ('I','C'))
    listenerViewVar.Units       = 'metre'
    listenerViewVar.Type        = 'cartesian'
    listenerViewVar[:]  = np.asarray([1,0,0])

    #single emitter for each measurement
    emitterPositionVar  = rootgrp.createVariable('EmitterPosition',     'f8',   ('E','C','I'))
    emitterPositionVar.Units   = 'metre'
    emitterPositionVar.Type    = 'spherical'
    # Equidistributed speakers in circle
    emitterPositionVar[:] = np.zeros((E,C,I))

    sourcePositionVar = rootgrp.createVariable('SourcePosition',        'f8',   ('M','C'))
    sourcePositionVar.Units   = 'metre'
    sourcePositionVar.Type    = 'cartesian'
    sourcePositionVar[:]      = source_pos

    sourceUpVar       = rootgrp.createVariable('SourceUp',              'f8',   ('I','C'))
    sourceUpVar.Units         = 'metre'
    sourceUpVar.Type          = 'cartesian'
    sourceUpVar[:]    = np.asarray([0,0,1])

    sourceViewVar     = rootgrp.createVariable('SourceView',            'f8',   ('I','C'))
    sourceViewVar.Units       = 'metre'
    sourceViewVar.Type        = 'cartesian'
    sourceViewVar[:]  = np.asarray([1,0,0])

    receiverPositionVar = rootgrp.createVariable('ReceiverPosition',  'f8',   ('R','C','I'))
    receiverPositionVar.Units   = 'metre'
    receiverPositionVar.Type    = 'cartesian'
    receiverPositionVar[:]      = np.zeros((R,C,I))

    samplingRateVar =   rootgrp.createVariable('Data.SamplingRate', 'f8',   ('I'))
    samplingRateVar.Units = 'hertz'
    samplingRateVar[:] = sr

    delayVar        =   rootgrp.createVariable('Data.Delay',        'f8',   ('I','R'))
    delay = np.zeros((I,R))
    delayVar[:,:] = delay

    dataIRVar =         rootgrp.createVariable('Data.IR', 'f8', ('M','R','N'))
    dataIRVar.ChannelOrdering   = 'acn' #standard ambi ordering
    dataIRVar.Normalization     = 'sn3d'
    dataIRVar[:] = rirs

    #----------Close it----------#

    rootgrp.close
    print(f"SOFA file saved to {filepath}")

def load_rir_pos(filepath, doas=True):
    sofa = pysofa.SOFAFile(filepath,'r')
    assert sofa.isValid()
    rirs = sofa.getVariableValue('Data.IR')
    source_pos = sofa.getVariableValue('SourcePosition')
    if doas:
        source_pos = source_pos * (1/np.sum(source_pos, axis=1))[:, np.newaxis] #normalize
    sofa.close()
    return rirs, source_pos

def load_rir(filepath):
    sofa = pysofa.SOFAFile(filepath,'r')
    assert sofa.isValid()
    rirs = sofa.getVariableValue('Data.IR')
    sofa.close()
    return rirs

def load_pos(filepath, doas=True):
    sofa = pysofa.SOFAFile(filepath,'r')
    assert sofa.isValid()
    source_pos = sofa.getVariableValue('SourcePosition')
    if doas:
        source_pos = source_pos * (1/np.sum(source_pos, axis=1))[:, np.newaxis] #normalize
    sofa.close()
    return source_pos


