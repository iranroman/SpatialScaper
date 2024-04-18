import os
import mat73
import sys
from spatialscaper import tau_utils
import numpy as np 
from netCDF4 import Dataset
import time
import pysofaconventions as pysofa


def load_flat_tau_srir(tau_db_dir, room_idx, aud_fmt="mic", traj=None, flip=True):
    rooms = [
        "bomb_shelter",
        "gym",
        "pb132",
        "pc226",
        "sa203",
        "sc203",
        "se203",
        "tb103",
        "tc352",
    ]
    room = rooms[room_idx]
    files = os.listdir(tau_db_dir)
    rir_file = [file for file in files if room in file][0]
    rirs = mat73.loadmat(os.path.join(tau_db_dir, rir_file))["rirs"]
    output_paths, path_metadata, room_metadata = tau_utils.load_paths(
        room_idx, tau_db_dir
    )
    n_traj, n_heights = output_paths.shape
    N, R, _ = rirs[aud_fmt][0][0].shape
    path_stack = np.empty((0, 3))
    rir_stack = np.empty((N, R, 0))
    M = 0
    if traj is None:
        traj_iter = np.arange(n_traj)
    else:
        traj_iter = [traj]
    for i in traj_iter:
        for j in range(n_heights):
            path = output_paths[i, j]
            path_rirs = rirs[aud_fmt][i][j]
            if flip:
                if j % 2 == 1:
                    # flip every other height, as in DCASE
                    path_rirs = path_rirs[:, :, ::-1]
                    path = path[::-1]
            path_stack = np.concatenate((path_stack, path), axis=0)
            rir_stack = np.concatenate((rir_stack, path_rirs), axis=2)
            M += output_paths[i, j].shape[0]

    rirs = np.moveaxis(rir_stack, [0, 2], [2, 0])
    source_pos = path_stack

    assert rirs.shape == (M, R, N)
    assert source_pos.shape == (M, 3)
    mic_pos = np.repeat([room_metadata["microphone_position"]], M, axis=0)

    return rirs, source_pos, mic_pos, room


def create_srir_sofa(
    filepath,
    rirs,
    source_pos,
    mic_pos,
    db_name="Default_db",
    room_name="Room_name",
    listener_name="foa",
    sr=24000,
    comment="N/A",
):
    """
    Creates a SOFA file with spatial room impulse response data.

    This function generates a SOFA (Spatially Oriented Format for Acoustics) file to store spatial room impulse responses (SRIRs).
    It includes metadata about the recording environment, such as source and microphone positions, room characteristics, and listener details.

    Parameters:
        filepath (str): The path where the SOFA file will be created or overwritten.
        rirs (numpy.array): A 3D array of room impulse responses (measurements x receivers x samples).
        source_pos (numpy.array): The positions of the sound sources (measurements x coordinates).
        mic_pos (numpy.array): The positions of the microphones/listeners (measurements x coordinates).
        db_name (str, optional): Name of the database. Default is "Default_db".
        room_name (str, optional): Name of the room. Default is "Room_name".
        listener_name (str, optional): Name of the listener. Default is "foa".
        sr (int, optional): Sampling rate of the impulse responses. Default is 24000 Hz.
        comment (str, optional): Additional comments. Default is "N/A".

    Returns:
        None: The function does not return a value. It creates or overwrites a SOFA file at the specified filepath.
    """
    M = rirs.shape[0]
    R = rirs.shape[1]
    N = rirs.shape[2]
    E = 1
    I = 1
    C = 3

    assert rirs.shape == (M, R, N)
    assert source_pos.shape == (M, C)

    # Need to delete it first if file already exists
    if os.path.exists(filepath):
        print(f"Overwriting {filepath}")
        os.remove(filepath)
    rootgrp = Dataset(filepath, "w", format="NETCDF4")

    # ----------Required Attributes----------#

    rootgrp.Conventions = "SOFA"
    rootgrp.Version = "2.1"
    rootgrp.SOFAConventions = "SingleRoomSRIR"
    rootgrp.SOFAConventionsVersion = "1.0"
    rootgrp.APIName = "pysofaconventions"
    rootgrp.APIVersion = "0.1.5"
    rootgrp.AuthorContact = "chris.ick@nyu.edu"
    rootgrp.Organization = "Music and Audio Research Lab - NYU"
    rootgrp.License = "Use whatever you want"
    rootgrp.DataType = "FIR"
    rootgrp.DateCreated = time.ctime(time.time())
    rootgrp.DateModified = time.ctime(time.time())
    rootgrp.Title = db_name + " - " + room_name
    rootgrp.RoomType = "shoebox"
    rootgrp.DatabaseName = db_name
    rootgrp.ListenerShortName = listener_name
    rootgrp.RoomShortName = room_name
    rootgrp.Comment = comment

    # ----------Required Dimensions----------#

    rootgrp.createDimension("M", M)
    rootgrp.createDimension("N", N)
    rootgrp.createDimension("E", E)
    rootgrp.createDimension("R", R)
    rootgrp.createDimension("I", I)
    rootgrp.createDimension("C", C)

    # ----------Required Variables----------#
    listenerPositionVar = rootgrp.createVariable("ListenerPosition", "f8", ("M", "C"))
    listenerPositionVar.Units = "metre"
    listenerPositionVar.Type = "cartesian"
    listenerPositionVar[:] = mic_pos

    listenerUpVar = rootgrp.createVariable("ListenerUp", "f8", ("I", "C"))
    listenerUpVar.Units = "metre"
    listenerUpVar.Type = "cartesian"
    listenerUpVar[:] = np.asarray([0, 0, 1])

    # Listener looking forward (+x direction)
    listenerViewVar = rootgrp.createVariable("ListenerView", "f8", ("I", "C"))
    listenerViewVar.Units = "metre"
    listenerViewVar.Type = "cartesian"
    listenerViewVar[:] = np.asarray([1, 0, 0])

    # single emitter for each measurement
    emitterPositionVar = rootgrp.createVariable(
        "EmitterPosition", "f8", ("E", "C", "I")
    )
    emitterPositionVar.Units = "metre"
    emitterPositionVar.Type = "spherical"
    # Equidistributed speakers in circle
    emitterPositionVar[:] = np.zeros((E, C, I))

    sourcePositionVar = rootgrp.createVariable("SourcePosition", "f8", ("M", "C"))
    sourcePositionVar.Units = "metre"
    sourcePositionVar.Type = "cartesian"
    sourcePositionVar[:] = source_pos

    sourceUpVar = rootgrp.createVariable("SourceUp", "f8", ("I", "C"))
    sourceUpVar.Units = "metre"
    sourceUpVar.Type = "cartesian"
    sourceUpVar[:] = np.asarray([0, 0, 1])

    sourceViewVar = rootgrp.createVariable("SourceView", "f8", ("I", "C"))
    sourceViewVar.Units = "metre"
    sourceViewVar.Type = "cartesian"
    sourceViewVar[:] = np.asarray([1, 0, 0])

    receiverPositionVar = rootgrp.createVariable(
        "ReceiverPosition", "f8", ("R", "C", "I")
    )
    receiverPositionVar.Units = "metre"
    receiverPositionVar.Type = "cartesian"
    receiverPositionVar[:] = np.zeros((R, C, I))

    samplingRateVar = rootgrp.createVariable("Data.SamplingRate", "f8", ("I"))
    samplingRateVar.Units = "hertz"
    samplingRateVar[:] = sr

    delayVar = rootgrp.createVariable("Data.Delay", "f8", ("I", "R"))
    delay = np.zeros((I, R))
    delayVar[:, :] = delay

    dataIRVar = rootgrp.createVariable("Data.IR", "f8", ("M", "R", "N"))
    dataIRVar.ChannelOrdering = "acn"  # standard ambi ordering
    dataIRVar.Normalization = "sn3d"
    dataIRVar[:] = rirs

    # ----------Close it----------#

    rootgrp.close
    print(f"SOFA file saved to {filepath}")


def load_rir_pos(filepath, doas=True):
    """
    Loads room impulse responses (RIRs) and their corresponding source positions from a SOFA file.

    This function opens a SOFA file, validates its format, and extracts the RIR data along with the sampling rate and
    source positions. If the 'doas' flag is set to True, the source positions are normalized to unit vectors.

    Args:
        filepath (str): The path to the SOFA file containing RIR data.
        doas (bool): A flag to indicate whether to normalize the source positions to Direction of Arrival (DoA) vectors.
                     If True, source positions are normalized; if False, raw positions are returned.

    Returns:
        tuple: A tuple containing:
               - rirs (numpy.ndarray): An array of room impulse responses.
               - rirs_sr (float): The sampling rate of the impulse responses.
               - source_pos (numpy.ndarray): An array of source positions, normalized if doas is True.

    The function asserts the validity of the SOFA file and raises an assertion error if the file is not valid.
    The source positions are normalized to unit vectors if 'doas' is True to represent directions rather than positions.
    """
    sofa = pysofa.SOFAFile(filepath, "r")
    rirs = sofa.getVariableValue("Data.IR")
    rirs_sr = sofa.getVariableValue("Data.SamplingRate")
    source_pos = sofa.getVariableValue("SourcePosition")
    _, meas_type = sofa.getPositionVariableInfo('SourcePosition')
    if doas:
        source_pos = (
            source_pos * (1 / np.sqrt(np.sum(source_pos**2, axis=1)))[:, np.newaxis]
        )  # normalize
    sofa.close()

    if len(rirs.shape)>3:
        axes  = rirs.shape
        rirs = rirs.reshape((axes[0], axes[1], axes[-1]))
    
    if meas_type == 'cartesian':
        return rirs, rirs_sr, source_pos
    elif meas_type == 'spherical':
        azi = source_pos[:,0]/360 * 2 * np.pi
        ele = source_pos[:,1]/360 * 2 * np.pi
        r = source_pos[:,2]

        source_pos[:,0] = r * np.cos(azi) * np.cos(ele)
        source_pos[:,1] = r * np.sin(azi) * np.cos(ele)
        source_pos[:,2] = r * np.sin(ele)

        return rirs, rirs_sr, source_pos


def load_rir(filepath):
    sofa = pysofa.SOFAFile(filepath, "r")
    assert sofa.isValid()
    rirs = sofa.getVariableValue("Data.IR")
    sofa.close()
    return rirs


def load_pos(filepath, doas=True):
    """
    Loads source positions from a SOFA file and optionally normalizes them to unit vectors.

    This function opens a SOFA file, validates its format, and extracts the source position data.
    If the 'doas' flag is set to True, the source positions are normalized to unit vectors,
    representing Direction of Arrival (DoA) vectors rather than absolute positions.

    Args:
        filepath (str): The path to the SOFA file containing source position data.
        doas (bool): A flag to indicate whether to normalize the source positions.
                     If True, source positions are normalized; if False, raw positions are returned.

    Returns:
        numpy.ndarray: An array of source positions. These positions are normalized if 'doas' is True.

    The function asserts the validity of the SOFA file and raises an assertion error if the file is not valid.
    Normalization of source positions (when 'doas' is True) transforms them into unit vectors,
    useful for applications requiring directional information rather than absolute positions.
    """
    sofa = pysofa.SOFAFile(filepath, "r")
    source_pos = sofa.getVariableValue("SourcePosition")
    _, meas_type = sofa.getPositionVariableInfo('SourcePosition')
    
    if doas:
        source_pos = (
            source_pos * (1 / np.sqrt(np.sum(source_pos**2, axis=1)))[:, np.newaxis]
        )  # normalize
    sofa.close()
    if meas_type == 'cartesian':
        return source_pos
    elif meas_type == 'spherical':
        azi = source_pos[:,0]/360 * 2 * np.pi
        ele = source_pos[:,1]/360 * 2 * np.pi
        r = source_pos[:,2]

        source_pos[:,0] = r * np.cos(azi) * np.cos(ele)
        source_pos[:,1] = r * np.sin(azi) * np.cos(ele)
        source_pos[:,2] = r * np.sin(ele)

        return source_pos










