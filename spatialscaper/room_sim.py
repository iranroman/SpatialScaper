import numpy as np
import os
import pickle

import pyroomacoustics as pra
from pyroomacoustics import directivities as dr
from room_scaper import sofa_utils, tau_loading


def deg2rad(deg):
    return deg * 2 * np.pi / 360


def rad2deg(rad):
    return rad * 360 / (2 * np.pi)


def plot_energy_db(ax, rir, fs=24000):
    # The power of the impulse response in dB
    power = rir**2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    # remove the possibly all zero tail
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]
    ax.plot(energy_db)


def get_tetra_mics():
    # return geometry of standard tetrahedral mic config as in TAU-SRIR dataset

    # coordinates stored in radius (m), azimuth (deg) and elevation (deg)
    m1_coords = [0.042, 45, 35]
    m2_coords = [0.042, -45, -35]
    m3_coords = [0.042, 135, -35]
    m4_coords = [0.042, -135, 35]

    mic_coords = [m1_coords, m2_coords, m3_coords, m4_coords]
    mic_dirs = [
        dr.CardioidFamily(
            orientation=dr.DirectionVector(
                azimuth=coord[1], colatitude=90 - coord[2], degrees=True
            ),
            pattern_enum=dr.DirectivityPattern.HYPERCARDIOID,
        )
        for coord in mic_coords
    ]

    return mic_coords, mic_dirs


def center_mic_coords(mic_coords, mic_center):
    mic_locs = np.empty((0, 3))
    for coord in mic_coords:
        rad, azi, ele = coord
        azi = deg2rad(azi)
        ele = deg2rad(ele)
        x_offset = rad * np.cos(azi) * np.cos(ele)
        y_offset = rad * np.sin(azi) * np.cos(ele)
        z_offset = rad * np.sin(ele)
        mic_loc = mic_center + np.array([x_offset, y_offset, z_offset])
        mic_locs = np.vstack([mic_locs, mic_loc])
    return mic_locs


def unitvec_to_cartesian(path_unitvec, height, dist):
    if type(dist) == np.ndarray:
        z_offset = height
        rad = np.sqrt(dist[0] ** 2 + (dist[2] + z_offset) ** 2)
        scaled_path = tau_loading.map_to_cylinder(path_unitvec, rad, axis=1)
    else:
        scaled_path = tau_loading.map_to_cylinder(path_unitvec, dist, axis=2)
    return scaled_path
