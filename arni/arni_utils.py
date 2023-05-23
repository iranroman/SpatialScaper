import math

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