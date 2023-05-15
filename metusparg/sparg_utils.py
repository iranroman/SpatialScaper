import math

def az_ele_from_source(ref_point, src_point):
    """
    Calculates the azimuth and elevation between a reference point and a source point in 3D space
    Args:
        ref_point (list): A list of three floats representing the x, y, and z coordinates of the reference point
        src_point (list): A list of three floats representing the x, y, and z coordinates of the other point
    Returns:
        A tuple of two floats representing the azimuth and elevation angles in radians plus distance between reference and source point
    """
    dx = src_point[0] - ref_point[0]
    dy = src_point[1] - ref_point[1]
    dz = src_point[2] - ref_point[2]
    azimuth = math.atan2(dy, dx)
    distance = math.sqrt(dx**2 + dy**2 + dz**2)
    elevation = math.asin(dz/distance)
    return azimuth, elevation, distance

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

def get_mic_xyz():
    """
    Get em32 microphone coordinates in 3D space
    """
    return [(3 - 3) * 0.5, (3 - 3) * 0.5, (2 - 2) * 0.3 + 1.5]

# Full set of measurements from METU Sparg dataset
rir_meas_data = [
    '000', '012', '024', '041', '053', '100', '112', '124', '141', '153', '200', '212', '224', '241', '253', '300', '312', '324', '342', '354', '401', '413', '430', '442', '454', '501', '513', '530', '542', '554', '601', '613', '630', '642', '654',
    '001', '013', '030', '042', '054', '101', '113', '130', '142', '154', '201', '213', '230', '242', '254', '301', '313', '330', '343', '360', '402', '414', '431', '443', '460', '502', '514', '531', '543', '560', '602', '614', '631', '643', '660', 
    '002', '014', '031', '043', '060', '102', '114', '131', '143', '160', '202', '214', '231', '243', '260', '302', '314', '331', '344', '361', '403', '420', '432', '444', '461', '503', '520', '532', '544', '561', '603', '620', '632', '644', '661', 
    '003', '020', '032', '044', '061', '103', '120', '132', '144', '161', '203', '220', '232', '244', '261', '303', '320', '333', '350', '362', '404', '421', '433', '450', '462', '504', '521', '533', '550', '562', '604', '621', '633', '650', '662', 
    '004', '021', '033', '050', '062', '104', '121', '133', '150', '162', '204', '221', '233', '250', '262', '304', '321', '334', '351', '363', '410', '422', '434', '451', '463', '510', '522', '534', '551', '563', '610', '622', '634', '651', '663', 
    '010', '022', '034', '051', '063', '110', '122', '134', '151', '163', '210', '222', '234', '251', '263', '310', '322', '340', '352', '364', '411', '423', '440', '452', '464', '511', '523', '540', '552', '564', '611', '623', '640', '652', '664', 
    '011', '023', '040', '052', '064', '111', '123', '140', '152', '164', '211', '223', '240', '252', '264', '311', '323', '341', '353', '400', '412', '424', '441', '453', '500', '512', '524', '541', '553', '600', '612', '624', '641', '653'
    ]