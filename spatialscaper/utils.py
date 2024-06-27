import os
import glob
import random
import numpy as np
import scipy
import soundfile as sf
from scipy.spatial import KDTree
from scipy.interpolate import interp1d


def set_seed(seed=123456):
    """Set the random seeds for libraries used by SpatialScaper."""
    random.seed(seed)
    np.random.seed(seed)


def cartesian_to_polar(cartesian_coords, include_radius=True):
    """
    Converts Cartesian coordinates to polar coordinates (azimuth, elevation, and optionally radius).

    This function takes an array of Cartesian coordinates (X, Y, Z) and converts each coordinate
    to polar form, consisting of azimuth (angle in the XY plane) and elevation (angle from the XY plane).
    Optionally, it can also return the radius (distance from the origin). Azimuth and elevation are
    provided in degrees.

    Parameters:
        cartesian_coords (numpy.ndarray): An Nx3 matrix of Cartesian coordinates, where each row represents a (X, Y, Z) coordinate.
        include_radius (bool, optional): Whether to include the radius in the output. Defaults to True.

    Returns:
        numpy.ndarray: An Nx2 or Nx3 matrix of polar coordinates. Each row contains (azimuth in degrees, elevation in degrees)
                       and optionally the radius, depending on the value of include_radius.

    Note:
        Azimuth is calculated as the arctangent of Y/X and is in the range [-180, 180] degrees.
        Elevation is the arcsine of Z divided by the radius and is in the range [-90, 90] degrees.
        Radius is the Euclidean distance from the origin to the point, included if include_radius is True.
    """
    x, y, z = cartesian_coords[:, 0], cartesian_coords[:, 1], cartesian_coords[:, 2]

    # Radius (distance)
    radius = np.sqrt(x**2 + y**2 + z**2)

    # Azimuth (angle in the XY plane)
    azimuth = np.arctan2(y, x)
    # Convert azimuth from radians to degrees
    azimuth_deg = np.degrees(azimuth)

    # Elevation (angle from the XY plane)
    # Guard against division by zero in case the radius is zero
    with np.errstate(divide="ignore", invalid="ignore"):
        elevation = np.arcsin(z / radius)
    # Where radius is zero, set elevation to 0 (to handle NaNs)
    elevation[np.isnan(elevation)] = 0
    # Convert elevation from radians to degrees
    elevation_deg = np.degrees(elevation)

    if include_radius:
        return np.column_stack((azimuth_deg, elevation_deg, radius))
    else:
        return np.column_stack((azimuth_deg, elevation_deg))


def sort_matrix_by_columns(matrix, primary_col=0, secondary_col=2):
    """
    Sort the matrix based on two columns.
    First, sort by the primary column in ascending order.
    Then, within each group of identical values in the primary column, sort by the secondary column in ascending order.

    :param matrix: NumPy array to be sorted.
    :param primary_col: Index of the primary column for sorting.
    :param secondary_col: Index of the secondary column for sorting.
    :return: Sorted NumPy array.
    """
    # First, sort by the secondary column, then by the primary column
    sorted_matrix = matrix[
        np.lexsort((matrix[:, secondary_col], matrix[:, primary_col]))
    ]
    return sorted_matrix


def get_timegrid(nSamps, sr, ir_times, time_grid_resolution=0.1):
    """
    Generates a time grid for a given signal based on its sampling rate and total number of samples.

    This function creates a time grid starting from 0 to the duration of the signal,
    with a specified time grid resolution. It also checks to ensure that the last IR time
    does not exceed the signal's duration.

    Parameters:
        nSamps (int): Total number of samples in the signal.
        sr (int): Sampling rate of the signal.
        ir_times (list or numpy.array): Times corresponding to impulse responses.
        time_grid_resolution (float, optional): Interval between time points in the time grid. Default is 0.1 seconds.

    Returns:
        numpy.array: A time grid as a NumPy array.

    Example:
        # Example usage
        time_grid = get_timegrid(24000, 24000, [0, 0.5, 1], 0.01)
    """
    dur = nSamps / sr
    time_grid = np.arange(0, dur, time_grid_resolution)
    return time_grid


def get_labels(ir_times, time_grid, IR_XYZs, class_id=None, source_id=0, polar=True):
    """
    Generates labels for positions over time based on impulse response times and a time grid.

    This function interpolates positions at each time point in the time grid and can convert
    these coordinates from Cartesian to polar form. Each label includes frame number, class ID,
    source ID, and coordinates.

    Parameters:
        ir_times (numpy.array): Times at which impulse responses occur.
        time_grid (numpy.array): A grid of time points for generating labels.
        IR_XYZs (numpy.array): Cartesian coordinates corresponding to `ir_times`.
        class_id (int, optional): Class identifier for the labels. Default is None.
        source_id (int, optional): Identifier for the source. Default is 0.
        polar (bool, optional): Whether to convert coordinates to polar form. Default is True.

    Returns:
        numpy.array: An array of labels, each including frame number, class ID, source ID, and coordinates.

    Example:
        # Example usage
        ir_times = [0, 0.5, 1]
        time_grid = np.linspace(0, 1, 5)
        IR_XYZs = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        labels = get_labels(ir_times, time_grid, IR_XYZs, class_id=1, source_id=0, polar=True)
    """

    # Interpolation function for each coordinate dimension
    interpolated_x = interp1d(ir_times, IR_XYZs[:, 0], kind="linear")
    interpolated_y = interp1d(ir_times, IR_XYZs[:, 1], kind="linear")
    interpolated_z = interp1d(ir_times, IR_XYZs[:, 2], kind="linear")
    IR_XYZ_interp = np.array(
        [
            interpolated_x(time_grid),
            interpolated_y(time_grid),
            interpolated_z(time_grid),
        ]
    ).T
    nFrames = len(IR_XYZ_interp)
    if polar:
        coords = cartesian_to_polar(IR_XYZ_interp)[1:]
        coords[:, :2] = np.round(coords[:, :2], 3)
    else:
        coords = IR_XYZ_interp[1:]
    labels = np.hstack(
        (
            np.arange(nFrames - 1)[..., np.newaxis],
            np.array([(nFrames - 1) * [class_id]]).T,
            np.array([(nFrames - 1) * [source_id]]).T,
            coords,
        )
    )
    return labels


def save_output(audiofile, labelfile, x, sr, labels, fmt="mic"):
    """
    Saves audio data and corresponding labels to files.

    This function saves audio data as a WAV file and labels as a CSV file. The audio file is saved in a
    subdirectory named after the format (`fmt`), and the labels are saved in a 'labels' subdirectory.

    Parameters:
        outdir (str): The base directory where the output files will be saved.
        filename (str): The base name for the output files (without file extension).
        x (numpy.array): The audio data to be saved.
        sr (int): The sampling rate of the audio data.
        labels (numpy.array): The label data to be saved.
        fmt (str, optional): The format subdirectory for saving the audio file. Default is "mic".

    Example:
        # Example usage
        save_output('output', 'test', np.random.randn(24000), 24000, np.random.randint(0, 10, (10, 3)))
    """

    os.makedirs(os.path.dirname(audiofile), exist_ok=True)
    wavpath = f"{audiofile}.wav"
    sf.write(wavpath, x, sr)
    os.makedirs(os.path.dirname(labelfile), exist_ok=True)
    labelspath = f"{labelfile}.csv"
    np.savetxt(
        labelspath, labels, delimiter=",", fmt=["%i", "%i", "%i", "%i", "%i", "%.3f"]
    )


def IR_normalizer(IRs):
    """
    Normalizes impulse responses based on their energy.

    This function calculates the energy of each impulse response as the square root of the sum of its squared values.
    It then normalizes each impulse response by the mean energy across all responses.

    Parameters:
        IRs (numpy.array): A 2D or 3D array of impulse responses, where each row (in the 2D case) or matrix (in the 3D case)
                           represents an individual impulse response.

    Returns:
        numpy.array: The normalized impulse responses, having the same shape as the input `IRs`.

    Example:
        # Example of normalizing a set of impulse responses
        impulse_responses = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        normalized_IRs = IR_normalizer(impulse_responses)
    """
    E = np.sqrt(np.sum(np.power(np.abs(IRs), 2), axis=-1, keepdims=True))
    return IRs / np.mean(E, axis=-2, keepdims=True)


def find_indices_of_change(arr):
    """
    Identifies and returns the indices in an array where consecutive elements differ.

    This function processes an array and finds the indices at which the value of the array changes
    compared to the previous element. It is particularly useful for identifying changes in a sequence
    of spatial coordinates or any other sequence of values where changes need to be detected.

    Args:
        arr (numpy.ndarray): An array of values. The array can be multidimensional, but it is
                             typically used with 2D arrays where each row represents a distinct item
                             (e.g., XYZ coordinates).

    Returns:
        list: A list of indices where a change occurs in the array. The first index (0) is always included,
              as the first element is considered a 'change' from a non-existent previous element.

    The function converts each row of the array into a tuple for easy comparison, then iterates through
    these tuples to detect changes. An index is added to the result list if the current tuple is different
    from the previous one.
    """
    # Convert the array to a list of tuples for easy comparison
    arr_tuples = [tuple(row) for row in arr]

    # List to store indices of change
    change_indices = [0]  # Start with the first index

    # Iterate over the list of tuples, starting from the second element
    for i in range(1, len(arr_tuples)):
        # Check if the current element is different from the previous one
        if arr_tuples[i] != arr_tuples[i - 1]:
            change_indices.append(i)

    return change_indices


def traj_2_ir_idx(XYZs, trajectory):
    """
    Maps a set of trajectory points to their nearest neighbors in a given set of coordinates using a k-d tree.

    This function builds a k-d tree from a set of 3D coordinates and finds the nearest neighbor in these coordinates
    for each point in the specified trajectory.

    Parameters:
        XYZs (numpy.array): A 2D array of coordinates (e.g., [[x1, y1, z1], [x2, y2, z2], ...]) for k-d tree construction.
        trajectory (numpy.array): A 2D array of points for which nearest neighbors are to be found.

    Returns:
        list: Indices of the nearest neighbors in `XYZs` for each point in `trajectory`.

    Example:
        # Example set of coordinates and trajectory points
        coordinates = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        trajectory_points = np.array([[0.1, 0.1, 0.1], [1.5, 1.5, 1.5]])

        # Get indices of nearest neighbors
        nearest_indices = trajectory2indices(coordinates, trajectory_points)
    """

    # Create a k-d tree from IR_XYZ
    tree = KDTree(XYZs)

    # Find the nearest neighbor for each point in trajectory
    indices = []
    for point in trajectory:
        distance, index = tree.query(point)
        indices.append(index)

    return indices


def db2multiplier(db, x):
    """
    Calculates the multiplier factor from a decibel (dB) value that, when applied to x,
    adjusts its amplitude to reflect the specified dB. The relationship is
    based on the formula 20 * log10(factor * x) ≈ db

    Args:
        db (float): The target decibel change to be applied.
        x  (float): The original amplitude of x

    Returns:
        float: The multiplier factor.

    """
    return 10 ** (db / 20) / x


def generate_trajectory(xyz_start, xyz_end, npoints, shape):
    """
    Generate a linear or semicircular trajectory in 3D space.

    Parameters:
    xyz_start (list): Starting point [x, y, z].
    xyz_end (list): Ending point [x, y, z].
    npoints (int): Number of points in the trajectory.
    shape (str): 'linear' for a straight line or 'circular' for a semicircle.

    Returns:
    list of lists: Trajectory points [[x, y, z], [x, y, z], ...].
    """

    def linear_trajectory():
        # Linear interpolation between start and end points
        return [
            list(np.linspace(np.array(xyz_start), np.array(xyz_end), npoints)[i])
            for i in range(npoints)
        ]

    def circular_trajectory():
        # Vector from start to end
        start_to_end_vec = np.array(xyz_end) - np.array(xyz_start)
        midpoint = np.array(xyz_start) + start_to_end_vec / 2

        # Radius of the circle
        radius = np.linalg.norm(start_to_end_vec) / 2

        # Normal vector to the plane containing the circle
        if (start_to_end_vec == [0, 0, 0]).all():
            normal_vector = np.array([1, 0, 0])  # default normal vector
        else:
            normal_vector = np.array([1, 0, 0])  # initial guess
            if np.cross(normal_vector, start_to_end_vec).any():
                normal_vector = np.cross(start_to_end_vec, normal_vector)
            else:
                normal_vector = np.cross(start_to_end_vec, [0, 1, 0])
        normal_vector /= np.linalg.norm(normal_vector)

        # Finding two orthogonal unit vectors in the plane of the circle
        vec1 = start_to_end_vec / (2 * radius)
        vec2 = np.cross(normal_vector, vec1)

        # Generate points on the semicircle
        angle_range = np.linspace(np.pi, 0, npoints)  # Reversed angle range
        circle_points = []
        for angle in angle_range:
            point = midpoint + radius * (np.cos(angle) * vec1 + np.sin(angle) * vec2)
            circle_points.append(list(point))

        return circle_points

    # Select trajectory type
    if shape == "linear":
        return linear_trajectory()
    elif shape == "circular":
        return circular_trajectory()
    else:
        raise ValueError("Shape must be 'linear' or 'circular'.")


def get_label_list(folder_path):
    """
    modified from
    github.com/justinsalamon/scaper/master/scaper/util.py
    """

    label_list = []
    folder_names = os.listdir(folder_path)
    for fname in folder_names:
        if os.path.isdir(os.path.join(folder_path, fname)) and fname[0] != ".":
            label_list.append(fname)
    # ensure consistent ordering of labels
    label_list.sort()
    return label_list


def get_files_list(path, split):
    """
    Retrieves a list of file paths from a specified directory, optionally filtering by a subdirectory (split).

    This function searches for all files in a given directory and its subdirectories. If a 'split' is specified,
    it further narrows down the search to include files only from that particular subdirectory. This is particularly
    useful when dealing with datasets divided into splits like 'train', 'test', etc.

    Args:
        path (str): The base directory from which to retrieve file paths.
        split (str, optional): An optional subdirectory name within the base directory to filter the files.
                               If None, files from all subdirectories are included.

    Returns:
        list: A list of file paths (strings) that are found in the specified directory (and subdirectory, if specified).

    The function uses 'glob' to search recursively within the given directory. It filters out directories,
    ensuring only file paths are returned. If a 'split' is provided, the search is limited to that subdirectory.
    """
    if split:
        subfiles = glob.glob(os.path.join(path, split, "**"), recursive=True)
    else:
        subfiles = glob.glob(os.path.join(path, "**"), recursive=True)
    subfiles = [f for f in subfiles if os.path.isfile(f)]
    return subfiles


def new_event_exceeds_max_overlap(
    new_event_time, new_event_duration, other_events, max_overlap, increment
):
    """
    Checks if a new event overlaps with existing events more than a specified maximum amount.

    This function iterates through each time increment of the new event's duration and checks
    if it overlaps with other existing events. If the number of overlaps at any point exceeds
    'max_overlap', it indicates too much overlap.

    Args:
        new_event_time (float): The start time of the new event.
        new_event_duration (float): The duration of the new event.
        other_events (list): A list of other events to check for overlap. Each event in the list is expected to have
                             'event_time' and 'event_duration' attributes.
        max_overlap (int): The maximum number of events that can overlap at any time.
        increment (float): The time increment to check for overlap.

    Returns:
        bool: True if the new event overlaps with more than 'max_overlap' events at any point, False otherwise.

    The function is useful in scenarios where overlapping events are permissible to some extent,
    but excessive overlap needs to be avoided.
    """

    # Incrementally check each second of the new event's duration
    for t in np.arange(new_event_time, new_event_time + new_event_duration, increment):
        current_overlap = 0
        for event in other_events:
            if (
                t >= event.event_time - increment
                and t <= event.event_time + event.event_duration + increment
            ):
                current_overlap += 1
            if current_overlap > max_overlap - 1:
                return True  # Overlaps with more than max_overlap events
    return False  # Suitable time found


def count_leading_zeros_in_period(frequency_hz):
    """
    Counts the number of leading zeros in the decimal representation of the period of a frequency.

    This function is useful for determining the precision needed to represent the period of a frequency
    accurately in decimal form.

    Args:
        frequency_hz (float): The frequency in Hertz for which the period's leading zeros are to be counted.

    Returns:
        int: The number of leading zeros in the period of the given frequency.

    The function calculates the period as the reciprocal of the frequency, then converts it to a string format
    to count the leading zeros in its fractional part. The count stops at the first non-zero digit.
    """
    # Calculate the period
    period_seconds = 1 / frequency_hz

    # Convert the period to a string to find the leading zeros
    period_str = f"{period_seconds:.10f}"

    # Split the string at the decimal point and work with the fractional part
    fractional_part = period_str.split(".")[1]

    # Count leading zeros using a generator expression with a condition to stop after first non-zero digit
    return sum(
        1
        for i, digit in enumerate(fractional_part)
        if digit == "0" and "1" not in fractional_part[: i + 1]
    )
