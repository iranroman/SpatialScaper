"""
This script use audio channel swapping to augment audio data and metadata in training sets from given folders.
"""
import os
import random
import shutil

import numpy as np
import scipy.io.wavfile as wav


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        print('{} folder does not exist, creating it.'.format(folder_name))
        os.makedirs(folder_name)


def acs_mic(audio):
    # separate channels
    chan_1 = audio[:, 0]
    chan_2 = audio[:, 1]
    chan_3 = audio[:, 2]
    chan_4 = audio[:, 3]

    # swapping columns
    audio_aug = []
    audio_aug.append(np.dstack((chan_2, chan_4, chan_1, chan_3)))
    audio_aug.append(np.dstack((chan_4, chan_2, chan_3, chan_1)))
    audio_aug.append(np.dstack((chan_2, chan_1, chan_4, chan_3)))
    audio_aug.append(np.dstack((chan_3, chan_1, chan_4, chan_2)))
    audio_aug.append(np.dstack((chan_1, chan_3, chan_2, chan_4)))
    audio_aug.append(np.dstack((chan_4, chan_3, chan_2, chan_1)))
    audio_aug.append(np.dstack((chan_3, chan_4, chan_1, chan_2)))

    return audio_aug


def acs_foa(audio):
    # separate channels
    chan_1 = audio[:, 0]
    chan_2 = audio[:, 1]
    chan_3 = audio[:, 2]
    chan_4 = audio[:, 3]

    # swapping columns
    audio_aug = []
    audio_aug.append(np.dstack((chan_1, -chan_4, -chan_3, chan_2)))
    audio_aug.append(np.dstack((chan_1, -chan_4, chan_3, -chan_2)))
    audio_aug.append(np.dstack((chan_1, -chan_2, -chan_3, chan_4)))
    audio_aug.append(np.dstack((chan_1, chan_4, -chan_3, chan_2)))
    audio_aug.append(np.dstack((chan_1, chan_4, chan_3, chan_2)))
    audio_aug.append(np.dstack((chan_1, -chan_2, chan_3, -chan_4)))
    audio_aug.append(np.dstack((chan_1, chan_2, -chan_3, -chan_4)))

    return audio_aug


def acs_meta(csv_data):
    frame = csv_data[:, 0]
    id = csv_data[:, 1]
    source = csv_data[:, 2]
    azimuth = csv_data[:, 3]
    elevation = csv_data[:, 4]
    distance = csv_data[:, 5] if csv_data.shape[1] > 5 else np.full(csv_data.shape[0], None)

    # transform azimuth and elevation
    label_aug = []
    label_aug.append(np.dstack((frame, id, source, azimuth - 90, -elevation, distance)))
    label_aug.append(np.dstack((frame, id, source, -azimuth - 90, elevation, distance)))
    label_aug.append(np.dstack((frame, id, source, -azimuth, -elevation, distance)))
    label_aug.append(np.dstack((frame, id, source, azimuth + 90, -elevation, distance)))
    label_aug.append(np.dstack((frame, id, source, -azimuth + 90, elevation, distance)))
    label_aug.append(np.dstack((frame, id, source, azimuth + 180, elevation, distance)))
    label_aug.append(np.dstack((frame, id, source, -azimuth + 180, -elevation, distance)))

    return label_aug


def acs(data_dir, aug_dir):
    """

    Args:
        data_dir: A database of training audio data and metadata.
        aug_dir: A new database of augmented data.

    Returns:
        Create a new folder of 7 types of channel swapping data for each audio in the original database.
    """

    # determine augmentation method
    if "mic" in data_dir:
        aug_fx = acs_mic
    elif "foa" in data_dir:
        aug_fx = acs_foa
    elif "metadata" in data_dir:
        aug_fx = acs_meta
    else:
        raise NotImplementedError("The augmentation method for this data folder is not found.")

    for sub_folder in os.listdir(data_dir):
        if "train" not in sub_folder:
            continue

        loc_desc_folder = os.path.join(data_dir, sub_folder)
        loc_aug_folder = os.path.join(aug_dir, sub_folder + "-aug-acs")
        create_folder(loc_aug_folder)

        print("Start augmenting audio in folder {} to folder {}".format(loc_desc_folder, loc_aug_folder))

        for file_cnt, file_name in enumerate(os.listdir(loc_desc_folder)):
            filename = file_name.split('.')[0]
            file = os.path.join(loc_desc_folder, file_name)
            file_aug = os.path.join(loc_aug_folder, filename)

            if "metadata" in data_dir:
                data = np.genfromtxt(file, dtype=int, delimiter=',')
            else:
                fs, data = wav.read(file)

            # augmentation
            audio_aug = aug_fx(data)

            for i in range(1, 8):
                if "metadata" in data_dir:
                    np.savetxt(file_aug + "_aug_acs_{}.csv".format(i), audio_aug[i - 1].squeeze(), delimiter=',',
                               fmt='%s')
                else:
                    wav.write(file_aug + "_aug_acs_{}.wav".format(i), fs, audio_aug[i - 1].squeeze())

    print("Completed augmentation in {}".format(data_dir))


def rand_sample(base_dir, subset_dir):
    """

    Args:
        base_dir: A folder that contains all real, synthetic, and augmented data.

    Returns:
        Create a new folder with a subset of all data.

    """
    random.seed(29)
    create_folder(subset_dir)
    src_path = os.path.join(base_dir, "mic_dev")

    for folder in os.listdir(src_path):
        if "train" in folder:
            # get subfolder name
            src_mic = os.path.join(src_path, folder)
            dst_mic = os.path.join(subset_dir, "mic_dev", folder)

            src_foa = os.path.join(base_dir, "foa_dev", folder)
            dst_foa = os.path.join(subset_dir, "foa_dev", folder)

            src_meta = os.path.join(base_dir, "metadata_dev", folder)
            dst_meta = os.path.join(subset_dir, "metadata_dev", folder)

            create_folder(dst_mic)
            create_folder(dst_meta)
            create_folder(dst_foa)

            # select the same 1/8 portion of files from both foa and mic folders
            filenames = random.sample(os.listdir(src_mic), round(len(os.listdir(src_mic)) * 0.125))

            # copy all files to a new folder
            for fname in filenames:
                # copy mic data
                src_mic_file = os.path.join(src_mic, fname)
                dst_mic_file = os.path.join(dst_mic, fname)
                shutil.copyfile(src_mic_file, dst_mic_file)

                # copy foa data
                src_foa_file = os.path.join(src_foa, fname)
                dst_foa_file = os.path.join(dst_foa, fname)
                shutil.copyfile(src_foa_file, dst_foa_file)

                # copy corresponding metadata
                src_label = os.path.join(src_meta, os.path.splitext(fname)[0] + ".csv")
                dst_label = os.path.join(dst_meta, os.path.splitext(fname)[0] + ".csv")
                shutil.copyfile(src_label, dst_label)


if __name__ == "__main__":
    base_dir = "/datasets/STARSS2023/"
    aug_dir = "/datasets/starss2023_aug_acs/"
    subset_dir = "/datasets/STARSS2023_subset"

    foa_dir = base_dir + "foa_dev"
    meta_dir = base_dir + "metadata_dev"
    mic_dir = base_dir + "mic_dev"

    foa_dir_aug = aug_dir + "foa_dev"
    meta_dir_aug = aug_dir + "metadata_dev"
    mic_dir_aug = aug_dir + "mic_dev"

    # acs(mic_dir, mic_dir_aug)
    # acs(foa_dir, foa_dir_aug)
    # acs(meta_dir, meta_dir_aug)

    rand_sample(base_dir, subset_dir)
    rand_sample(aug_dir, subset_dir)
