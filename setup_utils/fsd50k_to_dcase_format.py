import os
import shutil

# Path to the DCASE .txt containing directory structure
FSD50K_DCASE_DIR_STRUCT_TXT = "/scratch/data/FSD50K/FSD50K_DCASE/FSD50K_selected.txt"
# Source path FSD50K directory
FSD50K_SRC_PATH = "/scratch/data/FSD50K/"
# Destination path FSD50K to DCASE directory structure
FSD50K_DEST_PATH = "/scratch/data/FSD50K/FSD50K_DCASE"

# Read the lines in the .txt file
with open(FSD50K_DCASE_DIR_STRUCT_TXT, 'r') as f:
    lines = f.readlines()

# Loop through each line in the .txt file
for line in lines:
    new_dir = os.path.dirname(line.strip())
    filename = os.path.basename(line.strip())
    print("Processing file to new directory:", new_dir)
    # Retrieve tnhe new source directory for the file
    if 'train' in new_dir:
        source = os.path.join(FSD50K_SRC_PATH, 'FSD50K.dev_audio')
    elif 'test' in new_dir:
        source = os.path.join(FSD50K_SRC_PATH, 'FSD50K.eval_audio')
    else:
        raise ValueError('Invalid directory structure in .txt file')
    # full path of the source file
    src_file = os.path.join(source, filename)
    # new directory, check if it doesn't exist
    os.makedirs(os.path.join(FSD50K_DEST_PATH, new_dir), exist_ok=True)
    # copy file to new directory
    shutil.copy(src_file, os.path.join(FSD50K_DEST_PATH, new_dir, filename))
