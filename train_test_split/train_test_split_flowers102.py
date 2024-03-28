from re import I
from scipy.io import loadmat

# Read .mat files
data = loadmat('setid.mat')
labels = loadmat('imagelabels.mat')['labels']
# Access variables and data in the data dictionary
trnid = data['trnid']
tstid = data['tstid']
valid = data['valid']

# Create 5-digit IDs for train, val, test
data_train = []
for line in trnid[0]:
    data_train.append(f'{int(line):05d}')

data_valid = []
for line in valid[0]:
    data_valid.append(f'{int(line):05d}')

data_test = []
for line in tstid[0]:
    data_test.append(f'{int(line):05d}')

import os
import shutil


image_folder = "./jpg"
image_list = os.listdir(image_folder)

for i in range(len(image_list)):
    image_ID = '{:05}'.format(i+1)
    source_path = os.path.join(image_folder, "image_" + image_ID + ".jpg")
    
    if image_ID in data_train:
        dest_path = './flowers102/train/' + str(labels[0][i])
        os.makedirs(dest_path, exist_ok=True)
        shutil.copy(source_path, dest_path)

    elif image_ID in data_valid:
        dest_path = './flowers102/val/' + str(labels[0][i])
        os.makedirs(dest_path, exist_ok=True)
        shutil.copy(source_path, dest_path)
    elif image_ID in data_test:
        dest_path = './flowers102/test/' + str(labels[0][i])
        os.makedirs(dest_path, exist_ok=True)
        shutil.copy(source_path, dest_path)