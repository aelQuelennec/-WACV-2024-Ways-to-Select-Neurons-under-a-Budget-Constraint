import json
import re
# Path to text files
path_to_txt_train = './annotations/trainval.txt'
path_to_txt_test = './annotations/test.txt'

# Open text files and read data
with open(path_to_txt_train, 'r') as file:
    data_train = file.readlines()

with open(path_to_txt_test, 'r') as file:
    data_test = file.readlines()

import os
import shutil


# Move files to respective directories
for line in data_train:
    image, _, _, _ = line.strip().split()

    # Extract label and id from file name
    label = '_'.join([i for i in image.split('_') if not bool(re.search(r'\d', i))])

    source_path = os.path.join('./images', image + '.jpg')
    dest_path = os.path.join('./pets/train', label)
    os.makedirs(dest_path, exist_ok=True)

    shutil.copy(source_path, dest_path)
    # print("move " + source_path + " to " + dest_path)

for line in data_test:
    image, _, _, _ = line.strip().split()

    # Extract label and id from file name
    label = '_'.join([i for i in image.split('_') if not bool(re.search(r'\d', i))])

    source_path = os.path.join('./images', image + '.jpg')
    dest_path = os.path.join('./pets/test', label)
    os.makedirs(dest_path, exist_ok=True)
    
    shutil.copy(source_path, dest_path)
    #print("move " + source_path + " to " + dest_path)
