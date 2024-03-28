import json

# Path to JSON files
path_to_json_train = './meta/train.json'
path_to_json_test = './meta/test.json'

# Open JSON files and read data
with open(path_to_json_train, 'r') as file:
    data_train = json.load(file)

with open(path_to_json_test, 'r') as file:
    data_test = json.load(file)

import os
import shutil

# Move files to respective directories
for item in data_train:
    dest_path = os.path.join('./food101/train', item)
    os.makedirs(dest_path, exist_ok=True)
    for image in data_train[item]:
        source_path = os.path.join('./images', image + ".jpg")
        shutil.move(source_path, dest_path)
        # print("move " + source_path + " to " + dest_path)

for item in data_test:
    dest_path = os.path.join('./food101/test', item)
    os.makedirs(dest_path, exist_ok=True)
    for image in data_test[item]:
        source_path = os.path.join('./images', image + ".jpg")
        shutil.move(source_path, dest_path)
        # print("move " + source_path + " to " + dest_path)