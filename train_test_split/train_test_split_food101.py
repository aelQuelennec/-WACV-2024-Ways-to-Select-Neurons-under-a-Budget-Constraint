import json
import os
import shutil
from tqdm import tqdm



from utils import download_data, extract_raw_data

if os.path.exists("./data/food101"):
    print("food101 is already prepared !")
    exit()

# Downloading raw tar dataset
url_data = 'http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz'

raw_data_folder = './raw_data/food101'


download_data(url_data, raw_data_folder)

# Extract raw data
raw_data_location = os.path.join(raw_data_folder, os.path.basename(url_data))
extract_raw_data(raw_data_location, raw_data_folder, "food-101")


#_____________________________________________________________________________________________________

# Path to JSON files
path_to_json_train = './raw_data/food101/food-101/meta/train.json'
path_to_json_test = './raw_data/food101/food-101/meta/test.json'

# Open JSON files and read data
with open(path_to_json_train, 'r') as file:
    data_train = json.load(file)

with open(path_to_json_test, 'r') as file:
    data_test = json.load(file)



# Move files to respective directories
with tqdm(total=len(data_train) + len(data_test), desc='Preparing data', unit=' file') as t:
    for item in data_train:
        dest_path = os.path.join('./data/food101/train', item)
        os.makedirs(dest_path, exist_ok=True)
        for image in data_train[item]:
            source_path = os.path.join('./raw_data/food101/food-101/images', image + ".jpg")
            shutil.move(source_path, dest_path)
            # print("move " + source_path + " to " + dest_path)
            t.update(1)

    for item in data_test:
        dest_path = os.path.join('./data/food101/test', item)
        os.makedirs(dest_path, exist_ok=True)
        for image in data_test[item]:
            source_path = os.path.join('./raw_data/food101/food-101/images', image + ".jpg")
            shutil.move(source_path, dest_path)
            # print("move " + source_path + " to " + dest_path)
            t.update(1)
print("Successfully prepared food101 data !")