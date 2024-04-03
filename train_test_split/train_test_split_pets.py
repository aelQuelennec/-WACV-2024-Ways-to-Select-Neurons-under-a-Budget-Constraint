import re
import shutil
from tqdm import tqdm
import os

from utils import download_data, extract_raw_data

if os.path.exists("./data/pets"):
    print("pets is already prepared !")
    exit()

# Downloading raw tar dataset
url_data = "https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz"
url_annotation = "https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz"

raw_data_folder = "./raw_data/pets"


download_data(url_data, raw_data_folder)
download_data(url_annotation, raw_data_folder)

# Extract raw data
raw_data_location = os.path.join(raw_data_folder, os.path.basename(url_data))
extract_raw_data(raw_data_location, raw_data_folder, "images")
raw_data_location = os.path.join(raw_data_folder, os.path.basename(url_annotation))
extract_raw_data(raw_data_location, raw_data_folder, "annotations")


# _____________________________________________________________________________________________________

# Path to text files
path_to_txt_train = "./raw_data/pets/annotations/trainval.txt"
path_to_txt_test = "./raw_data/pets/annotations/test.txt"

# Open text files and read data
with open(path_to_txt_train, "r") as file:
    data_train = file.readlines()

with open(path_to_txt_test, "r") as file:
    data_test = file.readlines()


# Move files to respective directories
with tqdm(
    total=len(data_train) + len(data_test), desc="Preparing data", unit=" file"
) as t:
    for line in data_train:
        image, _, _, _ = line.strip().split()

        # Extract label and id from file name
        label = "_".join([i for i in image.split("_") if not bool(re.search(r"\d", i))])

        source_path = os.path.join("./raw_data/pets/images", image + ".jpg")
        dest_path = os.path.join("./data/pets/train", label)
        os.makedirs(dest_path, exist_ok=True)

        shutil.copy(source_path, dest_path)
        # print("move " + source_path + " to " + dest_path)
        t.update(1)

    for line in data_test:
        image, _, _, _ = line.strip().split()

        # Extract label and id from file name
        label = "_".join([i for i in image.split("_") if not bool(re.search(r"\d", i))])

        source_path = os.path.join("./raw_data/pets/images", image + ".jpg")
        dest_path = os.path.join("./data/pets/test", label)
        os.makedirs(dest_path, exist_ok=True)

        shutil.copy(source_path, dest_path)
        # print("move " + source_path + " to " + dest_path)
        t.update(1)
print("Successfully prepared Pets data !")
