import os
import shutil
import re


from tqdm import tqdm
from utils import download_data, extract_raw_data


if os.path.exists("./data/cub200"):
    print("cub200 is already prepared !")
    exit()

# Downloading raw tar dataset
url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz'
raw_data_folder = './raw_data'
os.makedirs(raw_data_folder, exist_ok=True)
download_data(url, raw_data_folder)

# Extract raw data
raw_data_location = os.path.join(raw_data_folder, os.path.basename(url))
extract_raw_data(raw_data_location, raw_data_folder, "CUB_200_2011")

#_____________________________________________________________________________________________________
# Read information from train_test_split.txt
with open('./raw_data/CUB_200_2011/train_test_split.txt', 'r') as split_file:
    split_lines = split_file.readlines()

# Read information from images.txt
with open('./raw_data/CUB_200_2011/images.txt', 'r') as images_file:
    images_lines = images_file.readlines()

# Create a dictionary to store index information from train_test_split.txt
index_dict = {}
for line in split_lines:
    image_id, index = line.strip().split()
    index_dict[image_id] = int(index)


# Move files to respective directories
with tqdm(total=len(images_lines), desc='Preparing data', unit=' file') as t:
    for line in images_lines:
        image_id, image_name = line.strip().split()
        index = index_dict.get(image_id, -1)  # Use -1 if index is not found
        
        # Extract label and id from file name
        label = '_'.join([i for i in image_name.split('/')[1].split('_') if not bool(re.search(r'\d', i))])
        
        source_path = os.path.join('./raw_data/CUB_200_2011/images', image_name)
        
        if index == 1:
            dest_path = os.path.join('./data/cub200/train', label)
            # Create destination directory if it doesn't exist
            os.makedirs(dest_path, exist_ok=True)
            shutil.copy(source_path, dest_path)
            #print("move " + source_path + " to " + dest_path)
        elif index == 0:
            dest_path = os.path.join('./data/cub200/test', label)
            os.makedirs(dest_path, exist_ok=True)
            shutil.copy(source_path, dest_path)
            #print("move " + source_path + " to " + dest_path)
        t.update(1)

print("Successfully prepared CUB dataset !")