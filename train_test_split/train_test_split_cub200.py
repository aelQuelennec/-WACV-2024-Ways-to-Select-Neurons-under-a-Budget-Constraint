import os
import shutil
import re

# Read information from train_test_split.txt
with open('train_test_split.txt', 'r') as split_file:
    split_lines = split_file.readlines()

# Read information from images.txt
with open('images.txt', 'r') as images_file:
    images_lines = images_file.readlines()

# Create a dictionary to store index information from train_test_split.txt
index_dict = {}
for line in split_lines:
    image_id, index = line.strip().split()
    index_dict[image_id] = int(index)


# Move files to respective directories
for line in images_lines:
    image_id, image_name = line.strip().split()
    index = index_dict.get(image_id, -1)  # Use -1 if index is not found
    
    # Extract label and id from file name
    label = '_'.join([i for i in image_name.split('/')[1].split('_') if not bool(re.search(r'\d', i))])
    
    source_path = os.path.join('./images', image_name)
    
    if index == 1:
        dest_path = os.path.join('./cub200/train', label)
        # Create destination directory if it doesn't exist
        os.makedirs(dest_path, exist_ok=True)
        shutil.copy(source_path, dest_path)
        #print("move " + source_path + " to " + dest_path)
    elif index == 0:
        dest_path = os.path.join('./cub200/test', label)
        os.makedirs(dest_path, exist_ok=True)
        shutil.copy(source_path, dest_path)
        #print("move " + source_path + " to " + dest_path)

#print("Files moved successfully.")