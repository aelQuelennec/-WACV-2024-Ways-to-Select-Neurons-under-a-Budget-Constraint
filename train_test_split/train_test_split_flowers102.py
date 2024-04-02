from scipy.io import loadmat
import shutil
from tqdm import tqdm
import os

from utils import download_data, extract_raw_data


if os.path.exists("./data/flowers102"):
    print("flowers102 is already prepared !")
    exit()

# Downloading raw tar dataset
url_data = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
url_label = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'
url_split = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat'

raw_data_folder = './raw_data/flowers102'


download_data(url_data, raw_data_folder)
download_data(url_label, raw_data_folder)
download_data(url_split, raw_data_folder)

# Extract raw data
raw_data_location = os.path.join(raw_data_folder, os.path.basename(url_data))
extract_raw_data(raw_data_location, raw_data_folder, "jpg")

#_____________________________________________________________________________________________________
# Read .mat files
data = loadmat('./raw_data/flowers102/setid.mat')
labels = loadmat('./raw_data/flowers102/imagelabels.mat')['labels']
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



image_folder = "./raw_data/flowers102/jpg"
image_list = os.listdir(image_folder)

with tqdm(total=len(image_list), desc='Preparing data', unit=' file') as t:
    for i in range(len(image_list)):
        image_ID = '{:05}'.format(i+1)
        source_path = os.path.join(image_folder, "image_" + image_ID + ".jpg")
        
        if image_ID in data_train:
            dest_path = './data/flowers102/train/' + str(labels[0][i])
            os.makedirs(dest_path, exist_ok=True)
            shutil.copy(source_path, dest_path)

        elif image_ID in data_valid:
            dest_path = './data/flowers102/val/' + str(labels[0][i])
            os.makedirs(dest_path, exist_ok=True)
            shutil.copy(source_path, dest_path)
        elif image_ID in data_test:
            dest_path = './data/flowers102/test/' + str(labels[0][i])
            os.makedirs(dest_path, exist_ok=True)
            shutil.copy(source_path, dest_path)
        t.update(1)
print("Successfully prepared Flowers dataset !")