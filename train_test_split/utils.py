from tqdm import tqdm
from wget import download
import tarfile
import os

def download_data(url, saved_location):
    os.makedirs(saved_location, exist_ok=True)
    file_path = os.path.join(saved_location, os.path.basename(url))

    if not os.path.exists(file_path):
        print("Downloading...")
        progress_bar = tqdm(total=None)
        def update_progress_bar(chunk, file_handle, bytes_remaining):
            progress_bar.update(chunk)
        download(url, out=saved_location, bar=update_progress_bar)
        progress_bar.close()
        print(f"Successfully downloaded. File is saved in '{os.path.join(saved_location, os.path.basename(url))}'.")
    else:
        print(f"{os.path.basename(url)} is already available !")

def extract_raw_data(raw_data_location, destination, name):
    if not os.path.exists(os.path.join(destination, name)):
        with tarfile.open(raw_data_location, 'r:gz') as tar:
            total_files = sum(1 for _ in tar.getmembers())

            with tqdm(total=total_files, desc='Extracting', unit='file') as pbar:
                for member in tar:
                    tar.extract(member, destination)
                    pbar.update(1)

        print(f"Extracted {raw_data_location} to {destination}.")
    else:
        print(f"{name} is already extracted")