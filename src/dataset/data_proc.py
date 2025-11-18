import os
import requests
import zipfile
import random
import csv
import argparse
from tqdm import tqdm
from typing import Literal, List
import shutil
from omegaconf import OmegaConf


def move_to_root(display:bool):
    path = os.getcwd().split('/')[::-1]
    if 'image_colorizer' in path:
        for folder in path:
            if folder != 'image_colorizer':
                os.chdir('..')
            else:
                if display:
                    print(f"Moved to root directory of the project:{os.getcwd()}")
                return
    raise NameError(f"Can not move to root of the project({os.getcwd()})")

class Data_processor():
    def __init__(self, data_format:Literal["rgb", "tif", "both"]="rgb"):
        self.datasets_info = {}

        if data_format == 'both':
            self.datasets_info['rgb'] = ('https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1', 'EuroSAT_RGB')
            self.datasets_info['tif'] = ('https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1', 'EuroSAT_MS')
        elif data_format == 'rgb':
            self.datasets_info['rgb'] = ('https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1', 'EuroSAT_RGB')
        elif data_format == 'tif':
            self.datasets_info['tif'] = ('https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1', 'EuroSAT_MS')
    
    def _move_to_raw(self):
        #Move to root of the project
        move_to_root(display=False)

        #Create and change the currntly working direcory
        os.makedirs('data/raw', exist_ok=True)
        os.chdir('data/raw')
        print(f"Moved to {os.getcwd()}")
    
    def download_zip(self):
        #Move to data/raw
        self._move_to_raw()
        
        print("Downloading EuroSAT dataset...")
        
        for data_format in self.datasets_info.keys():

            dataset_url, dataset_name = self.datasets_info[data_format]
            dataset_zip = f"{dataset_name}.zip"
            
            if not os.path.exists(dataset_zip):
                response = requests.get(dataset_url, stream=True)
                total_size = int(response.headers.get('content-length', 0))

                with open(dataset_zip, 'wb') as file, tqdm(
                    desc=f"Downloading {dataset_zip}",
                    total=total_size,
                    unit='iB',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for data in response.iter_content(chunk_size=1024):
                        size = file.write(data)
                        bar.update(size)
                        
    def unzip(self):
        #Move to data/raw
        self._move_to_raw()
        
        for data_format in self.datasets_info.keys():
            
            _, dataset_name = self.datasets_info[data_format]
            
            if not os.path.exists(dataset_name):
                print(f"Extracting files from {dataset_name}.zip") 
                
                with zipfile.ZipFile(f"{dataset_name}.zip", 'r') as zip_ref:
                    zip_ref.extractall('.')
    
    def create_prep_rgb(self):
        #Moving to root
        move_to_root(display=False)
        
        #Creating folders
        folder_name = self.datasets_info['rgb'][1]
        prep_data_folder = f'data/prep/{folder_name}' #Folder for storing all preprocessed data
        images_folder = f'{prep_data_folder}/images' #Folder for storing images
        os.makedirs(images_folder, exist_ok=True)
        
        #Iterating through raw pictures -> [{src_path:..., file_name: ..., original_class: ...}]
        print(f"Creating preprocessed dataset in {prep_data_folder}...")
        all_image_info: List[dict] = []
        raw_folder = f'data/raw/{folder_name}'
        for subdir, _, files in os.walk(raw_folder):
            class_name = os.path.basename(subdir)
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg')):
                    all_image_info.append({
                        'src_path': os.path.join(subdir, file),  # data/raw/class/file_name
                        'file_name': file,
                        'original_class': class_name if class_name != folder_name else 'unknown'
                    })
        print(f"Found {len(all_image_info)} images!")
        
        #Shuffling the images
        random.shuffle(all_image_info)
        print("Shuffled images!")        
        
        #Copying shuffled images to data/prep/EuroSAT_RGB/images
        for i, info in tqdm(enumerate(all_image_info, 1), total=len(all_image_info), desc="Transferring shuffled images"):
            src_path = os.path.abspath(info['src_path'])
            dst_path = os.path.join(images_folder, info['file_name'])
            shutil.copy2(src_path, dst_path)
        
        #Creating csv with image info in data/prep/EuroSAT_RGB
        print("Creating image_info.csv")
        csv_path = f'{prep_data_folder}/image_info.csv'
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['file_name', 'full_path', 'original_class']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for info in all_image_info:
                full_path = os.path.join(images_folder, info['file_name'])
                writer.writerow({
                    'file_name': info['file_name'],
                    'full_path': full_path,
                    'original_class': info['original_class']
                })
        
        print("Data preprocessed finished in data/prep/EuroSAT_RGB")
    
    def remove_zip(self):
        self._move_to_raw()
        for data_format in self.datasets_info.keys():
            zip_file = self.datasets_info[data_format][1]
            zip_file = f"{zip_file}.zip"
            if os.path.exists(zip_file):
                os.remove(zip_file)
                print(f"Removed {zip_file}")
                
    def remove_raw(self):
        self._move_to_raw()
        for data_format in self.datasets_info.keys():
            folder_name = self.datasets_info[data_format][1]
            if os.path.exists(folder_name):
                shutil.rmtree(folder_name)
                print(f"Removed raw {folder_name}")
                    
if __name__ == '__main__':
    
    # Main parser of the file
    parser = argparse.ArgumentParser()
    
    subparsers = parser.add_subparsers(dest='command')
    download_parser = subparsers.add_parser('download') #subparser of download
    unzip_parser = subparsers.add_parser('unzip')       #subparser of unzip
    prep_parser = subparsers.add_parser('prep')         #subparser of preprocessing
    cleanup_parser = subparsers.add_parser('cleanup')   #subparser of cleaning up
    
    #getting arguments from argparser
    args = parser.parse_args()
    
    #reading parameters from params.yml
    try:
        params = OmegaConf.load('params.yaml')
        remove_zip = params.data_pipeline.get('remove_zip', False)
        remove_raw = params.data_pipeline.get('remove_raw', False)
    except FileNotFoundError:
        print("params.yaml not found, using remove_zip=remove_raw=False")
        remove_zip = False
        remove_raw = False
    
    data_processor = Data_processor()
    
    if args.command == 'download':
        data_processor.download_zip()
    elif args.command == 'unzip':
        data_processor.unzip()
    elif args.command == 'prep':
        data_processor.create_prep_rgb()
    elif args.command == 'cleanup':
        if remove_zip:
            data_processor.remove_zip()
        if remove_raw:
            data_processor.remove_raw()
    