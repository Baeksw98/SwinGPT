import orjson
import os
from tqdm import tqdm
from swingpt.utils.data_utils import get_config

from swingpt.datasets.raw_data.coco_caption import COCOCaptionDataset
from swingpt.datasets.raw_data.coco_detect import COCODetectDataset
from swingpt.datasets.raw_data.refcoco import RefCOCODataset
from swingpt.datasets.raw_data.vg import VGDataset
from swingpt.datasets.raw_data.lvis import LVISDataset
from swingpt.utils.constants import *

class DatasetConcatenator:
    """ A utility for concatenating multiple datasets into a single large JSON file. """
    
    def __init__(self, config_path):
        self.cfg = get_config(config_path)
        self.processed_dir = self.cfg.main.datasets.processed_dir
        self.combined_dir = os.path.join(self.processed_dir, "combined")
        os.makedirs(self.combined_dir, exist_ok=True)
            
    def concatenate_datasets(self, dataset_paths):
        """ Concatenate JSON data from multiple dataset files. """
        combined_data = []
        for path in tqdm(dataset_paths, desc="Concatenating datasets"):
            with open(path, 'rb') as file:
                data = orjson.loads(file.read())
                combined_data.extend(data)
        return combined_data

    def save_dataset(self, combined_data, save_path):
        """ Save the combined dataset to a specified path in JSON format. """
        with open(save_path, 'wb') as file:
            file.write(orjson.dumps(combined_data))

    def process_datasets(self, category, save_filename):
        """ Load, concatenate, and save datasets for a specific category. """
        print(f"Processing category: {category}")
        dataset_files = getattr(self.cfg.main.datasets, category)
        dataset_paths = [os.path.join(self.processed_dir, filename) for filename in dataset_files]
        combined_data = self.concatenate_datasets(dataset_paths)
        save_path = os.path.join(self.combined_dir, save_filename)
        self.save_dataset(combined_data, save_path)
        print(f"Saved {save_filename} with {len(combined_data)} items.")

def process_individual_datasets(cfg):
    """ Process individual datasets if not already processed. """
    print("Processing individual datasets....")
    coco_caption_dataset = COCOCaptionDataset(cfg)
    coco_detect_dataset = COCODetectDataset(cfg)
    refcoco_dataset = RefCOCODataset(cfg)
    vg_dataset = VGDataset(cfg)
    lvis_dataset = LVISDataset(cfg)

    coco_caption_dataset.process()
    coco_detect_dataset.process()
    refcoco_dataset.process()
    vg_dataset.process()
    lvis_dataset.process()

    print("All data are processed and stored in the data_processed folder.")

def main(config_path):
    """ Main function to process and concatenate datasets. """
    cfg = get_config(config_path)
    concatenator = DatasetConcatenator(config_path)

    if cfg.main.create_data:
        process_individual_datasets(cfg)
    else:
        print("Individual data are already processed. Skip!")

    print("Processing dataset concatenations....")
    # Concatenating train datasets
    # concatenator.process_datasets('train_MSCOCO', 'train_MSCOCO.json')
    # concatenator.process_datasets('train_refCOCO', 'train_refCOCO.json')
    # concatenator.process_datasets('train_VG', 'train_VG.json')
    # concatenator.process_datasets('train_LVIS', 'train_LVIS.json')
    # concatenator.process_datasets('train_all', 'train_all.json')
    concatenator.process_datasets('train_detect', 'train_detect.json')

    # # Concatenating validation datasets
    # concatenator.process_datasets('val_IC', 'val_IC.json')
    # concatenator.process_datasets('val_OD', 'val_OD.json')
    # concatenator.process_datasets('val_all', 'val_all.json')
    
    print("Finished concatenating datasets, saved in the data_processed/combined folder.")

if __name__ == "__main__":
    data_config_path = "/data/cad-recruit-02_814/swbaek/config/config_datasets.yaml"      
    main(data_config_path)
