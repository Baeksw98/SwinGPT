import os
import orjson
from swingpt.utils.data_utils import save_to_json, get_config
from tqdm import tqdm
from swingpt.utils.constants import *
from joblib import Parallel, delayed

class LVISDataset:
    def __init__(self, cfg, versions=['v0.5', 'v1'], splits=['train', 'val']):
        self.cfg = cfg
        self.versions = versions
        self.splits = splits
        self.data_dir = self.cfg.main.datasets.data_dir 
        self.processed_dir = self.cfg.main.datasets.processed_dir  

    def load_lvis_data(self, version, split):
        lvis_file_path = os.path.join(self.data_dir, 'coco', 'annotations', f'lvis_{version}_{split}.json')
        with open(lvis_file_path, 'rb') as file:  
            return orjson.loads(file.read())

    def process_data_split(self, version, split):
        lvis_data = self.load_lvis_data(version, split)
        combined_list = []
        categories = {cat['id']: cat['name'] for cat in lvis_data['categories']}

        for image in tqdm(lvis_data['images'], desc=f"Processing images for {version}-{split}"):
            img_id = image['id']
            coco_url = image['coco_url']

            # Extract sub-folder and filename from the coco_url
            path_parts = coco_url.split('/')
            sub_folder = path_parts[-2]
            file_name = path_parts[-1]
            image_path = os.path.join(self.data_dir, 'coco', 'images', sub_folder, file_name)
            
            # Check if the image file exists
            if os.path.exists(image_path): 
                annotations = [ann for ann in lvis_data['annotations'] if ann['image_id'] == img_id]
                
                for ann in annotations:
                    # Directly formatting in line without pre-calculation
                    combined_dict = {
                        'conversations': [
                            {'from': 'human', 'value': '{}{}{}'.format(DEFAULT_DETECT_TOKEN, categories[ann['category_id']], DEFAULT_EOP_TOKEN)},
                            {'from': 'gpt', 'value': '({},{},{},{})'.format(ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3])}
                        ],
                        'id': img_id,
                        'image': image_path,
                        'category_ids': ann['category_id']
                    }
                    combined_list.append(combined_dict)

        save_to_json(combined_list, "lvis_{}_{}_combined.json".format(split, version), self.processed_dir)

    def process(self):
        tasks = [(version, split) for version in self.versions for split in self.splits]
        # Use joblib for parallel processing & tqdm for progress bar
        Parallel(n_jobs=-1)(delayed(self.process_data_split)(version, split) for version, split in tqdm(tasks, desc="Processing LVIS datasets"))

if __name__ == "__main__":
    # Load config path and create LVISDataset instance
    data_config_path = "/data/cad-recruit-02_814/swbaek/config/config_datasets.yaml"      
    cfg = get_config(data_config_path)
    lvis_dataset = LVISDataset(cfg=cfg)

    # Process datasets
    lvis_dataset.process()