import orjson
import os
from pycocotools.coco import COCO
from swingpt.utils.data_utils import save_to_json, get_config
from swingpt.utils.constants import *
from tqdm import tqdm
from joblib import Parallel, delayed

class RefCOCODataset:
    def __init__(self, cfg, datasets=['refcoco', 'refcoco+', 'refcocog'], splits=['train', 'val']):
        self.cfg = cfg
        self.datasets = datasets
        self.splits = splits
        self.data_dir = self.cfg.main.datasets.data_dir
        self.processed_dir = self.cfg.main.datasets.processed_dir

    def load_refcoco_data(self, file_path):
        with open(file_path, 'rb') as file:  
            return orjson.loads(file.read())

    def process_refcoco(self, dataset, split):
        file_name = f'finetune_{dataset}_{split}.json'
        ref_coco_file = os.path.join(self.data_dir, 'refcoco_ann', 'annotations', file_name)
        ref_coco_dict = self.load_refcoco_data(ref_coco_file)
        category_dict = {item['id']: item['name'] for item in ref_coco_dict['categories']}
        ref_coco_caption_list, ref_coco_detect_list = [], []

        for annotation in tqdm(ref_coco_dict['annotations'], desc="Processing Annotations"):
            image = next((img for img in ref_coco_dict['images'] if img['id'] == annotation['image_id']), None)
            if image:
                image_path = self.construct_image_path(image)
                # Check if the image file actually exists in the path
                if os.path.exists(image_path):  
                    caption_dict, detect_dict = self.create_dicts(annotation, category_dict, image)
                    ref_coco_caption_list.append(caption_dict)
                    ref_coco_detect_list.append(detect_dict)
                else:
                    print(f"Image not found at {image_path}")
            else:
                print("Image metadata not found")

        # Save lists to JSON
        save_to_json(ref_coco_caption_list, f"{dataset}_{split}_caption.json", self.processed_dir)
        save_to_json(ref_coco_detect_list, f"{dataset}_{split}_detect.json", self.processed_dir)

    def create_dicts(self, annotation, category_dict, image):
        # Get imgae path
        image_path = self.construct_image_path(image)

        common_dict = {
            'id': annotation['image_id'],
            'image': image_path,
            'category_ids': annotation['category_id'],
        }

        caption_dict = {
            **common_dict,
            'conversations': [
                {'from': 'human', 'value': '{}({},{},{},{}){}'.format(DEFAULT_CAPTION_TOKEN, *annotation['bbox'], DEFAULT_EOP_TOKEN)},
                {'from': 'gpt', 'value': category_dict[annotation['category_id']]}
            ]
        }
        
        detect_dict = {
            **common_dict,
            'conversations': [
                {'from': 'human', 'value': '{}{}{}'.format(DEFAULT_DETECT_TOKEN, category_dict[annotation['category_id']], DEFAULT_EOP_TOKEN)},
                {'from': 'gpt', 'value': '({},{},{},{})'.format(*annotation['bbox'])}
            ]
        }
        return caption_dict, detect_dict
    
    def construct_image_path(self, image):
        # Correctly extract the full numeric ID from the filename
        dataset_type = image['file_name'].split('_')[1]  # Extract folder type if needed
        return os.path.join(self.data_dir, 'coco', 'images', dataset_type, image['file_name'])

    def process(self):
        tasks = [(dataset, split) for dataset in self.datasets for split in self.splits]
        # Use joblib for parallel processing & tqdm for progress bar
        Parallel(n_jobs=-1)(delayed(self.process_refcoco)(dataset, split) for dataset, split in tqdm(tasks, desc="Processing RefCOCO datasets"))

if __name__ == "__main__":
    # Load config path and create COCODetectDataset instance
    data_config_path = "/data/cad-recruit-02_814/swbaek/config/config_datasets.yaml"      
    cfg = get_config(data_config_path)
    refcoco_dataset = RefCOCODataset(cfg)

    # Process datasets
    refcoco_dataset.process()