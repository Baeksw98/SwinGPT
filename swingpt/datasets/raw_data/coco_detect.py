import os
from pycocotools.coco import COCO
from swingpt.utils.data_utils import save_to_json, get_config
from swingpt.utils.constants import *
from tqdm import tqdm
from joblib import Parallel, delayed

class COCODetectDataset:
    def __init__(self, cfg, years=['2014', '2017'], splits=['train', 'val']):
        self.cfg = cfg
        self.years = years
        self.splits = splits
        self.data_dir = self.cfg.main.datasets.data_dir
        self.processed_dir = self.cfg.main.datasets.processed_dir

    def process_data_split(self, year, split):
        # Adjust file naming to accommodate different splits
        detect_file = os.path.join(self.data_dir, f'coco/annotations/instances_{split}{year}.json')
        file_name = detect_file.split('.')[-2].split('/')[-1]
        coco = COCO(detect_file)
        
        # Process category information for detections
        category_dict = {item['id']: item['name'] for item in coco.loadCats(coco.getCatIds())}
        
        # Generate detection and caption items list
        detection_list = self.generate_detection_list(coco, category_dict, year, split)
        caption_list = self.generate_caption_list(coco, category_dict, year, split)
        
        # Save detections and captions to JSON using the utility function
        save_to_json(detection_list, f"{file_name}_detect.json", self.processed_dir)
        save_to_json(caption_list, f"{file_name}_caption.json", self.processed_dir)
    
    def generate_detection_list(self, coco, category_dict, year, split):
        detection_list = []
        for ann_id in tqdm(coco.anns.keys(), desc="Processing Detections"):
            ann = coco.anns[ann_id]
            img_info = coco.loadImgs(ann['image_id'])[0]
            image_path = os.path.join(self.data_dir, f'coco/images/{split}{year}', img_info["file_name"])
            
            # Check if image exists
            if os.path.exists(image_path):  
                detect_dict = {
                    'conversations': [
                        {'from': 'human', 'value': f'{DEFAULT_DETECT_TOKEN}{category_dict[ann["category_id"]]}{DEFAULT_EOP_TOKEN}'},
                        {'from': 'gpt', 'value': f'({ann["bbox"][0]},{ann["bbox"][1]},{ann["bbox"][2]},{ann["bbox"][3]})'}
                    ],
                    'id': ann['image_id'],
                    'image': image_path,
                    'category_ids': ann['category_id']
                }
                detection_list.append(detect_dict)
        
        return detection_list
    
    def generate_caption_list(self, coco, category_dict, year, split):
        caption_list = []
        for ann_id in tqdm(coco.anns.keys(), desc="Processing Captions"):
            ann = coco.anns[ann_id]
            img_info = coco.loadImgs(ann['image_id'])[0]
            image_path = os.path.join(self.data_dir, f'coco/images/{split}{year}', img_info["file_name"])
            # Check if image exists
            if os.path.exists(image_path):  
                caption_dict = {
                    'conversations': [
                        {'from': 'human', 'value': f'{DEFAULT_CAPTION_TOKEN}({ann["bbox"][0]},{ann["bbox"][1]},{ann["bbox"][2]},{ann["bbox"][3]}){DEFAULT_EOP_TOKEN}'},
                        {'from': 'gpt', 'value': category_dict[ann['category_id']]}
                    ],
                    'id': ann['image_id'],
                    'image': image_path,
                    'category_ids': ann['category_id']
                }
                caption_list.append(caption_dict)
        
        return caption_list
    
    def process(self):
        tasks = [(year, split) for year in self.years for split in self.splits]
        # Use joblib for parallel processing & tqdm for progress bar
        Parallel(n_jobs=-1)(delayed(self.process_data_split)(year, split) for year, split in tqdm(tasks, desc="Processing COCO Detection datasets"))

if __name__ == "__main__":
    # Load config path and create COCODetectDataset instance
    data_config_path = "/data/cad-recruit-02_814/swbaek/config/config_datasets.yaml"      
    cfg = get_config(data_config_path)
    coco_dataset = COCODetectDataset(cfg)

    # Process datasets
    coco_dataset.process()