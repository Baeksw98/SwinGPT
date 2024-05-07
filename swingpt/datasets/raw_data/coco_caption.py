import os
from pycocotools.coco import COCO
from swingpt.utils.data_utils import save_to_json, get_config
from swingpt.utils.constants import *
from tqdm import tqdm
from joblib import Parallel, delayed

class COCOCaptionDataset:
    def __init__(self, cfg, years=['2014', '2017'], splits=['train', 'val']):
        self.cfg = cfg
        self.years = years
        self.splits = splits
        self.data_dir = self.cfg.main.datasets.data_dir
        self.processed_dir = self.cfg.main.datasets.processed_dir

    def process_data_split(self, year, split):
        caption_file = os.path.join(self.data_dir, f'coco/annotations/captions_{split}{year}.json')
        detect_file = os.path.join(self.data_dir, f'coco/annotations/instances_{split}{year}.json')
        
        coco_captions = COCO(caption_file)
        coco_detections = COCO(detect_file)

        # Existing combined list generation
        combined_list = self.generate_combined_list(coco_captions, coco_detections, year, split)
        save_to_json(combined_list, f"coco_{split}{year}_combined.json", self.processed_dir)
        
    def generate_combined_list(self, coco_captions, coco_detections, year, split):
        combined_list = []
        for img_id in tqdm(coco_captions.getImgIds(), desc="Processing Images for MSCOCO-caption"):
            img_info = coco_captions.loadImgs(img_id)[0]
            image_path = os.path.join(self.data_dir, f'coco/images/{split}{year}', img_info["file_name"])

            # Check if the image file exists
            if os.path.exists(image_path):  

                caption_anns = coco_captions.loadAnns(coco_captions.getAnnIds(imgIds=img_id))
                detection_anns = coco_detections.loadAnns(coco_detections.getAnnIds(imgIds=img_id))
                
                # Calculate full image bounding box dimensions
                full_img_bbox = f'(0,0,{img_info["width"]},{img_info["height"]})'
                                
                for caption_ann in caption_anns:
                    combined_dict = {
                        'conversations': [],
                        'id': img_id,
                        'image': image_path,
                    }
                    # Whole image caption
                    combined_dict['conversations'].append({'from': 'human', 'value': f'{DEFAULT_CAPTION_TOKEN}{full_img_bbox}{DEFAULT_EOP_TOKEN}'})
                    combined_dict['conversations'].append({'from': 'gpt', 'value': caption_ann['caption']})
                    
                    # Initialize a list to hold category IDs for all existing captions
                    category_ids = []

                    # Check for category names in the caption and create bbox captions
                    caption_lower = caption_ann['caption'].lower()
                    for detection_ann in detection_anns:
                        category_name = coco_detections.loadCats(detection_ann['category_id'])[0]['name']
                        if category_name.lower() in caption_lower:
                            combined_dict['conversations'].append({'from': 'human', 'value': f'{DEFAULT_DETECT_TOKEN}{category_name}{DEFAULT_EOP_TOKEN}'})
                            combined_dict['conversations'].append({'from': 'gpt', 'value': f'({detection_ann["bbox"][0]},{detection_ann["bbox"][1]},{detection_ann["bbox"][2]},{detection_ann["bbox"][3]})'})
                            category_ids.append(detection_ann['category_id'])  
                    combined_dict['category_ids'] = category_ids
                    combined_list.append(combined_dict)
        return combined_list
    
    def process(self):
        tasks = [(year, split) for year in self.years for split in self.splits]
        
        # Use joblib for parallel processing & tqdm for progress bar
        Parallel(n_jobs=-1)(delayed(self.process_data_split)(year, split) for year, split in tqdm(tasks, desc="Processing COCO Caption datasets"))

if __name__ == "__main__":
    # Load config path and create COCOCaptionDataset instance
    data_config_path = "/data/cad-recruit-02_814/swbaek/config/config_datasets.yaml"      
    cfg = get_config(data_config_path)
    coco_dataset = COCOCaptionDataset(cfg)

    # Process datasets
    coco_dataset.process()