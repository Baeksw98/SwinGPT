import orjson
import os
from swingpt.utils.data_utils import save_to_json, get_config
from swingpt.utils.constants import *
from tqdm import tqdm
from joblib import Parallel, delayed

class VGDataset:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data_dir = self.cfg.main.datasets.data_dir
        self.processed_dir = self.cfg.main.datasets.processed_dir
        self.vg_data = self.load_vg_data()
        
    def load_vg_data(self):
        vg_file_path = os.path.join(self.data_dir, 'VisualGenome', 'metadata', 'region_descriptions.json')
        with open(vg_file_path, 'rb') as file:  # Open as binary
            return orjson.loads(file.read())
    
    def process_detections(self):
        detections = []
        for image in tqdm(self.vg_data, desc="Processing Detections"):
            image_path = os.path.join(self.data_dir, 'VisualGenome', 'images', f"{image['id']}.jpg")
            
            # Check if the image file exists
            if os.path.exists(image_path):  
                for region in image['regions']:
                    detection = {
                        'conversations': [
                            {'from': 'human', 'value': f"{DEFAULT_DETECT_TOKEN}{region['phrase']}{DEFAULT_EOP_TOKEN}"},
                            {'from': 'gpt', 'value': f"({region['x']},{region['y']},{region['width']},{region['height']})"}
                        ],
                        'id': image['id'],
                        'image': image_path
                    }
                    detections.append(detection)
        save_to_json(detections, "vg_detect.json", self.processed_dir)
    
    def process_captions(self):
        captions = []
        for image in tqdm(self.vg_data, desc="Processing Captions"):
            image_path = os.path.join(self.data_dir, 'VisualGenome', 'images', f"{image['id']}.jpg")
            # Check if the image file exists
            if os.path.exists(image_path):  
                for region in image['regions']:
                    caption = {
                        'conversations': [
                            {'from': 'human', 'value': f"{DEFAULT_CAPTION_TOKEN}({region['x']},{region['y']},{region['width']},{region['height']}){DEFAULT_EOP_TOKEN}"},
                            {'from': 'gpt', 'value': region['phrase']}
                        ],
                        'id': image['id'],
                        'image': image_path
                    }
                    captions.append(caption)
        save_to_json(captions, "vg_caption.json", self.processed_dir)
    
    def process(self):
        tasks = [self.process_detections, self.process_captions]
        # Use joblib for parallel processing & tqdm for progress bar
        Parallel(n_jobs=-1)(delayed(task)() for task in tqdm(tasks, desc="Processing Visual Genome Datasets"))
    
    def visualize_single_instance(self):
        import json  
        vg_file_path = os.path.join(self.data_dir, 'VisualGenome', 'metadata', 'region_descriptions.json')
        with open(vg_file_path, 'rb') as file:
            # Load only the first element for visualization
            data = orjson.loads(file.read())
            first_image = data[0]  

            # Assuming there's at least one region in the first image
            if first_image['regions']:
                first_region = first_image['regions'][0]
                image_path = os.path.join(self.data_dir, 'VisualGenome', 'images', f"{image['id']}.jpg")
                
                # Prepare the detection format
                detection = {
                    'conversations': [
                        {'from': 'human', 'value': '{}{}{}'.format(DEFAULT_DETECT_TOKEN, first_region['phrase'], DEFAULT_EOP_TOKEN)},
                        {'from': 'gpt', 'value': '({},{},{},{})'.format(first_region['x'], first_region['y'], first_region['width'], first_region['height'])}
                    ],
                    'id': first_image['id'],
                    'image': image_path
                }

                # Prepare the caption format
                caption = {
                    'conversations': [
                        {'from': 'human', 'value': '{}({},{},{},{}){}'.format(DEFAULT_CAPTION_TOKEN, first_region['x'], first_region['y'], first_region['width'], first_region['height'], DEFAULT_EOP_TOKEN)},
                        {'from': 'gpt', 'value': first_region['phrase']}
                    ],
                    'id': first_image['id'],
                    'image': image_path
                }

                # Print the formatted detection and caption
                print("Detection Format:")
                print(json.dumps(detection, indent=4))
                print("\nCaption Format:")
                print(json.dumps(caption, indent=4))

if __name__ == "__main__":
    # Load config path and create COCODetectDataset instance
    data_config_path = "/data/cad-recruit-02_814/swbaek/config/config_datasets.yaml"      
    cfg = get_config(data_config_path)
    vg_dataset = VGDataset(cfg)

    # Process datasets
    vg_dataset.process()
