import os

import torch
from PIL import Image
import matplotlib.pyplot as plt

from swingpt.utils.train_utils import *
from swingpt.utils.mm_utils import * 
from swingpt.utils.data_utils import * 
from swingpt.utils.inference_utils import * 
from swingpt.evaluation.swingpt_evaluator import * 
from swingpt.model.model_loader import *

def inference(config_dir):
    # Load arguments from config file
    model_args, data_args, training_args = load_configuration(config_dir)

    # Instantiate model and tokenizer 
    model, tokenizer = load_model_and_tokenizer(model_args=model_args, training_args=training_args, data_args=data_args)
    
    # Define device
    device_map = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths to the JSON files
    json_path1 = "/data/cad-recruit-02_814/swbaek/data_processed/coco_val2017_combined.json"
    json_path2 = "/data/cad-recruit-02_814/swbaek/data_processed/refcoco_val_detect.json"
    
    # Extract image paths
    samples_IC = extract_random_samples(json_path1, num_samples=5)
    samples_OD = extract_random_samples(json_path2, num_samples=5)

    # Define the default conversation based on model arg version
    conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]

    # Instantiate the Evaluator class
    SG_IC_Evaluator = SwinGPTEvaluator(
        model=model, tokenizer=tokenizer, training_args=training_args, mode="IC"
    )
    SG_OD_Evaluator = SwinGPTEvaluator(
        model=model, tokenizer=tokenizer, training_args=training_args, mode="OD"
    )
    
    for sample in samples_IC:
        # Initialize pred_captions, pred_bboxes, extracted_tokens
        pred_captions, pred_bboxes, extracted_tokens = None, None, None 
        
        # Get batch by processing a sample
        batch = preprocess_sample(sample, model, tokenizer)
                
        # Process batch for object detection
        pred_captions = SG_IC_Evaluator.predict_step(batch, batch_idx=0)
        
        # Bounding box localization if trigger indices are present 
        if len(batch['trigger_indices']) > 0 :
            # Get text tokens to process for bbox detection 
            extracted_tokens = extract_tokens(batch, tokenizer)
            
            # Ensure batch data is on the same device as the model
            batch = {k: move_to_device(v, device_map) for k, v in batch.items()}
            
            # Run generate bbox functions to get pred_bboxes
            pred_bboxes = model.generate_bbox(**batch)

        if pred_captions is not None:
            pred_captions = ' '.join(pred_captions)
            
        # Print image with its annotations 
        annotated_images(sample['image'], pred_captions, pred_bboxes, extracted_tokens)           
        
    # Object Detection Demonstrations
    for sample in samples_OD:
        # Get batch by processing a sample
        batch = preprocess_sample(sample, model, tokenizer)
        
        # Process batch for object detection
        pred_bboxes = SG_OD_Evaluator.predict_step(batch, batch_idx=0)
        
        # Get text tokens to process for bbox detection 
        extracted_tokens = extract_tokens(batch, tokenizer)
        
        # Print image with its annotations 
        annotated_images(sample['image'], None, pred_bboxes, extracted_tokens)     
        
            
if __name__ == "__main__":
    # Load configuration directory
    config_dir = "/data/cad-recruit-02_814/swbaek/config/config_arguments_test.yaml"

    inference(config_dir)
    