import copy
import re
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from swingpt.utils.constants import *
from swingpt.utils.box_utils import *
from swingpt.utils.data_utils import *
from swingpt.utils import conversation as conversation_lib


def insert_tokens(prompt_chunks, tokens, offset):
    """ Inserts tokens into the prompt at specified intervals. """
    input_ids = []
    for i, chunk in enumerate(prompt_chunks):
        input_ids.extend(chunk)
        if i < len(tokens):
            input_ids.extend([tokens[i]] * (offset + 1))
    return input_ids

def tokenizer_image_bbox_token(prompt, tokenizer, return_tensors=None):
    """ Tokenizes text while inserting special tokens for images and bounding boxes. """
    
    # Instantiate the bounding box pattern
    bbox_patterns = r'\(\d+(?:\.\d+)?,\d+(?:\.\d+)?,\d+(?:\.\d+)?,\d+(?:\.\d+)?\)'
    
    # Initialize image and bbox token indices
    bbox_token_index = tokenizer.bbox_token_id
    image_token_index = tokenizer.image_token_id
    trigger_token_index = tokenizer.trigger_token_id 
    separator_tokens = tokenizer.encode('\n###', add_special_tokens=False) 

    # Split and tokenize the prompt by <image> and bbox patterns
    image_splits = prompt.split('<image>')
    input_ids = []
    first = True
    for split in image_splits:
        # Tokenize each split part and then split by bbox pattern to interleave bbox tokens
        text_parts = re.split(bbox_patterns, split)
        tokenized_parts = [tokenizer(part).input_ids for part in text_parts]
        bbox_tokens = [bbox_token_index] * (len(tokenized_parts) - 1)
        
        # Insert image token index before each split, except the first
        if not first:
            input_ids.append(image_token_index)  
        input_ids.extend(insert_tokens(tokenized_parts, bbox_tokens, 0))
        first = False

    # Apply the filter to remove unwanted <bbox> tokens (In between <trigger> and "\n###")
    input_ids = filter_bbox_tokens(input_ids, trigger_token_index, bbox_token_index, separator_tokens)
        
    # Find all bbox coordinates in the prompt
    bbox_inputs = [[float(coord) for coord in re.split(r'[^\d\.]+', coords) if coord] for coords in re.findall(bbox_patterns, prompt)]

    # Regex pattern to capture bbox coordinates only within the 'gpt:' to '\n' segment
    bbox_target_pattern = r'gpt:([^\\n]*?)\n'
    bbox_labels = []
    for target in re.findall(bbox_target_pattern, prompt):
        target_bboxes = re.findall(bbox_patterns, target)
        for bbox in target_bboxes:
            bbox_labels.append([float(coord) for coord in re.split(r'[^\d\.]+', bbox) if coord])

    # Convert to tensors if requested
    if return_tensors is not None:
        if return_tensors == 'pt':
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            bbox_inputs = torch.tensor(bbox_inputs, dtype=torch.float)
            bbox_labels = torch.tensor(bbox_labels, dtype=torch.float)
            return dict(input_ids=input_ids, bbox_inputs=bbox_inputs, bbox_labels=bbox_labels)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')

    return dict(input_ids=input_ids, bbox_inputs=bbox_inputs, bbox_labels=bbox_labels)

def filter_bbox_tokens(input_ids, trigger_token_id, bbox_token_id, separator_sequence):
    """ Filters out <bbox> tokens that appear after <trigger> and before the sequence representing "\n###". """
    result = []
    in_trigger_section = False
    check_for_separator = False
    separator_index = 0

    for token in input_ids:
        if token == trigger_token_id:
            in_trigger_section = True  # Start of trigger section
            separator_index = 0  # Reset the separator sequence check
            check_for_separator = False
        elif in_trigger_section and check_for_separator:
            # Check if the current token matches the expected token in the separator sequence
            if token == separator_sequence[separator_index]:
                separator_index += 1
                # Check if the entire separator sequence has been matched
                if separator_index == len(separator_sequence):
                    in_trigger_section = False  # End of trigger section
                    check_for_separator = False
                    separator_index = 0
            else:
                # Reset if the sequence does not match
                separator_index = 0
                check_for_separator = False

        # Start checking for separator sequence when encountering the first token of the sequence
        if token == separator_sequence[0] and in_trigger_section:
            check_for_separator = True
            separator_index = 1

        # Append tokens normally outside trigger sections or if they are not <bbox> tokens
        if not (in_trigger_section and token == bbox_token_id):
            result.append(token)

    return result

def expand2square(pil_img, background_color):
    """ Expands an image to a square by padding with the specified background color. """
    
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def process_images(images, image_processor, model_cfg):
    """ Processes a list of images according to the model configuration. """
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def preprocess_bbox_inputs(bbox_inputs):
    """
    Check the first bounding box in bbox_inputs to see if it represents the whole image.
    If it matches [0, 0, 1, 1], return it; otherwise, return None.
    """
    # Define the whole image bounding box pattern
    whole_image_bbox = torch.tensor([0.0, 0.0, 1.0, 1.0])

    # Iterate over each tensor in the batch
    for bbox in bbox_inputs:
        if bbox is None:
            continue
        
        # Ensure the device and dtype match for comparison
        if bbox.device != whole_image_bbox.device:
            whole_image_bbox = whole_image_bbox.to(bbox.device)
        if bbox.dtype != whole_image_bbox.dtype:
            whole_image_bbox = whole_image_bbox.to(dtype=bbox.dtype)

        # Check if the current bounding box matches the whole image bbox
        if torch.all(torch.isclose(bbox, whole_image_bbox, atol=1e-6)):
            # Return the matching bbox if it represents the whole image
            return bbox.unsqueeze(0)  

    # If no matching bbox is found, return None
    return None