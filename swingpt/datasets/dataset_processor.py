import copy
from dataclasses import dataclass, field
import json
from typing import Dict, Optional, Sequence, List
from PIL import UnidentifiedImageError

import transformers
import torch
from torch.utils.data import Dataset
from PIL import Image
import yaml 

from swingpt.utils import conversation as conversation_lib
from swingpt.utils.mm_utils import *
from swingpt.utils.constants import *
from swingpt.utils.box_utils import * 
from swingpt.utils.data_utils import * 
from config.arguments import *
import re 

def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    """Preprocess multimodal sources by cleaning image tokens."""
    if not data_args.is_multimodal:
        return sources
    
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].strip()
    return sources

def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """Preprocess sources to create multimodal input dictionaries for model training."""
    if not has_image:
        raise ValueError("Image is required for multimodal processing")
    
    # Add signals for the speakers: '### ' at the beginning each sentence, with end signal '\n'
    header = f"{conversation_lib.default_conversation.system}\n"
    conversations = []
    for source in sources:
        conversation = add_speaker_and_signal(header, source)
        # Concatenate conversations together
        conversations.append(conversation)

    # Tokenize the conversations to create input_ids, bbox_inputs, bbox_labels
    results = [tokenizer_image_bbox_token(prompt=prompt, tokenizer=tokenizer, return_tensors='pt') for prompt in conversations]
    input_ids, bbox_inputs, bbox_labels = results[0]['input_ids'], results[0]['bbox_inputs'], results[0]['bbox_labels']

    # Deep copy for unmasked_labels 
    unmasked_labels = copy.deepcopy(input_ids)

    # Get masked labels 
    input_ids, labels = mask_non_gpt_content(tokenizer, unmasked_labels)

    return dict(input_ids=input_ids, labels=labels, bbox_inputs=bbox_inputs, bbox_labels=bbox_labels)
    
class LazyDataset(Dataset):
    """Dataset class for fine-tuning SwinGPT Model on multimodal data."""
    def __init__(self, json_file_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazyDataset, self).__init__()
        self.list_data_dict = json.load(open(json_file_path, "r"))
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """Retrieve an item by index for training, with support for multimodal inputs."""
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Not supposed to be wrapped to a list" 

        try:
            if 'image' in sources[0]:
                # Initializing image and its processor
                image_path = self.list_data_dict[i]['image']
                processor = self.data_args.image_processor

                # Opening image and preprocessing them 
                image = Image.open(image_path).convert('RGB')
                image_width, image_height = image.width, image.height
                if self.data_args.image_aspect_ratio == 'pad':
                    image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                else:
                    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                
                # Preprocessing sources for multimodal inputs
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            else:
                # Preprocessing sources for only text inputs 
                sources = copy.deepcopy([e["conversations"] for e in sources])
        except (FileNotFoundError, UnidentifiedImageError):
            # Handle cases where the image file is not found or is corrupt
            print(f"Warning: Image at {image_path} not found or corrupt. Skipping this sample.")
        
        # Preprocess for data dictionary function
        data_dict = preprocess(sources, self.tokenizer, has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"], 
                            labels=data_dict["labels"],
                            bbox_inputs=data_dict["bbox_inputs"],
                            bbox_labels=data_dict["bbox_labels"])

        # Preprocess bounding boxes (xych -> xyxy), then normalizing bboxes
        data_dict['bbox_inputs'] = preprocess_bbox(data_dict['bbox_inputs'], image_width, image_height, normalize=True)
        data_dict['bbox_labels'] = preprocess_bbox(data_dict['bbox_labels'], image_width, image_height, normalize=True)

        # Image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # Crop images to right size
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        # Category ID exists in the data 
        if 'category_ids' in self.list_data_dict[i]:
            data_dict['category_ids'] = torch.tensor(self.list_data_dict[i]['category_ids'])
                        
        return data_dict
    
@dataclass
class DataCollator(object):
    """ Data collator for data loader generation for fine-tuning of SwinGPT """ 
    
    tokenizer: transformers.PreTrainedTokenizer
    is_eval_mode: bool  
    device: Optional[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extend_tensor(self, tensor, length, pad_value):
        current_length = tensor.size(0)
        if current_length > length:
            # If the current length is greater, truncate the tensor
            tensor = tensor[:length]
        elif current_length < length:
            # If the current length is smaller, pad the tensor
            pad_size = length - current_length
            if tensor.dim() > 1:
                padding = tensor.new_full((pad_size, tensor.size(1)), pad_value)
            else:
                padding = tensor.new_full((pad_size,), pad_value)
            tensor = torch.cat([tensor, padding], dim=0)
        return tensor
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        # Extract input_ids, labels, bbox_inputs, and bbox_labels from instances
        input_ids, labels, bbox_inputs, bbox_labels = (
            tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "bbox_inputs", "bbox_labels"))
        )
        
        # Determine the maximum sequence length from input_ids and labels
        max_length = min(max(seq.size(0) for seq in input_ids), self.tokenizer.model_max_length)

        # Extend input_ids and labels to max_length
        extended_input_ids = [self.extend_tensor(seq, max_length, self.tokenizer.pad_token_id) for seq in input_ids]
        extended_labels = [self.extend_tensor(seq, max_length, IGNORE_INDEX) for seq in labels]

        # Stack extended tensors
        batch_input_ids = torch.stack(extended_input_ids)
        batch_labels = torch.stack(extended_labels)

        # Prepare the attention mask
        separator_tokens = ['\n\n'] #'###'
        separator_token_ids = {token: self.tokenizer.encode(token, add_special_tokens=False)[0] for token in separator_tokens}

        # Prepare the initial attention mask where all non-pad tokens get attention
        attention_mask = batch_input_ids.ne(self.tokenizer.pad_token_id)

        # Further modify the attention mask to ignore specific separator tokens
        for token_id in separator_token_ids.values():
            attention_mask &= batch_input_ids.ne(token_id)
            
        # Initialize batch dictionary
        batch = {
            'input_ids': batch_input_ids,
            'labels': batch_labels,
            'bbox_inputs': bbox_inputs,
            'bbox_labels': bbox_labels,
            'attention_mask': attention_mask,
        }

        # Process images if they exist in instances
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances if instance['image'] is not None]
            assert all(image.shape == images[0].shape for image in images), "All images must have the same shape to be stacked"
            batch['images'] = torch.stack(images)

        # Add category Id to batch if it exists in instances 
        if 'category_ids' in instances[0] and self.is_eval_mode:
            category_ids = [instance['category_ids'] for instance in instances if instance['category_ids'] is not None]
            batch['category_ids'] = category_ids
        
        return batch
