import os
import random
import re
import yaml
import orjson
import torch
from typing import Dict, List, Optional
import transformers

from swingpt.utils.constants import *
from swingpt.utils import conversation as conversation_lib

class CFG:
    """ A configuration class that loads settings from a YAML file or dictionary. """
    
    def __init__(self, yaml_path):
        """ Initialize the configuration by loading it from a YAML file or a dictionary. """
        if isinstance(yaml_path, str):
            with open(yaml_path, "r") as f:
                yaml_data = yaml.safe_load(f)
        else:
            yaml_data = yaml_path

        for key, value in yaml_data.items():
            if isinstance(value, dict):
                setattr(self, key, CFG(value))
            else:
                setattr(self, key, value)

def get_config(file_path):
    """ Load configuration from a YAML file. """
    return CFG(file_path)

def save_to_json(data, filename, directory):
    """ Save data to a JSON file in the specified directory. """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Create the full path for the file
    file_path = os.path.join(directory, filename)
    
    # Write the data to a JSON file
    with open(file_path, 'wb') as f:  
        f.write(orjson.dumps(data))
    print(f"Data saved to {file_path}")
    
def find_sequence_positions(input_ids, sequence):
    """ Find the start positions of a sequence within an input_ids tensor. """
    # Ensure the sequence is a tensor
    seq_tensor = torch.tensor(sequence, device=input_ids.device) if isinstance(sequence, list) else sequence

    # Get length of the sequence to match
    seq_length = seq_tensor.size(0)

    # Create a sliding window view of input_ids of width equal to sequence length
    windows = input_ids.unfold(dimension=1, size=seq_length, step=1)

    # Determine where all elements in each window match the sequence
    matches = (windows == seq_tensor[None, None, :]).all(dim=2)

    # Find the start indices of all matches
    match_indices = matches.nonzero(as_tuple=True)

    # Return the column indices (start positions of the sequence) if found, else None
    return match_indices[1] if match_indices[0].numel() > 0 else None

def extract_random_samples(json_file_path, num_samples=3):
    """ Extracts a specified number of random samples from a JSON file. """
    with open(json_file_path, 'rb') as file:
        data = orjson.loads(file.read())
    
    # Randomly sample full dictionary entries from the data
    random_samples = random.sample(data, num_samples)
    
    return random_samples

def smart_tokenizer_and_embedding_resize(special_tokens_dict: Dict, tokenizer: transformers.PreTrainedTokenizer,model: transformers.PreTrainedModel,):
    """ Add special tokens to tokenizer and resize embeddings in the model accordingly. """
    
    #Resize tokenizer and embedding.
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def mask_non_gpt_content(tokenizer, unmasked_labels):
    """ Masks all content in the input text except where the speaker is 'gpt'. """
    # Define the pattern to extract 'gpt' speaking parts
    gpt_pattern = re.compile(r"(\n###gpt:.*?)(?=\n###|$)")

    # Generate text and Replace multiple spaces with a single space
    text = tokenizer.decode(unmasked_labels, skip_special_tokens=False)

    # Find all 'gpt' segments
    gpt_segments = gpt_pattern.findall(text)

    # Tokenize the entire text
    full_input_ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False))

    # Mask all tokens initially
    masked_labels = torch.full_like(full_input_ids, fill_value=IGNORE_INDEX)

    # Process each 'gpt' segment and unmask it
    current_position = 0
    for segment in gpt_segments:
        segment_ids = torch.tensor(tokenizer.encode(segment, add_special_tokens=False))

        found = False
        # Search for the exact position of the segment in the full text
        for start_position in range(current_position, len(full_input_ids) - len(segment_ids) + 1):
            if torch.equal(full_input_ids[start_position:start_position + len(segment_ids)], segment_ids):
                end_position = start_position + len(segment_ids)
                masked_labels[start_position:end_position] = full_input_ids[start_position:end_position]
                current_position = end_position
                found = True
                break

        if not found:
            current_position += 1  # Increment if not found to continue searching
    
    return full_input_ids, masked_labels

def add_speaker_and_signal(header, source, get_conversation=True):
    """ Format sentences by adding speaker labels and signals. """
    
    #Add speaker and start/end signal on each round.
    BEGIN_SIGNAL = "###"
    END_SIGNAL = "\n"
    conversation = header
    
    # Regular expression pattern to identify bounding box coordinates
    bbox_pattern = re.compile(r'\(\d+(?:\.\d+)?,\d+(?:\.\d+)?,\d+(?:\.\d+)?,\d+(?:\.\d+)?\)')

    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
            # Remove DEFAULT_TRIGGER_TOKEN token from human speaking parts
            sentence["value"] = sentence["value"].replace(DEFAULT_TRIGGER_TOKEN, "")
        
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
            # Append DEFAULT_TRIGGER_TOKEN after bounding box coordinates for gpt speaking parts
            sentence["value"] = re.sub(bbox_pattern, lambda match: match.group(0) + " " + DEFAULT_TRIGGER_TOKEN, sentence["value"])
        
        else:
            from_str = 'unknown'
        formatted_sentence = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += formatted_sentence
    conversation += BEGIN_SIGNAL
    return conversation