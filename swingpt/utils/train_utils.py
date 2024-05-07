import os
import glob
import yaml
from pathlib import Path
import numpy as np

import torch.distributed as dist
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from dataclasses import dataclass

from config.arguments import *
from swingpt.utils import conversation as conversation_lib
from swingpt.datasets.dataset_processor import *

def load_configuration(config_path):
    """ Loads configuration settings from a YAML file and initializes argument classes. """
    # Ensure the config_path is a Path object for consistency
    config_path = Path(config_path)
    
    # Use the path to open and read the YAML configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Create instances of your argument classes using the configuration dictionary
    model_args = ModelArguments(**config['model_args'])
    data_args = DataArguments(**config['data_args'])
    training_args = TrainingArguments(**config['training_args'])

    return model_args, data_args, training_args

def setup_dist():
    """ Sets up the distributed environment for training with specific environment variables. """
    try:
        rank = int(os.environ.get('RANK', '0'))
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '56591')
        dist_url = f"tcp://{master_addr}:{master_port}"

        dist.init_process_group(
            backend='nccl',
            init_method=dist_url,
            rank=rank,
            world_size=world_size
        )
        print(f"Distributed environment ready: Rank {dist.get_rank()} of {dist.get_world_size()}")
    
    except Exception as e:
        print(f"Failed to initialize the distributed environment: {str(e)}")
        raise

def get_latest_checkpoint(checkpoint_dir):
    """ Finds the latest checkpoint file in the specified directory. """
    # List all checkpoint files
    list_of_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))  

    # If no files were found, return none
    if not list_of_files:  
        return None
    
    # Get the most recent file
    return max(list_of_files, key=os.path.getctime)  

def get_parameter_names(model, parameter_groups):
    """ Retrieves names of parameters that are part of specified groups within the model. """
    # Loop through modules and identify parameter groups
    parameters = []
    for name, module in model.named_modules():
        if any([pg in name for pg in parameter_groups]):
            parameters.extend([f"{name}.{p_name}" for p_name, p in module.named_parameters()])
    return parameters

@dataclass
class CustomCausalLMOutputWithPast(CausalLMOutputWithPast):
    """ Extended output class for causal language models to include lm_loss and cycle_loss. """
    lm_loss: Optional[torch.FloatTensor] = None
    cycle_loss: Optional[torch.FloatTensor] = None