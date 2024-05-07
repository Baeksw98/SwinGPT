import os
import torch
import torch.nn as nn

from transformers import ViTImageProcessor, SwinConfig
from .custom_swin import CustomSwinModel

class SwinVisionTower(nn.Module):
    """ A neural network module encapsulating a Swin Transformer as a vision tower. """
    
    def __init__(self, vision_tower, args, delay_load=False):
        """ Initializes the Swin Vision Tower. """
        super().__init__()
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.is_loaded = False

        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = SwinConfig.from_pretrained(self.vision_tower_name)


    def load_model(self, device_map=None, dtype=None):
        """ Loads the model from a pretrained state, sets up the necessary configurations for image processing. """

        # If model is loaded don't proceed and just return 
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        # Load the image processor and vision tower 
        self.image_processor = ViTImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CustomSwinModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        
        # Set the device and dtype for the vision tower
        if dtype is not None:
            self.vision_tower = self.vision_tower.to(dtype=dtype)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        """ Selects features from the specified layer outputs of the vision tower. """
        image_features = image_forward_outs.hidden_states[self.select_layer]
        return image_features[:, 1:] if self.select_feature == 'patch' else image_features

    @torch.no_grad()
    def forward(self, images):
        """ Defines the forward pass of the vision tower on the given images. """
        
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_outs = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_outs).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_forward_outs, image_features

    """ Property methods to access the configurations and states of the vision tower. """
    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


def build_vision_tower(vision_tower_cfg, **kwargs):
    """ Builder function to create a vision tower based on the provided configuration. """
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_path_exists = os.path.exists(vision_tower) or any(vision_tower.startswith(prefix) for prefix in ["openai", "microsoft"])

    if is_path_exists:
        return SwinVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
