import torch
import torch.nn as nn
import re

class IdentityMap(nn.Module):
    """ A simple identity mapping module that returns the input as output without any transformation. """
    
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        """ Perform identity operation, returning the input as it is. """
        return x

    @property
    def config(self):
        """ Configuration for the identity mapping, specifying its type in the model. """
        return {"mm_projector_type": 'identity'}
    
def build_vision_projector(config, delay_load=False, **kwargs):
    """ Constructs a vision projector module based on the provided configuration. """
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    # Simple linear transformation
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    # Multi-layer perceptron with GELU activation
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    # Identity map that returns its input as its output
    if projector_type == 'identity':
        return IdentityMap()

    # If none of the known types match, raise an error.
    raise ValueError(f'Unknown projector type: {projector_type}')
