
import torch
from timm.models.registry import register_model

@register_model
def unet(in_channels=3, out_channels=1, init_features=32, **kwargs):
    # Use UNet from BrainMRI from torch.hub
    model = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features
    )
    return model