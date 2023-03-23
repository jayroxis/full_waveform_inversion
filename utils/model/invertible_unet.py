
import torch
import torch.nn as nn

from timm.models.registry import register_model
from iunets import iUNet
from .block import *


@register_model
def invertible_unet(
    amp_chans: int, 
    vel_chans: int, 
    hidden_chans: int = 256,
    num_layers: int = 3,
    num_coupling_blk: int = 4,
    conv_hidden_chans: int = 64,
    **kwargs,
):
    # the encoders
    amp_encoder = DownsamplingBlock(
        in_chans=amp_chans, 
        out_chans=hidden_chans, 
        hidden_chans=conv_hidden_chans
    )
    vel_encoder = DownsamplingBlock(
        in_chans=vel_chans, 
        out_chans=hidden_chans, 
        hidden_chans=conv_hidden_chans
    )

    # the decoders
    amp_decoder = UpsamplingBlock(
        in_chans=hidden_chans, 
        out_chans=amp_chans, 
        hidden_chans=conv_hidden_chans
    )
    vel_decoder = UpsamplingBlock(
        in_chans=hidden_chans, 
        out_chans=vel_chans, 
        hidden_chans=conv_hidden_chans
    )

    # the iUNet backbone
    arch = tuple([num_coupling_blk for _ in range(num_layers)])
    iunet_model = iUNet(
        in_channels=hidden_chans,
        dim=2,
        architecture=arch
    )
    model = IUnetModel(
        amp_encoder=amp_encoder, 
        vel_encoder=vel_encoder, 
        amp_decoder=amp_decoder, 
        vel_decoder=vel_decoder, 
        iunet_model=iunet_model,
    )
    return model



class IUnetModel(nn.Module):
    def __init__(
        self, 
        amp_encoder, 
        vel_encoder, 
        amp_decoder, 
        vel_decoder, 
        iunet_model
    ):
        super(IUnetModel, self).__init__()
        self.amp_encoder = amp_encoder
        self.vel_encoder = vel_encoder
        self.amp_decoder = amp_decoder
        self.vel_decoder = vel_decoder
        self.iunet_model = iunet_model
        
    def amp_to_vel(self, x):
        x = self.amp_encoder(x)
        x = self.iunet_model(x)
        x = self.vel_decoder(x)
        return x
    
    def vel_to_amp(self, x):
        x = self.vel_encoder(x)
        x = self.iunet_model.inverse(x)
        x = self.amp_decoder(x)
        return x 
    
