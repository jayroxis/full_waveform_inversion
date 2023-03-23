
import torch
import torch.nn as nn
import functools
from timm.models.registry import register_model



@register_model
def patch_gan_disc(
    input_nc=3, 
    ndf=64, 
    n_layers=3, 
    norm_layer=nn.BatchNorm2d, 
    **kwargs
):
    """Defines a PatchGAN discriminator"""
    model = NLayerDiscriminator(
        input_nc=input_nc, 
        ndf=ndf, 
        n_layers=n_layers, 
        norm_layer=norm_layer, 
        **kwargs
    )
    return model



class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, **kwargs):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        # no need to use bias as BatchNorm2d has affine parameters
        if type(norm_layer) == functools.partial:  
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(
                input_nc, ndf, 
                kernel_size=kw, 
                stride=2, 
                padding=padw
            ), 
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev, 
                    ndf * nf_mult, 
                    kernel_size=kw, 
                    stride=2, 
                    padding=padw, 
                    bias=use_bias
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev, 
                ndf * nf_mult, 
                kernel_size=kw, 
                stride=1, 
                padding=padw, 
                bias=use_bias
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(
            ndf * nf_mult, 
            1, 
            kernel_size=kw, 
            stride=1, 
            padding=padw
        )] 
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

