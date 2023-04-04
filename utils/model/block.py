
import torch
import torch.nn as nn


class BottleneckBlock(nn.Module):
    def __init__(
        self, 
        in_chans, 
        out_chans, 
        hidden_chans, 
        output_size: int = 256
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, hidden_chans, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(hidden_chans, hidden_chans, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(hidden_chans, out_chans, kernel_size=5, padding=2)
        self.act = nn.GELU()
        self.norm1 = nn.BatchNorm2d(in_chans)
        self.norm2 = nn.BatchNorm2d(hidden_chans)
        self.norm3 = nn.BatchNorm2d(hidden_chans)
        self.resize = nn.Upsample(
            size=(output_size, output_size), 
            mode='bicubic'
        )

    def forward(self, x):
        out = self.norm1(x)
        out = self.resize(out)
        out = self.act(self.conv1(out))
        out = self.norm2(out)
        out = self.act(self.conv2(out))
        out = self.norm3(out)
        out = self.act(self.conv3(out))
        return out


class DownsamplingBlock(nn.Module):
    def __init__(
        self, 
        in_chans, 
        out_chans, 
        hidden_chans, 
        output_size: int = 256
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, hidden_chans, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(4 * hidden_chans, hidden_chans, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(4 * hidden_chans, out_chans, kernel_size=5, padding=2)
        self.act = nn.GELU()
        self.norm1 = nn.BatchNorm2d(in_chans)
        self.norm2 = nn.BatchNorm2d(hidden_chans)
        self.norm3 = nn.BatchNorm2d(hidden_chans)
        self.downsample = nn.PixelUnshuffle(2)
        self.resize = nn.Upsample(
            size=(output_size, output_size), 
            mode='bicubic'
        )

    def forward(self, x):
        out = self.norm1(x)
        out = self.resize(out)
        out = self.act(self.conv1(out))
        out = self.norm2(out)
        out = self.downsample(out)
        out = self.act(self.conv2(out))
        out = self.norm3(out)
        out = self.downsample(out)
        out = self.act(self.conv3(out))
        return out



class UpsamplingBlock(nn.Module):
    def __init__(
        self, 
        in_chans, 
        out_chans, 
        hidden_chans, 
        output_size: int = 256
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, 4 * hidden_chans, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(hidden_chans, 2 * hidden_chans, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(2 * hidden_chans, out_chans, kernel_size=5, padding=2)
        self.act = nn.GELU()
        self.norm1 = nn.BatchNorm2d(in_chans)
        self.norm2 = nn.BatchNorm2d(4 * hidden_chans)
        self.norm3 = nn.BatchNorm2d(2 * hidden_chans)
        self.upsample = nn.PixelShuffle(2)
        self.resize = nn.Upsample(
            size=(output_size, output_size), 
            mode='bicubic'
        )

    def forward(self, x):
        out = self.norm1(x)
        out = self.act(self.conv1(out))
        out = self.norm2(out)
        out = self.upsample(out)
        out = self.act(self.conv2(out))
        out = self.norm3(out)
        out = self.resize(out)
        out = self.conv3(out)
        return out