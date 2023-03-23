
import os
from glob import glob

import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset


class OpenFWIDataset(Dataset):
    """
    PyTorch dataset for loading OpenFWI data.

    Args:
        amp_path (str): Path of folder storing .npy files containing amplitude data.
        vel_path (list): Path of folder storing .npy files containing velocity data.
        output_size (int): The output size of the data (default: 256).

    Attributes:
        amp_data (torch.Tensor): Concatenated amplitude data as a PyTorch tensor.
        vel_data (torch.Tensor): Concatenated velocity data as a PyTorch tensor.
    """

    def __init__(
        self, 
        amp_path: str, 
        vel_path: str, 
        output_size: int = 256
    ):
        
        # list all .npy files
        if os.path.isdir(amp_path):
            amp_files = glob(os.path.join(amp_path, "*.npy"))
        elif os.path.isfile(amp_path):
            with open(amp_path, 'r') as f:
                amp_files = [line.rstrip('\n') for line in f.readlines()]
        if os.path.isdir(vel_path):
            vel_files = glob(os.path.join(vel_path, "*.npy"))
        elif os.path.isfile(vel_path):
            with open(vel_path, 'r') as f:
                vel_files = [line.rstrip('\n') for line in f.readlines()]

        # read .npy files and concatenate along first dimension
        self.amp_data = []
        for file in amp_files:
            data = torch.from_numpy(np.load(file))
            self.amp_data.append(data)
        self.amp_data = torch.cat(self.amp_data, dim=0)

        self.vel_data = []
        for file in vel_files:
            data = torch.from_numpy(np.load(file))
            self.vel_data.append(data)
        self.vel_data = torch.cat(self.vel_data, dim=0)
        
        # transform to resize output tensors
        self.output_size = output_size
        self.transform = transforms.Resize(
            (output_size, output_size), 
            interpolation=InterpolationMode.BICUBIC,
        )
        
    def __len__(self):
        """
        Returns:
            int: The total number of data points in the dataset.
        """
        return self.amp_data.shape[0]
    
    def __getitem__(self, idx):
        """
        Returns the amplitude and velocity data at a given index.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            tuple: A tuple containing the amplitude and velocity data as PyTorch tensors.
        """
        # get amplitude and velocity data at given index
        amp = self.amp_data[idx]
        vel = self.vel_data[idx]
        
        # apply transformation to resize output tensors
        amp = self.transform(amp)
        vel = self.transform(vel)
        
        # return amplitude and velocity tensors
        return vel, amp



class DummyFWIDataset(Dataset):
    """
    Dummy dataset for sanity check.

    Args:
        output_size (int): The output size of the data (default: 256).
        num_samples (int): The number of samples to generate (default: 1000).

    Attributes:
        output_size (int): The output size of the data.
    """

    def __init__(self, output_size: int = 256, num_samples: int = 1000, **kwargs):
        self.output_size = output_size
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        amp_data = torch.ones(5, self.output_size, self.output_size)
        vel_data = torch.zeros(1, self.output_size, self.output_size)
        return vel_data, amp_data


