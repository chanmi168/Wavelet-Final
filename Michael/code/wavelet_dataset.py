
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class WaveletDataset(Dataset):
    
    def __init__(self, npy_coeff_file, label_file):
        
        self.coeff_data = np.load(npy_coeff_file)
        self.label_data = np.load(label_file)
        

    def __len__(self):
        return len(self.coeff_data)

    def __getitem__(self, idx):
    	coeff = self.coeff_data[idx]
    	label = self.label_data[idx]

        sample = (coeff, label)
        
        return sample