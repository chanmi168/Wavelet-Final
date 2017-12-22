
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

class WaveletDataset(Dataset):
    
    def __init__(self, npy_coeff_file, label_file):
        
        self.coeff_data = np.load(npy_coeff_file)
        self.label_data = np.load(label_file)
        self.coeff_data = self.coeff_data[:, :, 8:16]
    def __len__(self):
        return len(self.coeff_data)

    def __getitem__(self, idx):
        coeff = self.coeff_data[idx]
        label = self.label_data[idx]

        sample = (coeff, label)
        
        return sample