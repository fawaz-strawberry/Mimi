import torch
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod

class AbstractDatasetLoader(ABC, Dataset):
    def __init__(self):
        super().__init__()
        self.chunks = []

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]