import torch
from torch.utils.data import Dataset, DataLoader
from .AbstractDatasetLoader import AbstractDatasetLoader

class SimpleTextDataset(AbstractDatasetLoader, Dataset):
    def __init__(self, filename, is_multi_line=True, chunk_size=1024):
        self.chunks = []
        with open(filename, 'r', encoding="utf-8") as f:
            if is_multi_line:
                self.chunks = f.readlines()
            else:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break  # eof
                    self.chunks.append(chunk)

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        return self.chunks[idx]