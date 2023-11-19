import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class AbstractArchitecture(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x