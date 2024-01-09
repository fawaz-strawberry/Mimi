import torch
import torch.nn as nn
from .AbstractEmbedding import AbstractEmbedding

class SimpleSemanticEmbedding(AbstractEmbedding, nn.Module):
    def __init__ (self, vocab_size, embed_size, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size).to(device)
    
    def forward(self, x):
        return self.embedding(x)