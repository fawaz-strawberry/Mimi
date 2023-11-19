import torch
import torch.nn as nn
from .AbstractEmbedding import AbstractEmbedding

class SemanticEmbedding(AbstractEmbedding, nn.Module):
    def __init__ (self, vocab_size, embed_size):
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
    
    def forward(self, x):
        return self.embedding(x)