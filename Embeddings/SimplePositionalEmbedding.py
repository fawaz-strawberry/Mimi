import torch
import torch.nn as nn
from .AbstractEmbedding import AbstractEmbedding

max_context_len = 512
# Create positional embeddings to let the model learn the value of positions
class SimplePositionalEmbedding(AbstractEmbedding, nn.Module):
    def __init__(self, embedding_size=512, max_len=512):
        super(SimplePositionalEmbedding, self).__init__()
        
        # Create positional embeddings
        # Define a tensor of context length by input embedding dim
        pe = torch.zeros(max_len, embedding_size)
        # Create an array of position values for the context len [[0], [1], [2],..., [max_len]]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Purely a scaling factor for our tensors so that the values don't get blown up to a bajillion
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    # Add the positional embeddings to the input embeddings assuming the input embeddings are of shape (context_len, embedding_size)
    def forward(self, x):
        # Add positional embeddings to the input embeddings
        x = x + self.pe[:x.size(0), :]
        return x
