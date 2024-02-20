import torch
import torch.nn as nn
from .AbstractEmbedding import AbstractEmbedding

max_context_len = 512
# Create positional embeddings to let the model learn the value of positions
class SimplePositionalEmbedding(AbstractEmbedding, nn.Module):
    def __init__(self, embedding_size=512, context_length=512, device='cpu'):
        super(SimplePositionalEmbedding, self).__init__()
        
        # Create positional embeddings
        # Define a tensor of context length by input embedding dim
        pe = torch.zeros(context_length, embedding_size)
        # Create an array of position values for the context len [[0], [1], [2],..., [context_length]]
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        print(position.shape)
        # Purely a scaling factor for our tensors so that the values don't get blown up to a bajillion
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_size))
        print(div_term.shape)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.device = device
        self.register_buffer('pe', pe)

    # Add the positional embeddings to the input embeddings assuming the input embeddings are of shape (context_len, embedding_size)
    def forward(self, x):
        # x is expected to have shape [batch_size, context_len, embedding_size]
        batch_size, context_len, _ = x.shape

        # Repeat self.pe along the batch size dimension
        # Adjusting shape to [batch_size, context_len, embedding_size]
        pe = self.pe.repeat(1, batch_size, 1).transpose(0, 1)
        pe = pe.to(self.device)

        # Add positional embeddings to the input embeddings
        x = x + pe[:, :context_len, :]
        return x

