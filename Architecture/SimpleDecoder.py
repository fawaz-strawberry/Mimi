import torch
import torch.nn as nn
from .AbstractArchitecture import AbstractArchitecture
from .ArchitecturePieces.SimpleBlock import SimpleBlock



class SimpleDecoder(AbstractArchitecture, nn.Module):
    def __init__(self, embed_size, vocab_size, heads, dropout):
        super().__init__()
        self.blocks = nn.Sequential(*[SimpleBlock(embed_size=embed_size, heads=heads, dropout=dropout) for _ in range(6)])
        self.ln_out = nn.LayerNorm(embed_size)
        self.fc_out = nn.Linear(embed_size, vocab_size)


    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_out(x)
        x = self.fc_out(x)
        return x