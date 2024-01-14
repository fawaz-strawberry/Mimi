import torch
import torch.nn as nn

from .SimpleSelfAttention import SelfAttention

class SimpleBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, device):
        super(SimpleBlock, self).__init__()


        self.attention = SelfAttention(embed_size, heads, device)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

        self.mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4).to(device),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size).to(device)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        
        x = self.ln1(x)
        x = x + self.attention(x, mask)
        x = self.dropout(x)
        x = self.ln2(x)
        x = x + self.mlp(x)
        x = self.dropout(x)

        return x