import torch
import torch.nn as nn

from SimpleSelfAttention import SelfAttention

class SimpleBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(SimpleBlock, self).__init__()


        self.attention = SelfAttention(embed_size, heads)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

        self.mlp = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):

        attention = self.attention(x, mask)
        x = self.ln1(attention + x)
        x = self.dropout(x)

        mlp = self.mlp(x)
        x = self.ln2(mlp + x)
        x = self.dropout(x)

        return x