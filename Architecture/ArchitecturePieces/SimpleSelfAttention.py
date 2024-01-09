import torch
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, device):
        super(SelfAttention, self).__init__()

        '''
        embed_size: dimension of the input, obtained from the embedding layer
        heads: number of heads to split the embed_size into, where a head looks at a part of the total embed_size
        head_dim: dimension of each head after splitting the embed_size into heads
        '''
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * self.heads == embed_size), "Embed size needs to be divisible by heads"

        self.query = nn.Linear(self.embed_size, self.embed_size, bias=False).to(device)
        self.key = nn.Linear(self.embed_size, self.embed_size, bias=False).to(device)
        self.value = nn.Linear(self.embed_size, self.embed_size, bias=False).to(device)
        self.fc_out = nn.Linear(self.head_dim * heads, embed_size).to(device)

    '''
    query: input to the query linear layer
    key: input to the key linear layer
    value: input to the value linear layer
    mask: mask to mask out the padded tokens
    '''
    def forward(self, x, mask):

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)


        # Get the batch size
        batch_size = query.shape[0]


        # Split the embed_size into heads
        # Original shape: (batch_size, seq_length, embed_size)
        # New shape: (batch_size, seq_length, heads, head_dim)
        query = query.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.heads, self.head_dim).transpose(1, 2)


        key = key.transpose(-1, -2)

        # print(query.shape)
        # print(key.shape)

        # Matrix multiplication of query and key
        # Original shape of query: (batch_size, seq_length, heads, head_dim)
        # Original shape of key: (batch_size, seq_length, heads, head_dim)
        # New shape: (batch_size, heads, seq_length, seq_length)
        attention_scores = torch.matmul(query, key) / math.sqrt(self.head_dim)

        # Mask out the padded tokens
        if mask is not None:
            # Expand the dimensions of the mask tensor to [batch_size, context_len, heads, heads_dim]
            # Apply the mask across the heads
            attention_scores = attention_scores.masked_fill(mask == 1, float("-inf"))

        # Softmax the attention scores
        attention = torch.softmax(attention_scores, dim=-1)
        # Multiply the attention scores with the value
        x = torch.matmul(attention, value)

        # Reshape the x
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.heads * self.head_dim)


        # Pass the x through the fc_out layer
        x = self.fc_out(x)

        return x
        

