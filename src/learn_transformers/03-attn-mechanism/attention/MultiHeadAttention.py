"""
Up to this point, we have implemented a MultiHeadAttentionWrapper that combined
multiple single-head attention modules. However, these are processed sequentially via
[head(x) for head in self.heads] in the forward method.

We can improve this implementation by processing the heads in parallel.
One way to achieve this is by com-
puting the outputs for all attention heads simultaneously via matrix multiplication.
"""

import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout,
        num_heads, qkv_bias=False):
            super().__init__()

            assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

            self.d_out = d_out
            self.num_heads = num_heads

            self.head_dim = d_out // num_heads

            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

            self.out_proj = nn.Linear(d_out, d_out)

            self.dropout = nn.Dropout(dropout)

            self.register_buffer(
                'mask',
                torch.triu(torch.ones(context_length, context_length),
                    diagonal=1)
            )


    def forward(self, x):
        """
        The MultiHeadAttention class takes an integrated approach. It starts with a
        multi-head layer and then internally splits this layer into individual attention heads, as
        illustrated in figure 3.26

        The splitting of the query, key, and value tensors is achieved through tensor reshap-
        ing and transposing operations using PyTorch’s .view and .transpose methods
        """
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # implicitly split the matrix by adding a num_heads dim
        # then unroll the last dim: (b, num_tokens, d_out) --> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2) # b, num_tokens, n_heads, head_dim

         # combines heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        context_vec = self.out_proj(context_vec) # adds optional linear projection

        return context_vec
