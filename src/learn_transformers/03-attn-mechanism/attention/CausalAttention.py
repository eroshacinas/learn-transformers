import torch
import torch.nn as nn

"""
The use of register_buffer in
PyTorch is not strictly necessary for all use cases but offers several advantages here. For
instance, when we use the CausalAttention class in our LLM, buffers are automati-
cally moved to the appropriate device (CPU or GPU) along with our model, which will
be relevant when training our LLM. This means we don’t need to manually ensure
these tensors are on the same device as your model parameters, avoiding device mis-
match errors.
"""

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
                diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2) # transpose: keep the batch dim at first position

        # torch ops with a trailing underscore are performed in-place; this avoids unnecessary memory copies
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values

        return context_vec
