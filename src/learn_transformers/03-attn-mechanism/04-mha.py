# %%
"""
The term “multi-head” refers to dividing the attention mechanism into multiple
“heads,” each operating independently. In this context, a single causal attention mod-
ule can be considered single-head attention, where there is only one set of attention
weights processing the input sequentially.
"""

import torch
from attention.MultiHeadAttentionWrapper import MultiheadAttentionWrapper

torch.manual_seed(123)

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
    [0.55, 0.87, 0.66], # journey
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with
    [0.77, 0.25, 0.10], # one
    [0.05, 0.80, 0.55]] # step
)

batch = torch.stack([inputs, inputs], dim=0)


context_length = batch.shape[1] # This is the number of tokens
d_in, d_out = 3, 2

# %% stack multiple single head attention
"""
implementing multi-head attention involves creating multiple
instances of the self-attention mechanism (see figure 3.18), each with its own weights,
and then combining their outputs
"""



mha = MultiheadAttentionWrapper(d_in, d_out, context_length, dropout=0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)
# %% using mha
from attention.MultiHeadAttention import MultiHeadAttention

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2

mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)

print("context_vecs.shape:", context_vecs.shape)
