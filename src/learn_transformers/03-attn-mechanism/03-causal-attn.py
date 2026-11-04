"""
For many LLM tasks, you will want the self-attention mechanism to consider only the
tokens that appear prior to the current position when predicting the next token in a
sequence. Causal attention, also known as masked attention, is a specialized form of self-
attention. It restricts a model to only consider previous and current inputs in a sequence
when processing any given token when computing attention scores.

This is in contrast
to the standard self-attention mechanism, which allows access to the entire input
sequence at once.
"""

# %%
import torch
from attention.SelfAttentionv2 import SelfAttentionv2
torch.manual_seed(789)

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
    [0.55, 0.87, 0.66], # journey
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with
    [0.77, 0.25, 0.10], # one
    [0.05, 0.80, 0.55]] # step
)

x_2 = inputs[1] # query
d_in = inputs.shape[1] # input embedding size (d_in)
d_out = 2 # output embedding (d_out)

sa_v2 = SelfAttentionv2(d_in, d_out)


# %%
"""
applying a causal attention mask:
    step 1: compute attention scores
    step 2: compute attention weights through normalization
    step 3: mask attention scores with 0s above diagonal
    step 4: normalize masked attention weights
"""
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)

attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
print(attn_weights)

# %% implement masking function
context_length = attn_scores.shape[0]

# make a MxM matric where elements > array[i][i] is 0 for i \in 0 to M-1
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

# %%
masked_simple = attn_weights * mask_simple # element-wise product
print(masked_simple)

# %% step 3: renormalize attention weights to sum up to 1
"""
The third step is to renormalize the attention weights to sum up to 1 again in each
row. We can achieve this by dividing each element in each row by the sum in each row
"""

row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

"""
information leakage:
after masking and renormalization, the distribution of attention
weights is as if it was calculated only among the unmasked positions to begin with.
This ensures there’s no information leakage from future (or otherwise masked)
tokens as we intended
"""

# %% improve mask

"""
The softmax function converts its inputs into a probability distribution. When nega-
tive infinity values (-∞) are present in a row, the softmax function treats them as zero
probability
"""

mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

attn_weights = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=1)
print(attn_weights)

# %% masking additional attention weights with dropout
"""
dropout in the atten-
tion mechanism is typically applied at two specific times: after calculating the atten-
tion weights or after applying the attention weights to the value vectors

Here we will apply the dropout mask after computing the attention weights,
"""

torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5) # 50% dropout
example = torch.ones(6,6)
print(dropout(example))
"""
To compensate for the reduction in active elements,
the values of the remaining elements in the matrix are scaled up by a
factor of 1/0.5 = 2.

This scaling is crucial to maintain the overall balance of the attention weights, ensuring that the average
influence of the attention mechanism remains consistent during both training and inference.
"""

# %% applying dropout to attention weight
torch.manual_seed(123)
print(attn_weights)
print(dropout(attn_weights))

# %% simulate batched samples by duplicating inputs
batch = torch.stack([inputs, inputs], dim=0)
print(batch.shape)

# %% using our causal attn
from attention.CausalAttention import CausalAttention
torch.manual_seed(123)

context_length = batch.shape[1]
ca = CausalAttention(d_in, d_out, context_length, 0.0)
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)
