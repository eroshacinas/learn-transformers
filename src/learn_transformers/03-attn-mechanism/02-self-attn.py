"""
implement self attention with trainable weights. use scaled dot-product attention
as in the original paper to do so.

three trainable weight matrices Wq, Wk, and Wv. These three matrices are used to
project the embedded input tokens, x(i), into query, key, and value vectors, respec-
tively
"""

# %%
import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
    [0.55, 0.87, 0.66], # journey
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with
    [0.77, 0.25, 0.10], # one
    [0.05, 0.80, 0.55]] # step
)

# %%
"""
The query vector q(2) is obtained via matrix multiplication between the input x(2) and
the weight matrix Wq. Similarly, we obtain the key and value vectors via matrix multiplication
involving the weight matrices Wk and Wv

weight params vs attention weights
weight parameters are the fundamental, learned coefficients that define
the network’s connections, while attention weights are dynamic, context-specific values

in other words, the K,Q,V weights are from linear layers (or linear equation) that
gets adjusted during training. its respective
projections (x@W_k, x@W_q, and x@W_v) are used to calculate attention
"""

x_2 = inputs[1] # query
d_in = inputs.shape[1] # input embedding size (d_in)
d_out = 2 # output embedding (d_out)

# %%
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# %%
query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(query_2)

# %% keys and values
keys = inputs @ W_key
values = inputs @ W_value

# we see that we projected the six input tokens from a 3-d to 2-d embedding space
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
