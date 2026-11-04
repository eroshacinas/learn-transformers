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

# %% step 1: calculate attn score w_22
"""
The attention score computation is a dot-product computation similar to what we used in the
simplified self-attention mechanism in section 3.3. The new aspect here is that we are not directly computing the
dot-product between the input elements but using the query and key obtained by transforming the inputs via the
respective weight matrices.
"""
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

# %% step 2: attention weights; normalize
"""
The reason for the normalization by the embedding dimension size is to improve the
training performance by avoiding small gradients. For instance, when scaling up the
embedding dimension, which is typically greater than 1,000 for GPT-like LLMs, large
dot products can result in very small gradients during backpropagation due to the
softmax function applied to them.

As dot products increase, the softmax function
behaves more like a step function, resulting in gradients nearing zero. These small
gradients can drastically slow down learning or cause training to stagnate.

The scaling by the square root of the embedding dimension is the reason why this
self-attention mechanism is also called scaled-dot product attention
"""
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1) # we also scale it by sqrt(d_out)
print(attn_weights_2)
# %% step 3: compute context vector
"""
the attention weights serve as a weighting factor that
weighs the respective importance of each value vector
"""

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)
# %%
"""
why query, key, value:
    The terms “key,” “query,” and “value” in the context of attention mechanisms are
    borrowed from the domain of information retrieval and databases, where similar con-
    cepts are used to store, search, and retrieve information
"""

from attention.SelfAttentionv1 import SelfAttentionv1
torch.manual_seed(123)
sa_v1 = SelfAttentionv1(d_in, d_out)
print(sa_v1(inputs))
