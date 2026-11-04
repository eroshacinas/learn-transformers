"""
The goal of self-attention is to compute a context vector for each input
element that combines information from all other input elements

In self-attention, our goal is to calculate context vectors z(i) for each element x(i)
in the input sequence. A context vector can be interpreted as an enriched embedding
vector
"""
# %%
import torch

# %% example embedding dim
inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your
    [0.55, 0.87, 0.66], # journey
    [0.57, 0.85, 0.64], # starts
    [0.22, 0.58, 0.33], # with
    [0.77, 0.25, 0.10], # one
    [0.05, 0.80, 0.55]] # step
)

# %% compute how a query token relates to each tokens
"""
the dot product is a measure of similarity
because it quantifies how closely two vectors are aligned: a higher dot product indi-
cates a greater degree of alignment or similarity between the vectors. In the context
of self-attention mechanisms, the dot product determines the extent to which
each element in a sequence focuses on, or “attends to,” any other element: the
higher the dot product, the higher the similarity and attention score between two
elements
"""
query = inputs[1] # embedded query token
attn_scores_2 = torch.empty(inputs.shape[0])

for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product

print(attn_scores_2)

# %% normalization
"""
The main goal behind the normalization is to obtain attention
weights that sum up to 1. This normalization is a convention that is useful for interpre-
tation and maintaining training stability in an LLM
"""

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print(f"Attention weights: {attn_weights_2_tmp}")
print(f"sum: {attn_weights_2_tmp.sum():.4f}")
# %% softmax
"""
it’s more common and advisable to use the softmax function for normalization
as it offers more favorable gradient properties
and is better at managing extreme values
"""

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())
# %% in practice, its better to use torch's softmax
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())
# %% calculate context vector
"""
calculating the context vector z(2) by multiplying the
embedded input tokens, x(i), with the corresponding attention weights and then sum-
ming the resulting vectors.

Thus, context vector z(2) is the weighted sum of all input vec-
tors, obtained by multiplying each input vector by its corresponding attention weight
"""
query = inputs[1] # second input token
context_vec_2 = torch.zeros(query.shape)

for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i # scale each embedding dim by attention score (refer to Fig. 3.10)

print(context_vec_2) # weighted sum of all input vectors
# %% generalized computation: computing attention weights for all input tokens
"""
1. compute attention scores - dot products between each input's embedding dim
2. compute attention weights - normalized attention scores
3. compute context vectors - weighted sum over the inputs
"""

attn_scores = torch.empty(6,6) # based on number of tokens x number of tokens

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i,j] = torch.dot(x_i, x_j)

# each row in attn_scores represent the a token as query
print(attn_scores)

# %% IRL, we use matrix mult
attn_scores = inputs @ inputs.T
print(attn_scores)

# %% normalize (again)
attn_weights = torch.softmax(attn_scores, dim = 1)
print(attn_weights)

# verify all rows sum to 1
print(f"sum of values along y-axis: {torch.sum(attn_weights, dim=1)}")
