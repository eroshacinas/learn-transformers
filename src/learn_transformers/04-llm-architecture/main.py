# %%
import tiktoken
import torch
import torch.nn as nn
from gpt_model import GPTModel, LayerNorm, FeedForward, TransformerBlock

# refer to page 95 for what each variable means
GPT_CONFIG_124M = {
    "vocab_size": 50_257,
    "context_length": 1_024,
    "emb_dim": 768,
    "n_heads": 12, # number of attention heads
    "n_layers": 12, # number of layers
    "drop_rate": 0.1, # dropout rate
    "qkv_bias": False
}

# %%
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim=0)
print(batch)

# %% feed it to gpt
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)

# the output is 2 samples x 4 tokens x vocab size
print(logits)

# %% layer norm test
torch.manual_seed(123)
batch_example = torch.randn(2, 5)
layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
out = layer(batch_example)
print(out)

mean = out.mean(dim=-1, keepdim=True)
var = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)

# %% apply layer norm to test sample
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim=-1, keepdim=True)
var = out_norm.var(dim=-1, keepdim=True)
print("Normalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)

# %% actualy LayerNorm
ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)

mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", var)


# %% test FF layer
ffn = FeedForward(GPT_CONFIG_124M)
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)

# %% test TransformerBLock
torch.manual_seed(123)
x = torch.rand(2, 4, 768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

"""
As we can see, the transformer block maintains the input dimensions in its output, indi-
cating that the transformer architecture processes sequences of data without altering
their shape throughout the network

The preservation of shape throughout the transformer block architecture is not
incidental but a crucial aspect of its design. This design enables its effective applica-
tion across a wide range of sequence-to-sequence tasks, where each output vector
directly corresponds to an input vector, maintaining a one-to-one relationship
"""
print("Input shape:", x.shape)
print("Output shape:", output.shape)

# %% GPT model
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)

# %%
"""
weight tying
which was used in the original GPT-2
architecture.

It means that the original GPT-2 architecture reuses the weights from
the token embedding layer in its output layer

The token embedding and output layers are very large due to the number of rows for
the 50,257 in the tokenizer’s vocabulary
"""
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")

print("Token embedding layer shape:", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)

# %% remove output layer param count from GPT-2 model count
"""
As we can see, the model is now only 124 million parameters large, matching the orig-
inal size of the GPT-2 model


Weight tying reduces the overall memory footprint and computational complexity
of the model. However, in my experience, using separate token embedding and out-
put layers results in better training and model performance; hence, we use separate
layers in our GPTModel implementation

The same is true for modern LLMs
"""

total_params_gpt2 = (
total_params - sum(p.numel()
for p in model.out_head.parameters())
)
print(f"Number of trainable parameters "
f"considering weight tying: {total_params_gpt2:,}"
)


# %% compute memory requirements of 163M param model
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb:.2f} MB")

# %% autoregressive nature
"""
refer to Figure 4.17 for full description:
    1. encode text input into four token IDs
    2. return matrix containing 4 (token IDs) x 50,257 (vocab size)
    3. get last vector since we're interested in next token
    4. convert logits into prob. distrib. using softmax
    5. identify the largest value and get its token ID
    6. decode token ID and append token to current input seuquence
    7. rerun 1-6 until max length is achieved or END token is generated
"""

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    greedy-style decoding.
    idx is a (batch, n_tokens) array
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:] # get last CONTEXT_SIZE tokens

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :] # get last vector of each sample
        probas = torch.softmax(logits, dim=-1) # apply softmax col-wise
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) # get index of max score
        idx = torch.cat((idx, idx_next), dim=1) # append predicted text


    return idx

# %% test greedy decode
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

model.eval()

out = generate_text_simple(
    model=model,
    idx = encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output:", out)
print("Output length:", len(out[0]))
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
