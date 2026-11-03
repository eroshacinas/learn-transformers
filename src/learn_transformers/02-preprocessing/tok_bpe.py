# %%
from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))
# %% instantiate
tokenizer = tiktoken.get_encoding("gpt2")
# %%
text = (
"Hello, do you like tea? <|endoftext|> In the sunlit terraces"
"of someunknownPlace."
)
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
# %% decode
strings = tokenizer.decode(integers)
print(strings)

# %% test another word
text = "Akwirw ier"

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)
strings = tokenizer.decode(integers)
print(strings)
for i in integers:
    print(tokenizer.decode([i]))

# %% read dataset
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# tokenize
enc_text = tokenizer.encode(raw_text)
print(len(enc_text))

enc_sample = enc_text[50:]

# create input-target pairs for next word prediction
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1] # shift the target
print(f"x: {x}")
print(f"y: {y}")

# create next-word pred task
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)

# %% repeat prev code but convert to token IDs to text
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))


# %% data loader v1
from dataloader.GPTDatasetV1 import create_dataloader_v1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4,
    stride=1, shuffle=False)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)
# %% creating token embeddings
# for example: if we have a vocab size of 6, and output dim of 3
# our embedding table is 6x3
import torch
torch.manual_seed(123)

vocab_size = 6
output_dim = 3

embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

# %% apply embedding table to a token ID
print(embedding_layer(torch.tensor(3)))

# %% encoding word positions
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
raw_text, batch_size=8, max_length=max_length,
stride=max_length, shuffle=False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

# %%
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

# %% finally, inject pos embedding
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(f"pos embedding: {pos_embeddings.shape}")

input_embeddings = token_embeddings + pos_embeddings
print(f"input embeddings: {input_embeddings.shape}")
