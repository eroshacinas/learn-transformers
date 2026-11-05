# %%
import tiktoken
import torch
from gpt_model import DummyGPTModel

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
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)

# the output is 2 samples x 4 tokens x vocab size
print(logits)
