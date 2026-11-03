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
