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
