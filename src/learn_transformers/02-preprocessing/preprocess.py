# %%
import re

# %%
text = "Hello, world. This, is a test."
result = re.split(r'(\s)', text)
print(result)
# %%
result = re.split(r'([,.]|\s)', text)
print(result)

# %%
result = [item for item in result if item.strip()]
print(result)

# %%
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()]|--|\s+)', text)
result = [item for item in result if item and not item.isspace()]
print(result)
# %% open dataset
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])

# %% apply preprocess
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
# %%
print(preprocessed[:30])
# %% tokenize
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

# %% create vocabulary
vocab = {token:integer for integer,token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break

# %% import tokenizer and instantiate
from tokenizer.SimpleTokenizerV1 import SimpleTokenizerV1

tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
# %% test decode
print(tokenizer.decode(ids))

# %% test on sample not in training set
# this will throw a KeyError 'Hello'
text = "Hello, do you like tea?"
print(tokenizer.encode(text))

# %% extend vocab: include <|endoftext|> and <|unk|>
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))


# %% test again
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)

# %% simple tokenizer v2
from tokenizer.SimpleTokenizerV2 import SimpleTokenizerV2

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))
print(text)

# %%
tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))

# %%
print(tokenizer.decode(tokenizer.encode(text)))
"""
- [BOS] (beginning of sequence)—This token marks the start of a text. It signifies to
the LLM where a piece of content begins.

- [EOS] (end of sequence)—This token is positioned at the end of a text and
is especially useful when concatenating multiple unrelated texts, similar to
<|endoftext|>. For instance, when combining two different Wikipedia arti-
cles or books, the [EOS] token indicates where one ends and the next begins.

- [PAD] (padding)—When training LLMs with batch sizes larger than one, the
batch might contain texts of varying lengths. To ensure all texts have the same
length, the shorter texts are extended or “padded” using the [PAD] token, up to
the length of the longest text in the batch.

The tokenizer used for GPT models does not need any of these tokens; it only uses an
<|endoftext|> token for simplicity. <|endoftext|> is analogous to the [EOS] token.
<|endoftext|> is also used for padding.

However, as we’ll explore in subsequent
chapters, when training on batched inputs, we typically use a mask, meaning we don’t
attend to padded tokens. Thus, the specific token chosen for padding becomes incon-
sequential
"""
