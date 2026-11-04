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
