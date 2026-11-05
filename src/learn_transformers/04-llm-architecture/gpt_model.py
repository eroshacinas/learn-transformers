# %%
import torch
import torch.nn as nn


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg)
            for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        x = self.trf_blocks(x)
        self.final_norm(x)

        logits = self.out_head(x)

        return logits

"""
the following dummy blocks are placeholders for now; we'll code them later on
"""

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x


"""
The main idea behind layer normalization is to adjust the activa-
tions (outputs) of a neural network layer to have a mean of 0 and a variance of 1, also
known as unit variance

This adjustment speeds up the convergence to effective
weights and ensures consistent, reliable training.

In GPT-2 and modern transformer
architectures, layer normalization is typically applied before and after the multi-head
attention module.

The scale and shift are two trainable parameters (of the
same dimension as the input) that the LLM automatically adjusts during training if it
is determined that doing so would improve the model’s performance on its training
task. This allows the model to learn appropriate scaling and shifting that best suit the
data it is processing
"""

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))


    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        """
        This approach does not apply Bessel’s correction, which typically uses n – 1 instead of n in the denomi-
        nator to adjust for bias in sample variance estimation. This decision results in a so-
        called biased estimate of the variance. For LLMs, where the embedding dimension n
        is significantly large, the difference between using n and n – 1 is practically negligible
        """
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        norm_x = (x-mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift


"""
LayerNorm vs BatchNorm

If you are familiar with batch normalization, a common and traditional normalization
method for neural networks, you may wonder how it compares to layer normalization.
Unlike batch normalization, which normalizes across the batch dimension, layer nor-
malization normalizes across the feature dimension. LLMs often require significant
computational resources, and the available hardware or the specific use case can
dictate the batch size during training or inference.

Since layer normalization normal-
izes each input independently of the batch size, it offers more flexibility and stability
in these scenarios. This is particularly beneficial for distributed training or when
deploying models in environments where resources are constrained.


"""
