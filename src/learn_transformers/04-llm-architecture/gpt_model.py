# %%
import torch
import torch.nn as nn

# refer to page 95 for what each variable means
GPT_CONFIG = {
    "vocab_size": 50_257,
    "context_length": 1_024,
    "embed_dim": 768,
    "n_heads": 12, # number of attention heads
    "n_layers": 12, # number of layers
    "drop_rate": 0.1, # dropout rate
    "qkv_bias": False
}

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg)
            for _ in range(cfg["n_layers"])])

        self.final_norm = DummyLayerNorm(cfg["embed_dim"])
        self.out_head = nn.Linear(cfg["embed_dim"], cfg["vocab_size"], bias=False)

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



class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()


    def forward(self, x):
        return x
