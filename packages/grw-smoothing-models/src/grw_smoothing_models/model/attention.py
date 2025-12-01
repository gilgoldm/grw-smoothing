import torch
import torchvision
from torch import nn


class TransformerParams:
    def __init__(self, embed_dim: int, depth: int, num_heads: int, mlp_dim: int, num_classes: int):
        self.embed_dim: int = embed_dim
        self.depth: int = depth
        self.num_heads: int = num_heads
        self.mlp_dim: int = mlp_dim
        self.num_classes: int = num_classes

    def __str__(self):
        return f'TransformerParams = {self.__dict__}'


def posemb_sincos_1d(tokens, temperature=10000):
    _, N, dim, device, dtype = *tokens.shape, tokens.device, tokens.dtype

    N = torch.arange(N, device=device)
    assert (dim % 2) == 0, 'feature dimension must be multiple of 2 for sincos emb'
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    N = N.flatten()[:, None] * omega[None, :]
    pe = torch.cat((N.sin(), N.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_dim, stochastic_depth):
        super().__init__()
        self.stochastic_depth = stochastic_depth
        self.norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(embed_dim),
                nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True),
                FeedForward(embed_dim, mlp_dim)
            ]))

    def forward(self, x):
        for norm, attn, ff in self.layers:
            x_ = norm(x)
            x = x + torchvision.ops.stochastic_depth(attn(x_,x_,x_,need_weights=False)[0], p=self.stochastic_depth, mode="row", training=self.training)
            x = x + torchvision.ops.stochastic_depth(ff(x), p=self.stochastic_depth, mode="row", training=self.training)
        return self.norm(x)


class SimpleViT(nn.Module):
    def __init__(self, *, transformer_params: TransformerParams, dropout: float =0., stochastic_depth:float =0.):
        super().__init__()
        self.stochastic_depth: float = stochastic_depth
        self.dropout: float = dropout
        self.transformer = Transformer(embed_dim=transformer_params.embed_dim, depth=transformer_params.depth, num_heads=transformer_params.num_heads, mlp_dim= transformer_params.mlp_dim, stochastic_depth=stochastic_depth)
        self.pool = "mean"
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(transformer_params.embed_dim, transformer_params.num_classes)

    def forward(self, x):  # (B,num_frames, dim)
        x = x + posemb_sincos_1d(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = nn.functional.dropout(p=self.dropout, input=x, training=self.training)
        return self.linear_head(x)

