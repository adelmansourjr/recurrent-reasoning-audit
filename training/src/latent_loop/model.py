"""Continuous recurrent Transformer workspace; no LM head inside the loop.

The 28-layer Qwen architecture is retained as an encoder. A NEW shared small
Transformer reasons over pooled hidden states. This is a sidecar retrofit,
not a conversion of the pretrained Qwen stack into Huginn.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def __call__(self, q, memory):
        b, n, d = q.shape
        h = self.heads
        queries = self.q(q).reshape(b, n, h, d // h).transpose(0, 2, 1, 3)
        keys = self.k(memory).reshape(b, -1, h, d // h).transpose(0, 2, 1, 3)
        values = self.v(memory).reshape(b, -1, h, d // h).transpose(0, 2, 1, 3)
        weights = mx.softmax((queries @ keys.transpose(0, 1, 3, 2)) * (d // h) ** -.5, axis=-1)
        result = (weights @ values).transpose(0, 2, 1, 3).reshape(b, n, d)
        return self.out(result)


class Workspace(nn.Module):
    def __init__(self, hidden_size=2048, dim=128, slots=4, heads=4):
        super().__init__()
        self.project = nn.Linear(hidden_size, dim)
        self.seeds = mx.random.normal((slots, dim)) * .02
        self.cross = Attention(dim, heads)
        self.self_attn = Attention(dim, heads)
        self.n1, self.n2, self.n3 = nn.RMSNorm(dim), nn.RMSNorm(dim), nn.RMSNorm(dim)
        self.ff1, self.ff2 = nn.Linear(dim, dim * 2), nn.Linear(dim * 2, dim)
        self.gate = mx.array(-1.0)
        self.readout = nn.Linear(dim, 3)

    def __call__(self, concepts, loops=4, intervention=None):
        if loops < 0:
            raise ValueError("loops cannot be negative")
        memory = self.project(concepts.astype(mx.float32))
        query = memory[:, -1:, :]
        z0 = query + self.seeds[None]
        z = z0
        trace = []
        for _ in range(loops):
            if intervention == "reset":
                z = z0
            elif intervention == "zero":
                z = mx.zeros_like(z)
            # Re-inject the query; all memory remains accessible at every depth.
            z = z + nn.sigmoid(self.gate) * self.cross(self.n1(z + .1 * query), memory)
            z = z + self.self_attn(self.n2(z), self.n2(z))
            z = z + self.ff2(nn.silu(self.ff1(self.n3(z))))
            trace.append(mx.mean(z, axis=1))
        return self.readout(mx.mean(self.n3(z), axis=1)), trace


class QwenLoop(nn.Module):
    def __init__(self, backbone, dim=128, slots=4):
        super().__init__()
        self.backbone = backbone
        self.workspace = Workspace(backbone.args.hidden_size, dim, slots)

    def encode_prefix(self, ids):
        h = self.backbone.model.embed_tokens(ids)
        mask = create_attention_mask(h, None)
        for layer in self.backbone.model.layers[:-1]:
            h = layer(h, mask, None)
        return mx.stop_gradient(h)

    def concepts(self, prefix, boundaries):
        # Prefix is safe to cache: only final-layer q/v adapters are trainable.
        h = self.backbone.model.layers[-1](prefix, create_attention_mask(prefix, None), None)
        h = self.backbone.model.norm(h)
        return h[:, boundaries, :]

    def __call__(self, prefix, boundaries, loops=4, intervention=None):
        return self.workspace(self.concepts(prefix, boundaries), loops, intervention)
