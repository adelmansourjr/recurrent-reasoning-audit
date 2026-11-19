"""Learned open-vocabulary proposition links and a continuous recurrent workspace.

Nodes represent language propositions, not verified logical atoms. Learned links
estimate shared predicates; a link is neither implication nor a proof step.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask


class LanguageReasoner(nn.Module):
    def __init__(self, backbone, mode="graph", dim=64):
        super().__init__()
        self.backbone = backbone
        self.mode = mode
        self.project = nn.Linear(2048, dim)
        self.norm = nn.RMSNorm(dim)
        self.link_q = nn.Linear(dim, dim, bias=False)
        self.link_k = nn.Linear(dim, dim, bias=False)
        self.link_bias = mx.array(0.)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.n1, self.n2 = nn.RMSNorm(dim), nn.RMSNorm(dim)
        self.ff1, self.ff2 = nn.Linear(dim, 2*dim), nn.Linear(2*dim, dim)
        self.gate = mx.array(-1.)
        self.readout = nn.Linear(3*dim, 3)
        # Both controls have the same registered parameters. In MLP mode, the
        # shared blocks act independently on nodes, without cross-node messages.

    def features(self, prefix, spans):
        h = self.backbone.model.layers[-1](prefix, create_attention_mask(prefix, None), None)
        h = self.backbone.model.norm(h).astype(mx.float32)
        return mx.stack([.5*h[:, b-1] + .5*mx.mean(h[:, a:b], axis=1) for a,b in spans], axis=1)

    def reason(self, features, loops=4, reset=False, uniform_links=False):
        z0 = self.norm(self.project(features))
        z = z0
        a, b = self.link_q(z0), self.link_k(z0)
        raw = (a @ b.transpose(0,2,1)) / z.shape[-1]**.5
        raw = .5*(raw + raw.transpose(0,2,1)) + self.link_bias
        links = nn.sigmoid(raw)
        for _ in range(loops):
            if reset:
                z = z0
            h = self.n1(z + .1*z0)
            if self.mode == "graph":
                scores = (self.q(h) @ self.k(h).transpose(0,2,1)) / z.shape[-1]**.5
                if not uniform_links:
                    scores = scores + mx.log(.05 + .95*links)
                msg = self.out(mx.softmax(scores, axis=-1) @ self.v(h))
            else:
                # Uses the same q/k/v/out weights, independently per node.
                msg = self.out(self.v(h)*nn.sigmoid(self.q(h)+self.k(h)
                    +.5*(self.link_q(h)+self.link_k(h))+self.link_bias))
            z = z + nn.sigmoid(self.gate)*msg
            z = z + .25*self.ff2(nn.silu(self.ff1(self.n2(z))))
        z = self.norm(z)
        q = z[:, -1]
        memory = mx.mean(z[:, :-1], axis=1) if z.shape[1]>1 else mx.zeros_like(q)
        return self.readout(mx.concatenate([q, memory, q*memory], axis=-1)), raw

    def __call__(self, prefix, spans, loops=4, reset=False, uniform_links=False):
        return self.reason(self.features(prefix, spans), loops, reset, uniform_links)
