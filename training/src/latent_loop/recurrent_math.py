"""Shared Qwen tail recurrence with separate per-iteration attention caches.

The learned tail has one set of LoRA parameters. Refinement steps emit no text;
the vocabulary head is read only after the final continuous refinement.
"""
from __future__ import annotations
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache


def refine(backbone,h,loops,cache=None,mask=None):
    if not 1<=loops<=8:raise ValueError('Use 1 to 8 validated recurrent depths')
    if cache is None:cache=[None]*(2*loops)
    if len(cache)!=2*loops:raise ValueError('Each iteration needs its own two-layer cache')
    for k in range(loops):
        state=h
        for j,layer in enumerate(backbone.model.layers[-2:]):state=layer(state,mask,cache[2*k+j])
        h=state if loops==1 else h+(state-h)/loops
    return h


class RecurrentTailTrain(nn.Module):
    def __init__(self,backbone,loops=3):
        super().__init__();self.backbone=backbone;self.loops=loops
    def __call__(self,h,start,targets):
        h=refine(self.backbone,h,self.loops,mask=create_attention_mask(h,None))
        h=self.backbone.model.norm(h)[:,start:,:]
        logits=self.backbone.model.embed_tokens.as_linear(h).astype(mx.float32)
        return nn.losses.cross_entropy(logits,targets,reduction='mean')


class RecurrentQwen(nn.Module):
    def __init__(self,backbone,loops=3):
        super().__init__();self.backbone=backbone;self.loops=loops
        if len(backbone.model.layers)!=28:raise ValueError('This wrapper is for the inspected28-layer Qwen3')
    @property
    def layers(self):
        # A property for cache construction; shared modules are registered only once.
        return self.backbone.model.layers[:-2]+self.backbone.model.layers[-2:]*self.loops
    @property
    def args(self):return self.backbone.args
    @property
    def model_type(self):return self.backbone.model_type
    def make_cache(self):return [KVCache() for _ in self.layers]
    def __call__(self,inputs,cache=None,input_embeddings=None):
        cache=cache if cache is not None else [None]*len(self.layers)
        if len(cache)!=len(self.layers):raise ValueError('Wrong recurrent cache count')
        h=input_embeddings if input_embeddings is not None else self.backbone.model.embed_tokens(inputs)
        mask=create_attention_mask(h,cache[0])
        for layer,c in zip(self.backbone.model.layers[:-2],cache[:26]):h=layer(h,mask,c)
        h=refine(self.backbone,h,self.loops,cache[26:],mask)
        h=self.backbone.model.norm(h)
        return self.backbone.model.embed_tokens.as_linear(h)
