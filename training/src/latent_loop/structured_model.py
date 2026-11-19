"""A learned graph-masked recurrent Transformer on grounded belief vectors.

Only the dependency mask and boundary anchoring are architectural priors. The
categorical conjunction rule is learned. The exact min oracle is NOT called.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask


class Grounder(nn.Module):
    def __init__(self,backbone,head="linear"):
        super().__init__()
        self.backbone=backbone
        self.head_kind=head
        if head=="mlp":
            self.hidden=nn.Linear(backbone.args.hidden_size,31)
            self.readout=nn.Linear(31,3)
        else:
            self.readout=nn.Linear(backbone.args.hidden_size,3)

    def __call__(self,prefix):
        h=self.backbone.model.layers[-1](prefix,create_attention_mask(prefix,None),None)
        h=self.backbone.model.norm(h)[:,-1,:].astype(mx.float32)
        if self.head_kind=="mlp":
            h=nn.silu(self.hidden(h))
        return self.readout(h)


class GraphCell(nn.Module):
    def __init__(self,dim=48,heads=3):
        super().__init__()
        self.dim,self.heads=dim,heads
        self.embed=nn.Linear(3,dim)
        self.key=nn.Linear(dim,dim,bias=False)
        self.value=nn.Linear(dim,dim,bias=False)
        self.query=mx.random.normal((heads,dim//heads))*.1
        self.out=nn.Linear(dim,dim)
        self.n1,self.n2=nn.RMSNorm(dim),nn.RMSNorm(dim)
        self.ff1,self.ff2=nn.Linear(dim,dim*2),nn.Linear(dim*2,dim)
        self.readout=nn.Linear(dim,3)

    def priority_scores(self):
        """Canonical-state logits for a training-time ordering constraint."""
        keys=self.key(self.embed(mx.eye(3))).reshape(3,self.heads,-1).transpose(1,0,2)
        return mx.sum(keys*self.query[:,None,:],axis=-1)/(self.dim//self.heads)**.5

    def __call__(self,p,adjacency):
        b,n,_=p.shape
        h=self.embed(p)
        keys=self.key(h).reshape(b,n,self.heads,-1).transpose(0,2,1,3)
        values=self.value(h).reshape(b,n,self.heads,-1).transpose(0,2,1,3)
        # Every node asks the same learned local operator of its prerequisites.
        score=mx.sum(keys*self.query[None,:,None,:],axis=-1)/(self.dim//self.heads)**.5
        score=score[:,:,None,:]+mx.where(adjacency[:,None,:,:]>0,0.0,-1e9)
        weights=mx.softmax(score,axis=-1)
        message=(weights@values).transpose(0,2,1,3).reshape(b,n,self.dim)
        z=self.out(message)
        z=z+self.ff2(nn.silu(self.ff1(self.n1(z))))
        return mx.softmax(self.readout(self.n2(z)),axis=-1)


class BeliefGraph(nn.Module):
    def __init__(self,dim=48):
        super().__init__()
        self.cell=GraphCell(dim)

    def __call__(self,initial,adjacency,leaves,steps,reset=False):
        p=initial
        trace=[]
        for _ in range(steps):
            if reset:
                p=initial
            update=self.cell(p,adjacency)
            p=mx.where(leaves[:,:,None],initial,update)
            trace.append(p)
        return p,trace
