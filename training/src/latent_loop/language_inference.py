"""Natural-language input, inspectable learned links, and restrained answer rendering."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from latent_loop.language_data import LABELS, encode_text
from latent_loop.experiment import DEFAULT_MODEL, load_model, attach_lora, sha256, write_json


class EvidenceMemory:
    """Explicit fact memory with revisions and provenance; no inferred facts inserted."""
    def __init__(self,records=None):
        self.records=list(records or [])
        if any(not all(k in r for k in ("id","text","source")) for r in self.records):
            raise ValueError("Each memory record needs id, text and source")
    def remember(self,fact_id,text,source="user"):
        if not fact_id or not text.strip():raise ValueError("Empty fact id or text")
        self.records.append(dict(id=str(fact_id),text=text.strip(),source=str(source)))
    def current(self):
        latest={r["id"]:r for r in self.records}
        return list(latest.values())
    def save(self,path):write_json(path,self.records)


class LanguageRuntime:
    def __init__(self,adapter="results/v5/graph-lora-aux0.1-17",allow_control=False,cached_only=False):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from latent_loop.language_model import LanguageReasoner
        path=Path(adapter)
        cfg=json.loads((path/"config.json").read_text())
        manifest_path=path.parent/"cache/manifest.json"
        if sha256(manifest_path)!=cfg["manifest_sha256"]:raise ValueError("Cache manifest changed")
        manifest=json.loads(manifest_path.read_text())
        for filename,key in (("model.safetensors","checkpoint_sha256"),("tokenizer.json","tokenizer_sha256")):
            if sha256(Path(DEFAULT_MODEL)/filename)!=manifest[key]:raise ValueError(f"Mismatched {filename}")
        wrapper,self.tokenizer=load_model(DEFAULT_MODEL,cfg["seed"],cached_final_only=cached_only)
        if cfg["lora"]:attach_lora(wrapper)
        mx.random.seed(cfg["seed"]+7919)
        self.model=LanguageReasoner(wrapper.backbone,cfg["mode"],dim=cfg['dim'])
        weights=mx.load(str(path/"trainable.safetensors"))
        expected=dict(tree_flatten(self.model.trainable_parameters()))
        if set(weights)!=set(expected) or any(weights[k].shape!=expected[k].shape for k in expected):
            raise ValueError("Adapter keys/shapes differ from architecture")
        self.model.load_weights(list(weights.items()),strict=False)
        self.adapter=str(path.resolve())
        self.cached_only=cached_only
        self.variant=cfg.get("variant","full")
        self.calibration=None
        calibration_path=Path(f"results/v8/calibration/temperature-{cfg['seed']}.json")
        if not allow_control and calibration_path.exists():
            calibration=json.loads(calibration_path.read_text())
            if (calibration['weights_sha256']==sha256(path/'trainable.safetensors')
                and calibration['config_sha256']==sha256(path/'config.json')):
                self.calibration=calibration
        if self.variant!="full" and not allow_control:raise ValueError("A conclusion-only control is not a deployed reasoner")
    def decide(self,premises,conclusion,loops=4):
        import mlx.core as mx
        import numpy as np
        if self.cached_only:raise ValueError('This runtime was loaded for cached benchmark features only')
        if not premises or not all(isinstance(p,str) and p.strip() for p in premises):
            raise ValueError("Provide nonempty natural-language premises")
        if not isinstance(conclusion,str) or not conclusion.strip():raise ValueError("Empty conclusion")
        if not 1<=loops<=64:raise ValueError("loops must be between 1 and 64")
        ids,spans=encode_text(self.tokenizer,premises,conclusion)
        if len(ids)>1024:raise ValueError("Context exceeds tested 1024-token limit; no evidence was truncated")
        start=time.perf_counter()
        # Reuse the exact frozen cache boundary used in training, computed afresh.
        h=self.model.backbone.model.embed_tokens(mx.array([ids]))
        from mlx_lm.models.base import create_attention_mask
        mask=create_attention_mask(h,None)
        for layer in self.model.backbone.model.layers[:-1]:h=layer(h,mask,None)
        logits,raw=self.model(mx.stop_gradient(h),spans,loops)
        p=mx.softmax(logits,axis=-1)[0];g=mx.sigmoid(raw[0]);mx.eval(p,g)
        values=np.array(p).tolist();raw_values=list(values);links=np.array(g)
        from latent_loop.calibration import rescale,decide
        calibration=self.calibration if self.calibration and loops==self.calibration['latent_steps'] else None
        if calibration:values=rescale(values,calibration['temperature'])
        label=LABELS[int(np.argmax(values))]
        pair_links=[dict(premise_a=i,premise_b=j,strength=float(links[i,j]))
            for i in range(len(premises)) for j in range(i+1,len(premises))]
        return dict(label=label,probabilities=dict(zip(LABELS,values)),raw_probabilities=dict(zip(LABELS,raw_values)),
            text=f"The model classifies the conclusion as {label.lower()} from the supplied premises.",
            conclusion=conclusion,premises=premises,concept_sharing_links=pair_links,
            query_links=[dict(premise=i,strength=float(links[i,-1])) for i in range(len(premises))],
            proof_verified=False,temperature_scaling_applied=bool(calibration),
            calibration=calibration,probabilities_calibrated_for_current_domain=False,
            confidence_decision=decide(values),
            tokens=len(ids),latent_steps=loops,seconds=time.perf_counter()-start,adapter=self.adapter)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",help="JSON containing premises and conclusion, optionally memory records")
    p.add_argument("--adapter",default="results/v5/graph-lora-aux0.1-17")
    p.add_argument("--out",default="results/v5/language-example.json")
    a=p.parse_args();request=json.loads(Path(a.input).read_text())
    memory=EvidenceMemory(request.get("memory"))
    premises=[r["text"] for r in memory.current()]+request.get("premises",[])
    result=LanguageRuntime(a.adapter).decide(premises,request["conclusion"],request.get("loops",4))
    result["memory_provenance"]=memory.current()
    write_json(a.out,result);print(json.dumps(result,indent=2))


if __name__=="__main__":main()
