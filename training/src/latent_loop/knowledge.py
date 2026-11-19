"""Explain only source-backed concept associations; keep unknowns explicit.

This indexes model-inferred predicates against original memory evidence. It
does not certify that a predicate is a faithful interpretation of that evidence.
"""
from __future__ import annotations
from latent_loop.logic import parse,concept_graph


class ConceptKnowledge:
    def __init__(self,records,formulas):
        if len(records)!=len(formulas):raise ValueError('One checked representation per source record is required')
        self.records=list(records);self.formulas=list(formulas)
        self.graph=concept_graph([parse(f) for f in formulas]);self.index={}
        by_id={n['id']:n for n in self.graph['nodes']}
        for edge in self.graph['edges']:
            if edge['role']=='instantiates':
                concept=by_id[edge['target']];occurrence=by_id[edge['source']]
                self.index.setdefault(concept['name'].casefold(),dict(name=concept['name'],indices=set(),arities=set()))
                entry=self.index[concept['name'].casefold()];entry['indices'].add(occurrence['formula']);entry['arities'].add(concept['arity'])
    def explain(self,name):
        if not isinstance(name,str) or not name.strip():raise ValueError('Concept name required')
        entry=self.index.get(name.strip().casefold())
        if entry is None:
            return dict(concept=name,status='no_grounded_concept',evidence=[],related_concepts=[],
                text=f'I do not have enough grounded knowledge to explain {name}. I need a definition or relevant evidence.',
                grounding_verified=False,knowledge_scope='Supplied evidence only; absence is not a statement about all Qwen pretraining knowledge')
        indices=entry['indices'];related=[]
        for key,value in self.index.items():
            shared=sorted(indices&value['indices'])
            if shared and key!=name.strip().casefold():related.append(dict(concept=value['name'],source_indices=shared,link='co-occurs in a scoped logical formula; not necessarily implication'))
        evidence=[dict(index=i,formula=self.formulas[i],**self.records[i]) for i in sorted(indices)]
        definitions=[r for r in evidence if r.get('kind')=='definition']
        status='source_backed_definition' if definitions else 'partial_source_backed_knowledge'
        text=(f'My available evidence about {entry["name"]} is: '+
            ' '.join(f'“{r["text"]}” (source: {r["source"]}, id: {r["id"]}).' for r in evidence))
        if not definitions:text+=' I do not have an explicit definition or enough evidence for a broader explanation.'
        return dict(concept=entry['name'],status=status,evidence=evidence,related_concepts=related,
            text=text,grounding_verified=False,knowledge_scope='Supplied evidence; associations retain their original logical scope',
            graph=self.graph)
