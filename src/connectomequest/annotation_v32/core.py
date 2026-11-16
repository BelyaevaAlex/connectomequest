"""Public typed queries and deterministic, model-independent certificates."""
from dataclasses import dataclass, field
from functools import cached_property
from collections.abc import Mapping
from types import MappingProxyType
from numbers import Integral
from typing import Any


def _integer(value, name):
    if not isinstance(value,Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return int(value)


def _identifier(value,name):
    if not isinstance(value,str) or not value:
        raise ValueError(f'{name} must be a nonempty string')
    return value


@dataclass(frozen=True)
class Witness:
    witness_id: str
    requirements: tuple[tuple[int,int],...]
    vertices: tuple[int,...] = ()

    def __post_init__(self):
        _identifier(self.witness_id,'witness ID')
        object.__setattr__(self,'requirements',tuple((_integer(v,'cell ID'),_integer(t,'category')) for v,t in self.requirements))
        object.__setattr__(self,'vertices',tuple(_integer(v,'vertex ID') for v in self.vertices))


@dataclass(frozen=True)
class Query:
    query_id: str
    witnesses: tuple[Witness,...]
    metadata: Mapping[str,Any] = field(default_factory=dict)

    def __post_init__(self):
        _identifier(self.query_id,'query ID')
        witnesses=tuple(self.witnesses)
        if not all(isinstance(w,Witness) for w in witnesses):
            raise ValueError('query witnesses must be Witness instances')
        ids={}
        for w in witnesses:
            if w.witness_id in ids and ids[w.witness_id]!=w.requirements:
                raise ValueError('one witness ID cannot refer to different requirements')
            ids[w.witness_id]=w.requirements
        object.__setattr__(self,'witnesses',witnesses)
        object.__setattr__(self,'metadata',MappingProxyType(dict(self.metadata)))

    @cached_property
    def compiled(self):
        from .compiled import compile_query
        if any(t > 9223372036854775807 for w in self.witnesses for _,t in w.requirements): return None
        return compile_query(self)

    @cached_property
    def cells(self):
        return tuple(sorted({v for w in self.witnesses for v,_ in w.requirements}))


@dataclass(frozen=True)
class Verification:
    status: str
    witness_id: str | None
    facts: tuple[tuple[int,int],...]
    blocked_witnesses: tuple[str,...] = ()


def canonical_query(query_id,witnesses,metadata=None):
    ids={}; unique={}
    for w in witnesses:
        if not isinstance(w,Witness):
            raise ValueError('expected Witness')
        requirements=tuple(sorted(set(w.requirements)))
        if w.witness_id in ids and ids[w.witness_id]!=requirements:
            raise ValueError('one witness ID cannot refer to different requirements')
        ids[w.witness_id]=requirements
        if len({v for v,_ in requirements}) != len(requirements):
            continue
        candidate=Witness(w.witness_id,requirements,w.vertices)
        existing=unique.get(requirements)
        if existing is None or candidate.witness_id < existing.witness_id:
            unique[requirements]=candidate
    return Query(query_id,tuple(sorted(unique.values(),key=lambda w:w.witness_id)),metadata or {})


def _observations(observed):
    return {_integer(v,'cell ID'):None if t is None else _integer(t,'category') for v,t in observed.items()}


def live_witnesses(query,observed):
    if len(query.witnesses) >= 128 and query.compiled is not None: return query.compiled.live_witnesses(observed)
    observed=_observations(observed)
    return tuple(w for w in query.witnesses if all(observed.get(v) is None or observed[v]==t for v,t in w.requirements))


def active_cells(query,observed):
    if len(query.witnesses) >= 128 and query.compiled is not None: return query.compiled.active_cells(observed)
    return tuple(sorted({v for w in live_witnesses(query,observed) for v,_ in w.requirements if v not in observed}))


def evaluate(query,observed):
    if len(query.witnesses) >= 128 and query.compiled is not None: return query.compiled.evaluate(observed)
    if query.metadata.get('computationally_unsupported'):
        raise ValueError('computationally unsupported query cannot be verified')
    observed=_observations(observed)
    blockers=[]; blocked=[]
    for w in sorted(query.witnesses,key=lambda w:w.witness_id):
        if all(observed.get(v)==t for v,t in w.requirements):
            return Verification('positive',w.witness_id,tuple(sorted(set(w.requirements))))
        mismatch=[(v,observed[v]) for v,t in w.requirements if observed.get(v) is not None and observed[v]!=t]
        if mismatch:
            blockers.append(min(mismatch)); blocked.append(w.witness_id)
    if len(blocked)==len(query.witnesses):
        return Verification('negative',None,tuple(sorted(set(blockers))),tuple(blocked))
    return Verification('unresolved',None,())


def replay_certificate(query,result):
    """Validate only the supplied facts and public query; no model is consulted."""
    try:
        facts={}
        for v,t in result.facts:
            v=_integer(v,'cell ID'); t=_integer(t,'category')
            if v in facts and facts[v]!=t:
                return False
            facts[v]=t
        if result.status=='positive':
            matches=[w for w in query.witnesses if w.witness_id==result.witness_id]
            return bool(matches) and not result.blocked_witnesses and any(all(facts.get(v)==t for v,t in w.requirements) for w in matches)
        if result.status=='negative':
            return result.witness_id is None and tuple(sorted(result.blocked_witnesses))==tuple(sorted(w.witness_id for w in query.witnesses)) and evaluate(query,facts).status=='negative'
    except (TypeError,ValueError,AttributeError):
        return False
    return False


def query_to_dict(query):
    return {'query_id':query.query_id,'witnesses':[{'witness_id':w.witness_id,'requirements':[list(r) for r in w.requirements],'vertices':list(w.vertices)} for w in query.witnesses],'metadata':dict(query.metadata)}


def query_from_dict(payload):
    return canonical_query(payload['query_id'],[Witness(w['witness_id'],tuple(tuple(r) for r in w['requirements']),tuple(w.get('vertices',()))) for w in payload['witnesses']],payload.get('metadata',{}))
