"""Acquisition heuristics with a common, probability-independent verifier.

Policies receive only public queries, observations, and probability providers.
Costs here are unique annotation reads (unit cost). No approximation guarantee
is asserted for sampled EC2 or the certificate-aware heuristic.
"""
import numpy as np
from .core import evaluate, live_witnesses, active_cells
from .priors import ImpossibleObservation


def _unresolved(query, observed):
    return evaluate(query, observed).status == 'unresolved'


def certificate_scores(query, observed, prior):
    if len(query.witnesses) >= 128 and query.compiled is not None: return query.compiled.certificate_scores(observed,prior)
    cells = active_cells(query, observed)
    rows = prior.probabilities(cells, observed)
    probabilities = dict(zip(cells, rows))
    reject = {v:0. for v in cells}
    incidence = {v:0 for v in cells}
    completion = {v:set() for v in cells}
    for witness in live_witnesses(query, observed):
        missing = [(v,t) for v,t in witness.requirements if observed.get(v) is None]
        for v,t in missing:
            if v in probabilities:
                reject[v] += 1 - probabilities[v][t]
                incidence[v] += 1
        if len(missing) == 1 and missing[0][0] in probabilities:
            v,t = missing[0]
            completion[v].add(t)
    return {v:(float(sum(probabilities[v][t] for t in completion[v])), reject[v], incidence[v]) for v in cells}


def _fallback(query, observed):
    if len(query.witnesses) >= 128 and query.compiled is not None and hasattr(query.compiled,'incidence'):
        counts=query.compiled.incidence(observed)
        return min(counts,key=lambda v:(-counts[v],v)) if counts else None
    counts = dict.fromkeys(active_cells(query, observed), 0)
    for w in live_witnesses(query, observed):
        for v,_ in w.requirements:
            if v in counts:
                counts[v] += 1
    return min(counts, key=lambda v:(-counts[v],v)) if counts else None


class CertificateGreedy:
    """Prior-guided completion probability, then expected witness rejection."""
    def __init__(self, *, max_probability_entries=2_000_000):
        if max_probability_entries < 1:
            raise ValueError('max_probability_entries must be positive')
        self.max_probability_entries = max_probability_entries
        self.diagnostics = {}

    def begin_query(self, query, observed):
        self.diagnostics = {'fallback_count':0}

    def choose_next(self, query, observed, prior, rng):
        if not _unresolved(query, observed):
            return None
        if len(active_cells(query,observed))*prior.label_count > self.max_probability_entries:
            self.diagnostics['fallback_count'] = self.diagnostics.get('fallback_count',0)+1
            self.diagnostics['fallback_reason'] = 'probability_entry_budget'
            return _fallback(query,observed)
        try:
            scores = certificate_scores(query, observed, prior)
        except ImpossibleObservation:
            self.diagnostics['fallback_count'] = self.diagnostics.get('fallback_count',0)+1
            self.diagnostics['fallback_reason'] = 'zero_model_mass'
            return _fallback(query, observed)
        return min(scores, key=lambda v:(*(-x for x in scores[v]),v)) if scores else None


def sampled_fixed_costs(query, observed, cells, samples, order):
    """Exact stopping read-count for each sampled world, without reading truth.

    This vectorizes earliest positive witness and earliest all-witness rejection.
    Unresolved attempted reads (None) have infinite reveal time in every world.
    """
    if len(query.witnesses) >= 128 and query.compiled is not None: return query.compiled.fixed_costs(observed,cells,samples,order)
    samples = np.asarray(samples)
    count = len(samples)
    columns = {v:i for i,v in enumerate(cells)}
    positions = {v:i+1 for i,v in enumerate(order)}
    positive = np.full(count, np.inf)
    negative = np.zeros(count)
    for witness in query.witnesses:
        matched = np.ones(count, dtype=bool)
        completion_time = 0.
        block_time = np.full(count, np.inf)
        for v,t in witness.requirements:
            if v in observed and observed[v] is not None:
                equals = np.full(count, observed[v] == t)
                time = 0.
            else:
                equals = samples[:,columns[v]] == t
                time = positions.get(v, np.inf)
            matched &= equals
            completion_time = max(completion_time,time)
            block_time = np.minimum(block_time,np.where(equals,np.inf,time))
        positive = np.minimum(positive,np.where(matched,completion_time,np.inf))
        negative = np.maximum(negative,block_time)
    return np.minimum(positive,negative)


def _numeric_witness_order(query,observed,prior,*,cells=None,probs=None,first_only=False):
    compiled=query.compiled
    if compiled is None: raise ValueError('query cannot be compiled')
    if cells is None: cells=active_cells(query,observed)
    if probs is None: probs=dict(zip(cells,prior.probabilities(cells,observed)))
    _,_,mismatch=compiled._state(observed)
    live=~mismatch.any(axis=1)
    n=len(query.witnesses)
    priority=np.zeros(n); missing_count=np.zeros(n,dtype=np.int32); blocked=np.zeros(n,dtype=bool)
    row_index=np.full(len(compiled.cells),-1,dtype=np.int64)
    for i,v in enumerate(cells): row_index[compiled.columns[v]]=i
    is_none=np.array([v in observed and observed[v] is None for v in compiled.cells],dtype=bool)
    for j in range(compiled.index.shape[1]):
        indices=compiled.index[:,j]; valid=indices>=0; safe=np.maximum(indices,0)
        blocked |= valid & is_none[safe]
        eligible=valid & live & (row_index[safe]>=0)
        missing_count[eligible]+=1
        for t in np.unique(compiled.labels[eligible,j]):
            rows=eligible & (compiled.labels[:,j]==t)
            values=np.array([probs[v][t] for v in cells],dtype=np.float64)
            priority[rows]+=np.log(np.maximum(values[row_index[indices[rows]]],np.finfo(float).tiny))
    keep=live & (missing_count>0)
    priority[keep]-=np.log(missing_count[keep])
    ordered=compiled.order[keep[compiled.order]]
    if first_only:
        if not len(ordered): return cells[0] if cells else None
        available=ordered[blocked[ordered]==np.min(blocked[ordered])]
        best=np.max(priority[available])
        row=available[np.flatnonzero(priority[available]==best)[0]]
        missing=[(v,t) for v,t in query.witnesses[row].requirements if v in probs]
        return min(missing,key=lambda vt:(probs[vt[0]][vt[1]],vt[0]))[0]
    rank=np.lexsort((np.arange(len(ordered)), -priority[ordered], blocked[ordered]))
    result=[]; seen=set()
    for row in ordered[rank]:
        missing=[(v,t) for v,t in query.witnesses[row].requirements if v in probs]
        for v,t in sorted(missing,key=lambda vt:(probs[vt[0]][vt[1]],vt[0])):
            if v not in seen: result.append(v);seen.add(v)
        if len(seen)==len(cells): break
    result.extend(v for v in cells if v not in seen)
    return tuple(result)


def witness_first_order(query, observed, prior, *, cells=None, probs=None, first_only=False):
    """The fixed library witness order, also used by adaptive witness policy."""
    if cells is None: cells=active_cells(query,observed)
    if probs is None: probs=dict(zip(cells,prior.probabilities(cells,observed)))
    if len(query.witnesses) >= 128 and query.compiled is not None:
        return _numeric_witness_order(query,observed,prior,cells=cells,probs=probs,first_only=first_only)
    witnesses = []
    for w in live_witnesses(query,observed):
        missing = [(v,t) for v,t in w.requirements if v in probs]
        if not missing:
            continue
        # Log success probability / remaining length; None literals prevent
        # completion and therefore place such witnesses after completable ones.
        blocked_by_none = any(v in observed and observed[v] is None for v,_ in w.requirements)
        log_priority = sum(np.log(max(float(probs[v][t]),np.finfo(float).tiny)) for v,t in missing)-np.log(len(missing))
        witnesses.append((blocked_by_none,-log_priority,w.witness_id,missing))
    witness_order = []
    seen = set()
    for _,_,_,missing in sorted(witnesses):
        for v,t in sorted(missing,key=lambda vt:(probs[vt[0]][vt[1]],vt[0])):
            if v not in seen:
                witness_order.append(v); seen.add(v)
    witness_order.extend(v for v in cells if v not in seen)
    return (witness_order[0] if witness_order else None) if first_only else tuple(witness_order)


def fixed_order_library(query, observed, prior):
    cells = active_cells(query, observed)
    probs = dict(zip(cells, prior.probabilities(cells,observed)))
    scores = certificate_scores(query,observed,prior)
    reject_order = tuple(sorted(cells,key=lambda v:(-scores[v][1],-scores[v][2],v)))
    incidence_order = tuple(sorted(cells,key=lambda v:(-scores[v][2],v)))
    witness_order = witness_first_order(query, observed, prior, cells=cells, probs=probs)
    return tuple(dict.fromkeys((reject_order,tuple(witness_order),incidence_order,tuple(cells))))


class AdaptiveWitnessPolicy:
    """Recompute the fixed-library witness-first rule after each observation."""
    def __init__(self, *, max_probability_entries=2_000_000):
        if max_probability_entries < 1: raise ValueError('max_probability_entries must be positive')
        self.max_probability_entries=max_probability_entries
        self.diagnostics={}

    def begin_query(self,query,observed):
        self.diagnostics={'fallback_count':0}

    def choose_next(self,query,observed,prior,rng):
        if not _unresolved(query,observed): return None
        cells=active_cells(query,observed)
        if len(cells)*prior.label_count > self.max_probability_entries:
            self.diagnostics['fallback_count']+=1
            self.diagnostics['fallback_reason']='probability_entry_budget'
            return _fallback(query,observed)
        try:
            return witness_first_order(query,observed,prior,cells=cells,first_only=True)
        except ImpossibleObservation:
            self.diagnostics['fallback_count']+=1
            self.diagnostics['fallback_reason']='zero_model_mass'
            return _fallback(query,observed)


class FixedOrderPolicy:
    """Select a modeled-best library order once per query, with exact stopping."""
    def __init__(self, *, n_samples=128, logical_skip=False, max_sample_entries=2_000_000):
        if not isinstance(n_samples,int) or n_samples < 1:
            raise ValueError('n_samples must be positive')
        self.n_samples = n_samples
        if max_sample_entries < 1:
            raise ValueError('max_sample_entries must be positive')
        self.max_sample_entries = max_sample_entries
        self.logical_skip = bool(logical_skip)
        self.order = None
        self.diagnostics = {}
        self._initial = None

    def begin_query(self, query, observed):
        self._initial = dict(observed)
        self.order = None
        self.diagnostics = {'fallback_count':0}

    def choose_next(self, query, observed, prior, rng):
        if not _unresolved(query,observed):
            return None
        if self._initial is None:
            raise RuntimeError('begin_query must precede acquisition')
        if self.order is None and len(query.cells)*max(self.n_samples,prior.label_count)>self.max_sample_entries:
            if len(query.witnesses) >= 128 and query.compiled is not None and hasattr(query.compiled,'incidence'):
                counts=query.compiled.incidence(self._initial)
            else:
                counts=dict.fromkeys(active_cells(query,self._initial),0)
                for witness in live_witnesses(query,self._initial):
                    for v,_ in witness.requirements:
                        if v in counts:
                            counts[v]+=1
            self.order=tuple(sorted(counts,key=lambda v:(-counts[v],v)))
            self.diagnostics.update(fallback_count=1,fallback_reason='sample_entry_budget')
        if self.order is None:
            try:
                orders = fixed_order_library(query,self._initial,prior)
                samples = prior.sample(query.cells,self._initial,self.n_samples,rng)
                costs = [float(np.mean(sampled_fixed_costs(query,self._initial,query.cells,samples,o))) for o in orders]
                selected = min(range(len(orders)),key=lambda i:(costs[i],orders[i]))
                self.order = orders[selected]
                self.diagnostics.update(library_costs=[c if np.isfinite(c) else None for c in costs], library_unresolved=[not np.isfinite(c) for c in costs], selected_library_index=selected)
            except ImpossibleObservation:
                self.order = tuple(active_cells(query,self._initial))
                self.diagnostics.update(fallback_count=1,fallback_reason='zero_model_mass')
        eligible = set(active_cells(query,observed)) if self.logical_skip else set(query.cells)
        return next((v for v in self.order if v not in observed and v in eligible),None)


def world_answers(query, cells, worlds):
    """Boolean values on model worlds; never used to authorize a certificate."""
    if len(query.witnesses) >= 128 and query.compiled is not None: return query.compiled.world_answers(cells,worlds)
    columns = {v:i for i,v in enumerate(cells)}
    answers = np.zeros(len(worlds),dtype=bool)
    for witness in query.witnesses:
        valid = np.ones(len(worlds),dtype=bool)
        for v,t in witness.requirements:
            valid &= worlds[:,columns[v]] == t
        answers |= valid
    return answers


def edge_scores(cells, worlds, weights, answers, label_count):
    """EC2 class-edge mass objective; weights normalized once, not per child."""
    weights = np.asarray(weights,dtype=float)
    class_one = weights * answers
    class_zero = weights * ~answers
    remaining = class_zero.sum()*class_one.sum()
    scores = {}
    for j,v in enumerate(cells):
        zero = np.bincount(worlds[:,j],weights=class_zero,minlength=label_count)
        one = np.bincount(worlds[:,j],weights=class_one,minlength=label_count)
        scores[v] = max(0.,float(remaining-np.dot(zero+one,zero*one)))
    return scores


class SampledEC2Policy:
    """Published EC2 acquisition estimated with joint samples, strict stopping.

    A deterministic incidence fallback handles zero sampled class-edge mass.
    Optional sample-entry limit bounds model workspace, never the query roster.
    """
    def __init__(self, *, n_samples=128, max_sample_entries=2_000_000):
        if not isinstance(n_samples,int) or n_samples < 1:
            raise ValueError('n_samples must be positive')
        if max_sample_entries < 1:
            raise ValueError('max_sample_entries must be positive')
        self.n_samples = n_samples
        self.max_sample_entries = max_sample_entries
        self.diagnostics = {}

    def begin_query(self, query, observed):
        self.diagnostics = {'fallback_count':0}

    def _fallback(self,query,observed,reason):
        self.diagnostics['fallback_count'] = self.diagnostics.get('fallback_count',0)+1
        self.diagnostics['fallback_reason'] = reason
        return _fallback(query,observed)

    def choose_next(self, query, observed, prior, rng):
        if not _unresolved(query,observed):
            return None
        candidates = active_cells(query,observed)
        if not candidates:
            return None
        cells = query.cells
        if len(cells)*max(self.n_samples,prior.label_count) > self.max_sample_entries:
            return self._fallback(query,observed,'sample_entry_budget')
        try:
            worlds = prior.sample(cells,observed,self.n_samples,rng)
        except ImpossibleObservation:
            return self._fallback(query,observed,'zero_model_mass')
        answers = world_answers(query,cells,worlds)
        scores = edge_scores(cells,worlds,np.full(len(worlds),1/len(worlds)),answers,prior.label_count)
        if max(scores[v] for v in candidates) <= 0:
            return self._fallback(query,observed,'zero_sampled_class_edge_mass')
        return min(candidates,key=lambda v:(-scores[v],v))
