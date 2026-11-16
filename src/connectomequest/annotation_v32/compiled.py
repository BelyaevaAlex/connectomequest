"""Exact numeric representation of an existing DNF query; no model or oracle."""
import numpy as np
from types import MappingProxyType
from .core import Verification, _observations


class CompiledQuery:
    def __setattr__(self,name,value):
        if name in self.__dict__: raise AttributeError('compiled query attributes are immutable')
        object.__setattr__(self,name,value)

    def __init__(self, query):
        self.query=query
        self.cells=tuple(sorted({v for w in query.witnesses for v,_ in w.requirements}))
        self.columns=MappingProxyType({v:i for i,v in enumerate(self.cells)})
        width=max((len(w.requirements) for w in query.witnesses),default=0)
        self.index=np.full((len(query.witnesses),width),-1,dtype=np.int32)
        self.labels=np.zeros(self.index.shape,dtype=np.int64)
        for i,w in enumerate(query.witnesses):
            for j,(v,t) in enumerate(w.requirements):
                self.index[i,j]=self.columns[v]; self.labels[i,j]=t
        self.mask=self.index>=0
        self.safe=np.maximum(self.index,0)
        self.order=np.asarray(sorted(range(len(query.witnesses)),key=lambda i:query.witnesses[i].witness_id),dtype=np.int32)
        for array in (self.index,self.labels,self.mask,self.safe,self.order): array.flags.writeable=False

    def _state(self,observed):
        observed=_observations(observed)
        values=np.full(len(self.cells)+1,-1,dtype=np.int64)
        for v,t in observed.items():
            if t is not None and v in self.columns: values[self.columns[v]]=t if t <= np.iinfo(np.int64).max else -2
        known=values[self.safe]!=-1
        mismatch=self.mask & known & (values[self.safe]!=self.labels)
        return observed,known,mismatch

    def live_witnesses(self,observed):
        _,_,mismatch=self._state(observed)
        return tuple(self.query.witnesses[i] for i in np.flatnonzero(~mismatch.any(axis=1)))

    def active_cells(self,observed):
        observed,_,mismatch=self._state(observed)
        live=~mismatch.any(axis=1)
        indices=np.unique(self.index[live][self.mask[live]])
        return tuple(self.cells[i] for i in indices if self.cells[i] not in observed)

    def evaluate(self,observed):
        if self.query.metadata.get('computationally_unsupported'): raise ValueError('computationally unsupported query cannot be verified')
        observed,known,mismatch=self._state(observed)
        positive=np.all(~self.mask | (known & ~mismatch),axis=1)
        found=self.order[positive[self.order]]
        if len(found):
            w=self.query.witnesses[found[0]]
            return Verification('positive',w.witness_id,tuple(sorted(set(w.requirements))))
        blocked=mismatch.any(axis=1)
        if blocked.all():
            blockers=[]; ids=[]
            for i in self.order:
                w=self.query.witnesses[i]
                blockers.append(min((v,observed[v]) for v,t in w.requirements if observed.get(v) is not None and observed[v]!=t))
                ids.append(w.witness_id)
            return Verification('negative',None,tuple(sorted(set(blockers))),tuple(ids))
        return Verification('unresolved',None,())

    def certificate_scores(self,observed,prior):
        observed,known,mismatch=self._state(observed)
        live=~mismatch.any(axis=1)
        missing=self.mask & ~known & live[:,None]
        cells=self.active_cells(observed)
        probs=prior.probabilities(cells,observed)
        lookup={v:i for i,v in enumerate(cells)}
        col=np.array([lookup.get(v,-1) for v in self.cells],dtype=np.int64)
        idx=self.index[missing]; labels=self.labels[missing]
        candidate=col[idx]>=0
        target=col[idx[candidate]]; types=labels[candidate]
        rejection=np.zeros(len(cells),dtype=float); incidence=np.zeros(len(cells),dtype=np.int64)
        np.add.at(rejection,target,1-probs[target,types]); np.add.at(incidence,target,1)
        one=missing.sum(axis=1)==1
        complete={v:set() for v in cells}
        for row in np.flatnonzero(one):
            j=np.flatnonzero(missing[row])[0]; v=self.cells[self.index[row,j]]
            if v in complete: complete[v].add(int(self.labels[row,j]))
        return {v:(float(sum(probs[i,t] for t in complete[v])),float(rejection[i]),int(incidence[i])) for i,v in enumerate(cells)}

    def world_answers(self,cells,worlds):
        columns={v:i for i,v in enumerate(cells)}
        mapping=np.array([columns[v] for v in self.cells],dtype=np.int64)
        worlds=np.asarray(worlds); answer=np.zeros(len(worlds),dtype=bool)
        for start in range(0,len(self.query.witnesses),2048):
            stop=min(start+2048,len(self.query.witnesses))
            valid=np.ones((stop-start,len(worlds)),dtype=bool)
            for j in range(self.index.shape[1]):
                idx=self.index[start:stop,j]; mask=idx>=0
                if mask.any(): valid[mask] &= worlds[:,mapping[idx[mask]]].T==self.labels[start:stop,j][mask,None]
            answer |= valid.any(axis=0)
        return answer

    def fixed_costs(self,observed,cells,samples,order):
        samples=np.asarray(samples); columns={v:i for i,v in enumerate(cells)}
        mapping=np.array([columns[v] for v in self.cells],dtype=np.int64)
        positions={v:i+1 for i,v in enumerate(order)}
        times=np.array([0 if observed.get(v) is not None else positions.get(v,np.inf) for v in self.cells])
        positive=np.full(len(samples),np.inf); negative=np.zeros(len(samples))
        for start in range(0,len(self.query.witnesses),2048):
            stop=min(start+2048,len(self.query.witnesses)); n=stop-start
            matched=np.ones((n,len(samples)),dtype=bool)
            completion=np.zeros(n); block=np.full((n,len(samples)),np.inf)
            for j in range(self.index.shape[1]):
                idx=self.index[start:stop,j]; mask=idx>=0
                if not mask.any(): continue
                selected=idx[mask]; actual=samples[:,mapping[selected]].T.copy()
                labels=self.labels[start:stop,j][mask,None]
                equals=actual==labels
                for k,i in enumerate(selected):
                    v=self.cells[i]
                    if observed.get(v) is not None: equals[k,:]=observed[v]==int(labels[k,0])
                when=times[selected]
                matched[mask] &= equals
                completion[mask]=np.maximum(completion[mask],when)
                block[mask]=np.minimum(block[mask],np.where(equals,np.inf,when[:,None]))
            positive=np.minimum(positive,np.where(matched,completion[:,None],np.inf).min(axis=0))
            negative=np.maximum(negative,block.max(axis=0))
        return np.minimum(positive,negative)


def compile_query(query):
    return factor_convergence(query) or CompiledQuery(query)


class ConvergenceQuery(CompiledQuery):
    def __init__(self,query,anchor,atype,left,ltype,right,rtype):
        super().__init__(query)
        self.anchor,self.atype=anchor,atype
        self.left,self.ltype=tuple(sorted(left)),ltype
        self.right,self.rtype=tuple(sorted(right)),rtype

    def _live_groups(self,observed):
        observed=_observations(observed)
        if observed.get(self.anchor) not in (None,self.atype): return observed,(),()
        left=tuple(v for v in self.left if observed.get(v) in (None,self.ltype))
        right=tuple(v for v in self.right if observed.get(v) in (None,self.rtype))
        return observed,left,right

    def incidence(self,observed):
        observed,left,right=self._live_groups(observed)
        left=set(left);right=set(right)
        result={v:0 for v in self.active_cells(observed)}
        for v in left:
            if v in result: result[v]+=len(right)-(v in right)
        for v in right:
            if v in result: result[v]+=len(left)-(v in left)
        if self.anchor in result: result[self.anchor]+=len(left)*len(right)-len(left & right)
        return result

    def active_cells(self,observed):
        observed,left,right=self._live_groups(observed)
        if not left or not right or (len(left)==len(right)==1 and left[0]==right[0]): return ()
        active={v for v in left if len(right)>1 or right[0]!=v}
        active.update(v for v in right if len(left)>1 or left[0]!=v)
        active.add(self.anchor)
        return tuple(sorted(active-observed.keys()))

    def evaluate(self,observed):
        if self.query.metadata.get('computationally_unsupported'): raise ValueError('computationally unsupported query cannot be verified')
        observed,left,right=self._live_groups(observed)
        if observed.get(self.anchor)==self.atype:
            goodleft={v for v in left if observed.get(v)==self.ltype}
            goodright={v for v in right if observed.get(v)==self.rtype}
            if goodleft and goodright:
                found=min((w for w in self.query.witnesses if w.vertices[0] in goodleft and w.vertices[2] in goodright),key=lambda w:w.witness_id,default=None)
                if found is not None: return Verification('positive',found.witness_id,tuple(sorted(set(found.requirements))))
        if left and right and not (len(left)==len(right)==1 and left[0]==right[0]):
            return Verification('unresolved',None,())
        return super().evaluate(observed)

    def world_answers(self,cells,worlds):
        columns={v:i for i,v in enumerate(cells)}
        return ((worlds[:,columns[self.anchor]]==self.atype)
                & np.any(worlds[:,[columns[v] for v in self.left]]==self.ltype,axis=1)
                & np.any(worlds[:,[columns[v] for v in self.right]]==self.rtype,axis=1))

    def fixed_costs(self,observed,cells,samples,order):
        samples=np.asarray(samples); columns={v:i for i,v in enumerate(cells)}
        positions={v:i+1 for i,v in enumerate(order)}
        def times(group,label):
            values=samples[:,[columns[v] for v in group]].T.copy()
            reveal=np.array([0. if observed.get(v) is not None else positions.get(v,np.inf) for v in group])[:,None]
            match=values==label
            for i,v in enumerate(group):
                if observed.get(v) is not None: match[i]=observed[v]==label
            return np.where(match,reveal,np.inf),np.where(match,np.inf,reveal)
        ap,ab=times((self.anchor,),self.atype)
        lp,lb=times(self.left,self.ltype); rp,rb=times(self.right,self.rtype)
        positive=np.maximum(ap[0],np.maximum(lp.min(axis=0),rp.min(axis=0)))
        def top_two(block,group):
            order=np.argsort(block,axis=0,kind='stable')
            best=order[-1]; top=np.take_along_axis(block,best[None,:],axis=0)[0]
            ids=np.asarray(group,dtype=object)[best]
            second=np.take_along_axis(block,order[-2][None,:],axis=0)[0] if len(group)>1 else np.zeros(block.shape[1])
            return top,second,ids
        l1,l2,li=top_two(lb,self.left); r1,r2,ri=top_two(rb,self.right)
        negative=np.where(li!=ri,np.minimum(l1,r1),np.maximum(np.minimum(l1,r2),np.minimum(l2,r1)))
        return np.minimum(positive,np.minimum(ab[0],negative))


def factor_convergence(query):
    """Factor only after proving the exact off-diagonal Cartesian witness family."""
    edges=tuple(sorted(tuple(e) for e in query.metadata.get('directed_edges',())))
    fixed=query.metadata.get('fixed_roles',{})
    anchor=fixed.get(1,fixed.get('1'))
    if edges!=((0,1),(2,1)) or anchor is None or not query.witnesses: return None
    left=set(); right=set(); pairs=set(); labels=None
    for w in query.witnesses:
        if len(w.vertices)!=3 or len(w.requirements)!=3 or w.vertices[1]!=anchor: return None
        a,_,c=w.vertices
        requirements=dict(w.requirements)
        if a==c or len(requirements)!=3 or set(requirements)!={a,anchor,c}: return None
        triple=(requirements[a],requirements[anchor],requirements[c])
        if labels is None: labels=triple
        if labels!=triple or labels[0]==labels[2]: return None
        left.add(a);right.add(c);pairs.add((a,c))
    if len(pairs)!=len(query.witnesses) or len(pairs)!=len(left)*len(right)-len(left & right): return None
    return ConvergenceQuery(query,anchor,labels[1],left,labels[0],right,labels[2])
