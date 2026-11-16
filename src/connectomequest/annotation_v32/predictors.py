"""Public topology predictors, with no oracle access."""
import numpy as np
from collections import OrderedDict
from scipy import sparse
from scipy.special import logsumexp
from .priors import ProductPrior

def structural_features(graph):
    n=len(graph.node_ids)
    a=sparse.csr_matrix((graph.weights,graph.indices,graph.indptr),shape=(n,n),dtype=float)
    b=a.T.tocsr()
    outs=np.asarray(a.sum(1)).ravel(); ins=np.asarray(b.sum(1)).ravel()
    x=np.log1p(np.column_stack((np.diff(a.indptr),np.diff(b.indptr),outs,ins)))
    return np.column_stack((x,(a@x)/np.maximum(outs[:,None],1),(b@x)/np.maximum(ins[:,None],1))).astype('float32')

def score_probabilities(scores,temperature=1.):
    scores=np.asarray(scores,dtype=float)
    if scores.ndim!=2 or not scores.shape[1] or not np.isfinite(scores).all() or not np.isfinite(temperature) or temperature<=0:
        raise ValueError('finite scores and positive temperature required')
    z=scores/temperature
    return np.exp(z-logsumexp(z,axis=1,keepdims=True))

def fit_temperature(scores,labels,grid=(.001,.003,.01,.03,.1,.3,1.,3.,10.,30.,100.)):
    scores=np.asarray(scores); labels=np.asarray(labels,dtype=int)
    if not len(labels) or len(scores)!=len(labels) or np.any(labels<0) or np.any(labels>=scores.shape[1]): raise ValueError('invalid calibration set')
    losses=[-np.log(np.maximum(score_probabilities(scores,t)[np.arange(len(labels)),labels],1e-300)).mean() for t in grid]
    return float(grid[int(np.argmin(losses))])

def frequency_prior(labels,label_count,alpha=1.):
    if label_count<1 or alpha<=0: raise ValueError('positive smoothing required')
    counts=np.full(label_count,float(alpha))
    for value in labels.values():
        if value is not None:
            if not 0<=value<label_count: raise ValueError('invalid label')
            counts[value]+=1
    return ProductPrior({},label_count=label_count,default=counts)

class PredictionPrior:
    def __init__(self,node_ids,values):
        self.node_ids=np.array(node_ids,dtype=np.int64,copy=True)
        self.values=np.array(values,dtype=float,copy=True)
        if self.node_ids.ndim!=1 or np.any(np.diff(self.node_ids)<=0): raise ValueError('sorted unique IDs required')
        if self.values.ndim!=2 or len(self.values)!=len(self.node_ids) or not self.values.shape[1] or not np.isfinite(self.values).all() or np.any(self.values<0) or np.any(self.values.sum(1)<=0): raise ValueError('invalid probabilities')
        self.values/=self.values.sum(1,keepdims=True)
        self.label_count=self.values.shape[1]
        self.node_ids.flags.writeable=False; self.values.flags.writeable=False

    def _rows(self,cells):
        cells=np.asarray(cells,dtype=np.int64); rows=np.searchsorted(self.node_ids,cells)
        if np.any(rows>=len(self.node_ids)) or np.any(self.node_ids[rows]!=cells): raise KeyError('unknown cell')
        return rows

    def _condition(self,result,cells,observed):
        for i,cell in enumerate(cells):
            label=observed.get(int(cell))
            if label is not None:
                if not 0<=label<self.label_count: raise ValueError('invalid observed label')
                result[i]=0; result[i,label]=1
        return result

    def probabilities(self,cells,observed):
        return self._condition(self.values[self._rows(cells)].copy(),cells,observed)

    def sample(self,cells,observed,n_samples,rng):
        if n_samples<0: raise ValueError('negative samples')
        p=self.probabilities(cells,observed)
        result=np.empty((n_samples,len(cells)),dtype=np.int64)
        for j,row in enumerate(p): result[:,j]=rng.choice(self.label_count,size=n_samples,p=row)
        return result

    def save(self,path):
        with open(path,'xb') as stream: np.savez(stream,node_ids=self.node_ids,values=self.values)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z: return cls(z['node_ids'],z['values'])

class ModelPrior(PredictionPrior):
    """Lazy batched predictions; no full graph by vocabulary allocation."""
    def __init__(self,node_ids,features,model,temperature=1.,batch_size=256,cache_rows=1024):
        self.node_ids=np.array(node_ids,dtype=np.int64,copy=True)
        self.features=np.array(features,dtype=np.float32,copy=True)
        if np.any(np.diff(self.node_ids)<=0) or len(self.features)!=len(self.node_ids) or batch_size<1 or cache_rows<0: raise ValueError('invalid features')
        self.node_ids.flags.writeable=False; self.features.flags.writeable=False
        self.model=model; self.label_count=model.label_count; self.temperature=temperature; self.batch_size=batch_size
        self.cache_rows=cache_rows; self._cache=OrderedDict()

    def probabilities(self,cells,observed):
        rows=self._rows(cells)
        result=np.empty((len(rows),self.label_count),dtype=float)
        for start in range(0,len(rows),self.batch_size):
            part=rows[start:start+self.batch_size]
            missing=sorted(set(int(row) for row in part if int(row) not in self._cache))
            values=score_probabilities(self.model.logits(self.features[missing]),self.temperature) if missing else []
            new=dict(zip(missing,values))
            for offset,row in enumerate(part):
                row=int(row)
                value=self._cache[row] if row in self._cache else new[row]
                result[start+offset]=value
                self._cache[row]=value.copy(); self._cache[row].flags.writeable=False
                self._cache.move_to_end(row)
            while len(self._cache)>self.cache_rows: self._cache.popitem(last=False)
        return self._condition(result,cells,observed)

    def save(self,path):
        raise TypeError('save model separately from public features')

def fit_ntac(graph,training_labels,*,steps=3,features=None,max_workspace_bytes=64000000000):
    """Run unchanged author Ntac on explicitly supplied reference labels.

    For transfer, caller provides a disconnected source/evaluation union and
    source train labels only. Output is the author object, never calibrated
    probabilities. Labels are encoded as strings to preserve author '?' marker.
    """
    n=len(graph.node_ids); valid={int(k):int(v) for k,v in training_labels.items() if v is not None}
    if not valid or not 1<=steps<=12: raise ValueError('nonempty reference labels and 1..12 steps required')
    if not set(valid).issubset(set(map(int,graph.node_ids))) or any(v<0 or v>=len(graph.vocabulary) for v in valid.values()): raise ValueError('invalid reference labels')
    # Conservative dense similarities + duplicate embeddings + sparse work.
    estimate=8*n*(3*len(valid)+10*len(set(valid.values()))+64)+64*len(graph.indices)
    if estimate>max_workspace_bytes: raise MemoryError(f'NTAC estimated workspace {estimate} exceeds cap')
    from numba import set_num_threads
    set_num_threads(1)  # Author inverted-similarity kernel adds into shared entries.
    from ntac import Ntac
    matrix=sparse.csr_matrix((graph.weights,graph.indices,graph.indptr),shape=(n,n),dtype=float)
    labels=np.array([str(valid[int(cell)]) if int(cell) in valid else '?' for cell in graph.node_ids])
    features=structural_features(graph) if features is None else np.asarray(features)
    if features.shape!=(n,12) or not np.isfinite(features).all() or np.any(features<0): raise ValueError('NTAC requires finite nonnegative structural features')
    model=Ntac(matrix,labels=labels,lr=.3,topk=1,node_features=features,feature_weight=1.)
    for _ in range(steps): model.step()
    return model

def ntac_class_scores(model,rows,label_count):
    """Author get_topk_partition scoring: maximum reference similarity per class.

    Avoids the author's full N*K Python dictionary. Unrepresented classes get
    score zero, then source-only temperature calibration yields a proper simplex.
    """
    rows=np.asarray(rows,dtype=np.int64)
    result=np.zeros((len(rows),label_count),dtype=float)
    for encoded,internal in model.label_mapping.items():
        if encoded=='?': continue
        category=int(encoded)
        columns=np.flatnonzero(model.frozen_partition==internal)
        result[:,category]=model.similarity_matrix_to_frozen[np.ix_(rows,columns)].max(1)
    return result

def reference_union(source,evaluation):
    """Disconnected union with synthetic local IDs; external IDs returned by map.

    IDs from different datasets can collide, so offsets are explicit. No labels
    or anatomical cross-dataset edges are invented.
    """
    from .data import PublicAnnotationGraph
    if source.vocabulary!=evaluation.vocabulary: raise ValueError('vocabulary mismatch')
    def matrix(g): return sparse.csr_matrix((g.weights,g.indices,g.indptr),shape=(len(g.node_ids),len(g.node_ids)))
    a=sparse.block_diag((matrix(source),matrix(evaluation)),format='csr')
    ids=np.arange(1,a.shape[0]+1,dtype=np.int64)
    union=PublicAnnotationGraph(ids,a.indptr,a.indices,a.data,dict.fromkeys(map(int,ids)),source.vocabulary)
    return union,{int(c):i+1 for i,c in enumerate(source.node_ids)},{int(c):len(source.node_ids)+i+1 for i,c in enumerate(evaluation.node_ids)}

class NtacPrior(PredictionPrior):
    """Calibrated author similarities for external evaluation IDs.

    row_indices are zero-based rows in the union graph; source/evaluation ID
    namespaces remain separate. The fitted NTAC model is frozen for a session.
    """
    def __init__(self,node_ids,row_indices,model,label_count,temperature=1.,batch_size=256):
        self.node_ids=np.array(node_ids,dtype=np.int64,copy=True)
        self.row_indices=np.array(row_indices,dtype=np.int64,copy=True)
        if len(self.node_ids)!=len(self.row_indices) or np.any(np.diff(self.node_ids)<=0) or batch_size<1: raise ValueError('invalid NTAC row map')
        self.node_ids.flags.writeable=False; self.row_indices.flags.writeable=False
        self.model=model; self.label_count=label_count; self.temperature=temperature; self.batch_size=batch_size

    def probabilities(self,cells,observed):
        rows=self.row_indices[self._rows(cells)]
        result=np.empty((len(rows),self.label_count))
        for start in range(0,len(rows),self.batch_size):
            part=rows[start:start+self.batch_size]
            result[start:start+len(part)]=score_probabilities(ntac_class_scores(self.model,part,self.label_count),self.temperature)
        return self._condition(result,cells,observed)

    def save(self,path):
        raise TypeError('persist calibrated scores in bounded blocks, not the author model pickle')

class HistoricalHintPrior(PredictionPrior):
    """Common untuned historical-proposal mixture; proof facts win afterwards."""
    def __init__(self,base,provisional,alpha=.9):
        from types import MappingProxyType
        if not 0<=alpha<=1: raise ValueError('mixture outside [0,1]')
        self.base=base; self.provisional=MappingProxyType(dict(provisional)); self.alpha=alpha
        self.label_count=base.label_count
        if any(v is not None and not 0<=v<self.label_count for v in provisional.values()): raise ValueError('invalid provisional label')

    def probabilities(self,cells,observed):
        result=self.base.probabilities(cells,observed).copy()
        for i,cell in enumerate(cells):
            hint=self.provisional.get(int(cell))
            if hint is not None:
                result[i]*=1-self.alpha; result[i,hint]+=self.alpha
        return self._condition(result,cells,observed)

    def save(self,path):
        raise TypeError('save base artifact and public hint identity separately')

def export_predictions(prior,node_ids,root,identity,*,batch_size=256,max_entries=1000000000):
    """Atomic float32 disk artifact, bounded compute blocks, no observed facts.

    identity must describe model/source/features/vocabulary/dependency hashes
    and calibration. Caller freezes these identities before evaluation.
    """
    import hashlib,json,time
    from .runtime import ArtifactStore,content_key
    ids=np.asarray(node_ids,dtype=np.int64)
    if ids.ndim!=1 or np.any(np.diff(ids)<=0) or batch_size<1: raise ValueError('invalid prediction rows')
    if len(ids)*prior.label_count>max_entries: raise MemoryError('disk prediction entry cap exceeded')
    if batch_size*prior.label_count>2000000: raise MemoryError('prediction block cap exceeded')
    started=time.monotonic()
    descriptor={'schema':'v32-probability-f32-1','identity':identity,'node_ids_sha256':hashlib.sha256(ids.tobytes()).hexdigest(),'label_count':prior.label_count}
    key=content_key(descriptor); store=ArtifactStore(root)
    if store.validate(key): return key
    with store.claim(key) as claim:
        values=np.lib.format.open_memmap(claim.stage/'probabilities.npy',mode='w+',dtype=np.float32,shape=(len(ids),prior.label_count))
        for start in range(0,len(ids),batch_size):
            batch=ids[start:start+batch_size]
            block=prior.probabilities(batch,{})
            if not np.isfinite(block).all() or np.any(block<0) or np.any(block.sum(1)<=0): raise ValueError('invalid exported probabilities')
            values[start:start+len(batch)]=block/block.sum(1,keepdims=True)
            if time.monotonic()-started>5400: raise TimeoutError('prediction export exceeded 90 minute cap')
        values.flush(); del values
        np.save(claim.stage/'node_ids.npy',ids,allow_pickle=False)
        (claim.stage/'identity.json').write_text(json.dumps(descriptor,sort_keys=True))
        (claim.stage/'statistics.json').write_text(json.dumps({'wall_seconds':time.monotonic()-started,'probability_bytes':len(ids)*prior.label_count*4,'batch_size':batch_size}))
        claim.complete()
    return key

def load_prediction_artifact(root,key):
    from .runtime import ArtifactStore
    store=ArtifactStore(root)
    if not store.validate(key): raise ValueError('invalid prediction artifact')
    prior=object.__new__(PredictionPrior)
    prior.node_ids=np.load(store.path(key)/'node_ids.npy',mmap_mode='r',allow_pickle=False)
    prior.values=np.load(store.path(key)/'probabilities.npy',mmap_mode='r',allow_pickle=False)
    prior.label_count=prior.values.shape[1]
    return prior
