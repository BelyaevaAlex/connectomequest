#!/usr/bin/env python3
"""Train only on declared source; immutable content-addressed artifacts."""
import argparse
import json
from pathlib import Path
import importlib.metadata
import inspect
import time
import resource
import numpy as np
from connectomequest.annotation_v32.data import load_prepared_dataset
from connectomequest.annotation_v32.predictors import structural_features, fit_temperature, score_probabilities
from connectomequest.annotation_v32.training import source_split, train_mlp, train_graphsage
from scipy import sparse
from connectomequest.annotation_v32.runtime import ArtifactStore, ClaimBusy, content_key, hash_file

def cached_features(graph,root,public_identity,dependencies):
    identity={'public':public_identity,'implementation':inspect.getsource(structural_features),'dependencies':{k:dependencies[k] for k in ('numpy','scipy')}}
    key=content_key(identity); store=ArtifactStore(root)
    for attempt in range(60):
        if store.validate(key):
            with np.load(store.path(key)/'features.npz',allow_pickle=False) as z:
                if not np.array_equal(z['node_ids'],graph.node_ids): raise ValueError('feature node ID mismatch')
                return z['features'],key
        try:
            with store.claim(key) as claim:
                values=structural_features(graph)
                np.savez(claim.stage/'features.npz',node_ids=graph.node_ids,features=values)
                (claim.stage/'identity.json').write_text(json.dumps(identity,sort_keys=True))
                claim.complete()
                return values,key
        except (ClaimBusy,FileExistsError):
            time.sleep(1)
    raise TimeoutError('shared feature producer did not finish within 60 seconds')

def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--split-seed',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=40)
    parser.add_argument('--device',default='cpu',choices=('cpu','cuda'))
    parser.add_argument('--architecture',default='mlp',choices=('mlp','graphsage'))
    args=parser.parse_args()
    manifest=json.loads((args.source/'public/manifest.json').read_text())
    if not manifest.get('metadata',{}).get('dataset','').startswith('hemibrain:'): raise ValueError('training accepts declared hemibrain source only')
    root=Path(__file__).resolve().parents[1]
    paths=list((args.source/'public').glob('*'))+list((args.source/'private').glob('*'))
    code=[Path(__file__).resolve()]+[root/'src/connectomequest/annotation_v32'/name for name in ('__init__.py','data.py','predictors.py','priors.py','training.py','runtime.py')]
    identity={'schema':'v32-source-mlp-1','inputs':{str(p.relative_to(args.source)):hash_file(p) for p in paths if p.is_file()},'code':{str(p.relative_to(root)):hash_file(p) for p in code},'config':{'seed':args.seed,'epochs':args.epochs,'hidden':64,'batch_size':256,'device':args.device,'learning_rate':.01},'dependencies':{name:importlib.metadata.version(name) for name in ('numpy','scipy','torch')}}
    identity['config']['architecture']=args.architecture
    identity['config']['split_seed']=args.split_seed
    if args.architecture=='graphsage': identity['config']['batch_size']='full'
    key=content_key(identity); store=ArtifactStore(args.output)
    if store.validate(key): print(store.path(key)); return
    with store.claim(key) as claim:
        graph,private=load_prepared_dataset(args.source)
        train,cal,audit=source_split(private.labels,args.split_seed)
        features,feature_key=cached_features(graph,args.output/'_features',{k:v for k,v in identity['inputs'].items() if k.startswith('public/')},identity['dependencies'])
        def subset(cells):
            return features[np.searchsorted(graph.node_ids,cells)],np.array([private.labels[c] for c in cells])
        xt,yt=subset(train); xc,yc=subset(cal)
        if args.architecture=='mlp':
            model,report=train_mlp(xt,yt,xc,yc,label_count=len(graph.vocabulary),seed=args.seed,epochs=args.epochs,device=args.device)
        else:
            matrix=sparse.csr_matrix((graph.weights,graph.indices,graph.indptr),shape=(len(graph.node_ids),len(graph.node_ids)))
            labels=np.array([private.labels[int(c)] if private.labels[int(c)] is not None else -1 for c in graph.node_ids])
            model,report=train_graphsage(matrix,features,labels,np.searchsorted(graph.node_ids,train),np.searchsorted(graph.node_ids,cal),label_count=len(graph.vocabulary),seed=args.seed,epochs=args.epochs,device=args.device)
            embeddings=model.embeddings(matrix,features)
            xc=embeddings[np.searchsorted(graph.node_ids,cal)]
        temperature=fit_temperature(model.logits(xc),yc)
        xa,ya=subset(audit)
        if args.architecture=='graphsage': xa=embeddings[np.searchsorted(graph.node_ids,audit)]
        probabilities=score_probabilities(model.logits(xa),temperature)
        calibration_p=score_probabilities(model.logits(xc),temperature)
        report['architecture']=args.architecture
        report['calibration_nll']=float(-np.log(np.maximum(calibration_p[np.arange(len(yc)),yc],1e-300)).mean())
        report.update(temperature=temperature,audit_rows=len(audit),audit_accuracy=float((probabilities.argmax(1)==ya).mean()),audit_nll=float(-np.log(np.maximum(probabilities[np.arange(len(ya)),ya],1e-300)).mean()),train_class_coverage=len(set(yt)),vocabulary_size=len(graph.vocabulary),audit_other_fraction=float((ya==0).mean()),audit_predicted_other_mass=float(probabilities[:,0].mean()))
        model.save(claim.stage/'model.npz')
        report['total_wall_seconds']=time.monotonic()-started
        report['peak_process_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if args.device=='cuda':
            import torch
            report['peak_gpu_allocated_bytes']=torch.cuda.max_memory_allocated()
        report['feature_key']=feature_key
        for name,value in [('identity',identity),('report',report),('split',dict(train=train,calibration=cal,audit=audit))]:
            (claim.stage/(name+'.json')).write_text(json.dumps(value,sort_keys=True,indent=2))
        print(claim.complete())

if __name__=='__main__': main()
