#!/usr/bin/env python3
"""Version old public hints without parsing current reference annotations."""
import argparse,csv,json,shutil,tempfile
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from connectomequest.annotation_v32.data import load_public_graph,ontology_type,OTHER
from connectomequest.annotation_v32.runtime import hash_file,content_key
PIN='ebd66db2596fcc39c6950fb54ea3efa00f7fe8a0'
AUTHOR={'commit':PIN,'notebook_url':f'https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{PIN}/code/annotation_counts.ipynb','notebook_sha256':'4c1eca4368c0e997344f12557bfd10ce25117f823725aadf411a582cabb05e79','schema_url':f'https://raw.githubusercontent.com/flyconnectome/flywire_annotations/{PIN}/supplemental_files/Supplemental_files_columns.md','schema_sha256':'5bed86549dd6d2837397c2a451db7606718f5d5be8a8a7a7a49219aade1c9802'}

def public_hashes(path):
    path=Path(path)
    return {str(p.relative_to(path)):hash_file(p) for p in sorted((path/'public').iterdir()) if p.is_file()}

def choose_hint(cell_type,hemibrain_type,vocabulary):
    if cell_type is not None and cell_type!='':return cell_type,'cell_type'
    candidate=(hemibrain_type or '').strip()
    if not candidate:return None,'missing'
    if any(x in candidate for x in (',',';','|')):return None,'composite_skipped'
    mapped=ontology_type(candidate,'flywire:v2.1.0')
    if mapped==OTHER or mapped not in vocabulary:return None,'unmapped_skipped'
    return candidate,'hemibrain_type'

def public_equivalence(parent,derived):
    parent,derived=Path(parent),Path(derived);a,b=load_public_graph(parent),load_public_graph(derived)
    for name in ('node_ids','indptr','indices','weights'):
        if not np.array_equal(getattr(a,name),getattr(b,name)):raise ValueError('topology/row identity differs: '+name)
    if a.vocabulary!=b.vocabulary or a.metadata.get('ontology')!=b.metadata.get('ontology'):raise ValueError('vocabulary/ontology differs')
    if hash_file(parent/'public/topology.npz')!=hash_file(derived/'public/topology.npz'):raise ValueError('topology bytes differ')
    return {'equivalent':True,'topology_sha256':hash_file(parent/'public/topology.npz'),'vocabulary_sha256':content_key(list(a.vocabulary)),'scope':'Exact node order, weighted CSR and ontology; hints intentionally differ.'}

def correct(parent,old_annotations,output):
    parent,old_annotations,output=map(Path,(parent,old_annotations,output))
    if output.exists():raise FileExistsError(output)
    before=public_hashes(parent);graph=load_public_graph(parent);manifest=json.loads((parent/'public/manifest.json').read_text())
    if graph.metadata.get('dataset')!='flywire783' or graph.metadata.get('hint_release')!='flywire:v2.1.0':raise ValueError('unexpected old release metadata')
    old_sha=hash_file(old_annotations)
    if graph.metadata.get('files',{}).get(old_annotations.name)!=old_sha:raise ValueError('old release hash mismatch')
    with old_annotations.open(newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
    roots=[int(r['root_id']) for r in rows];sv=[r['supervoxel_id'] for r in rows]
    if len(set(roots))!=len(roots) or len(set(sv))!=len(sv):raise ValueError('duplicate old IDs')
    old=dict(zip(roots,rows));ids=list(map(int,graph.node_ids));lookup={n:i for i,n in enumerate(graph.vocabulary)}
    if not set(ids)<=old.keys():raise ValueError('old release lacks frozen IDs')
    counts=Counter();records=[]
    for cell in ids:
        row=old[cell]
        if graph.old_label_names[cell]!=(row['cell_type'] or None):raise ValueError('parent hints differ from old cell_type')
        name,origin=choose_hint(row['cell_type'],row['hemibrain_type'],lookup)
        code=graph.provisional[cell] if origin=='cell_type' else (None if name is None else lookup[ontology_type(name,'flywire:v2.1.0')])
        records.append((code,name,origin));counts[origin]+=1
    reference_sha=hash_file(parent/'private/labels.parquet')
    lineage={'schema':'v32-old-hint-correction-1','parent_public_inputs':before,'old_annotation_sha256':old_sha,'adapter_sha256':hash_file(__file__),'author_recipe':AUTHOR,'policy':'Preserve cell_type; only a known single hemibrain_type fills an empty cell_type. Composite and unmapped exclusions are our conservative restriction.','legacy_transition_interpretation':'Prior counts describe raw cell_type-field transitions, not newly available biological annotations.','reference_sha256_unchanged':reference_sha,'counts':dict(counts)}
    output.parent.mkdir(parents=True,exist_ok=True);stage=Path(tempfile.mkdtemp(prefix=output.name+'.stage.',dir=output.parent))
    try:
        pub=stage/'public';pri=stage/'private';pub.mkdir();pri.mkdir()
        shutil.copyfile(parent/'public/topology.npz',pub/'topology.npz')
        pq.write_table(pa.table({'cell_id':pa.array(ids,type=pa.int64()),'provisional':pa.array([r[0] for r in records],type=pa.int32()),'old_label_name':pa.array([r[1] for r in records],type=pa.string()),'hint_source_column':pa.array([r[2] for r in records],type=pa.string())}),pub/'nodes.parquet')
        manifest['metadata']={**manifest['metadata'],'historical_hint_correction':lineage};manifest['files']={n:hash_file(pub/n) for n in ('nodes.parquet','topology.npz')}
        (pub/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2))
        shutil.copyfile(parent/'private/labels.parquet',pri/'labels.parquet')
        if hash_file(pri/'labels.parquet')!=reference_sha:raise ValueError('reference bytes changed')
        (pri/'manifest.json').write_text(json.dumps({'schema':'annotation-v32-private-v1','metadata':{'oracle_version':graph.metadata.get('oracle_version'),'parent_private_manifest_sha256':hash_file(parent/'private/manifest.json'),'historical_transition_counts':'Not recomputed; legacy cell_type-only transitions do not describe corrected hints.'},'files':{'labels.parquet':reference_sha}},sort_keys=True,indent=2))
        public_equivalence(parent,stage)
        if public_hashes(parent)!=before:raise ValueError('parent changed during correction')
        stage.rename(output)
    except BaseException:shutil.rmtree(stage,ignore_errors=True);raise
    return lineage

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True);p.add_argument('--old-annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(correct(a.parent,a.old_annotations,a.output),sort_keys=True))
if __name__=='__main__':main()
