"""Frozen public held-out cohorts and bounded, journaled evaluation."""
from __future__ import annotations
import hashlib
from contextlib import contextmanager
import numpy as np
from .queries import source_roster
from .data import ontology_type

ANCHOR_GROUPS=(('olfactory','LHAV7b1'),('olfactory','LHAV2a3'),
               ('mushroom','MBON02'),('mushroom','MBON01'),('mushroom','MBON09'),('CX','EPG'))
SESSION_FAMILIES=('CX','olfactory','mushroom')
VALIDATION_ARMS=('nn_strict_fixed','nn_skip_fixed','nn_witness_fixed','nn_adaptive','nn_ec2','nn_one_step_ablation',
                 'ntac_skip_fixed','ntac_adaptive','frequency_skip_fixed','frequency_adaptive','uniform_adaptive')
MANDATORY_ARMS=tuple(arm for arm in VALIDATION_ARMS if arm!='nn_one_step_ablation')
FIXED_ARMS=('nn_strict_fixed','nn_skip_fixed','nn_witness_fixed','ntac_skip_fixed','frequency_skip_fixed')

def anchor_label(name):
    return ontology_type(name,'hemibrain:v1.2.1')

def fit_anchor_prototypes(source,training_labels,features):
    """Only declared source training labels enter task proposal prototypes."""
    x=np.asarray(features,dtype=np.float32)
    cells=np.array(sorted(k for k,v in training_labels.items() if v is not None),dtype=np.int64)
    rows=np.searchsorted(source.node_ids,cells)
    if x.shape!=(len(source.node_ids),12) or not np.isfinite(x).all() or not len(cells) or np.any(rows>=len(source.node_ids)) or np.any(source.node_ids[rows]!=cells):
        raise ValueError('invalid source training features/IDs')
    mean=x[rows].mean(0); scale=np.maximum(x[rows].std(0),1e-6)
    normalized=(x[rows]-mean)/scale
    result={'mean':mean.tolist(),'scale':scale.tolist(),'groups':{},
            'training_ids_sha256':hashlib.sha256(cells.tobytes()).hexdigest()}
    for family,name in ANCHOR_GROUPS:
        code=source.label_code(anchor_label(name))
        indices=[i for i,cell in enumerate(cells) if training_labels[int(cell)]==code]
        result['groups'][family+'/'+name]={'label_code':code,'count':len(indices),
            'prototype':normalized[indices].mean(0).tolist() if indices else None}
    return result

def select_heldout_anchors(graph,prototypes,dataset,features):
    if dataset not in ('malecns','flywire'): raise ValueError('held-out dataset must be malecns or flywire')
    x=np.asarray(features,dtype=np.float32)
    if x.shape!=(len(graph.node_ids),12) or not np.isfinite(x).all(): raise ValueError('invalid public features')
    x=(x-np.asarray(prototypes['mean'],dtype=np.float32))/np.asarray(prototypes['scale'],dtype=np.float32)
    used=set(); groups=[]
    salt='cq-v32-validation-roster-v1' if dataset=='malecns' else 'cq-v32-target-roster-v1'
    for family,name in ANCHOR_GROUPS:
        group=family+'/'+name; record=prototypes['groups'][group]
        def ranked(cells):
            return sorted((int(c) for c in cells if int(c) not in used),key=lambda c:(hashlib.sha256(f'{salt}:{dataset}:{group}:{c}'.encode()).digest(),c))
        if record['prototype'] is None: structural=[]
        else:
            distance=np.sum((x-np.asarray(record['prototype'],dtype=np.float32))**2,axis=1)
            pool=np.lexsort((graph.node_ids,distance))[:64]
            structural=graph.node_ids[pool]
        anchors=[]; reasons=[]; strata=[]
        modes=(('structural',4),) if dataset=='malecns' else (('historical',2),('structural',2))
        for mode,count in modes:
            eligible=[cell for cell,label in graph.provisional.items() if label==graph.label_code(anchor_label(name))] if mode=='historical' else structural
            selected=ranked(eligible)[:count]
            for i in range(count):
                cell=selected[i] if i<len(selected) else None
                reason=None if cell is not None else ('source_prototype_missing' if mode=='structural' and record['prototype'] is None else 'missing_'+mode+'_anchor')
                anchors.append(cell); reasons.append(reason); strata.append(mode)
                if cell is not None: used.add(cell)
        groups.append({'group':group,'family':family,'anchor_type':name,'anchor_role':0 if family=='CX' else 1,
                       'anchors':anchors,'reasons':reasons,'strata':strata})
    return groups

def session_layout(selected,dataset='heldout'):
    lookup={(g['family'],anchor_label(g['anchor_type'])):g for g in selected}; sessions=[]
    for replicate in range(4):
        for family in SESSION_FAMILIES:
            slots=[]
            for template in source_roster():
                if template.family!=family: continue
                role=0 if family=='CX' else 1
                group=lookup[family,template.role_types[role]]
                slots.append({'query_id':f'{dataset}:r{replicate}:{template.name}','template':template.name,
                              'anchor':group['anchors'][replicate],'anchor_role':role,
                              'stratum':group['strata'][replicate],'unsupported':group['reasons'][replicate]})
            sessions.append({'session_id':f'{dataset}:r{replicate}:{family}','family':family,'replicate':replicate,
                             'stratum':slots[0]['stratum'],'slots':slots})
    if len(sessions)!=12 or sum(len(s['slots']) for s in sessions)!=32: raise ValueError('unexpected common roster')
    return sessions

def enumerate_slot(graph,slot,*,max_witnesses=2000000,max_search_states=5000000):
    from .queries import enumerate_typed_query
    if slot.get('unsupported'): return None
    template=next(t for t in source_roster() if t.name==slot['template'])
    return enumerate_typed_query(graph,slot['query_id'],template.role_types,template.directed_edges,
        fixed_roles={slot['anchor_role']:slot['anchor']},max_witnesses=max_witnesses,max_search_states=max_search_states)

def freeze_cohort(graph,prototypes,features,dataset,output,identity):
    """Public-only freeze; no evaluator or private target labels accepted."""
    import copy,gzip,json,time,resource
    from .runtime import ArtifactStore,content_key
    from .core import query_to_dict
    from .queries import intent_audit
    started=time.process_time(); wall=time.monotonic()
    selection=select_heldout_anchors(graph,prototypes,dataset,features)
    sessions=session_layout(selection,dataset)
    descriptor={'schema':'v32-heldout-cohort-1','dataset':dataset,'identity':identity,
        'prototypes':prototypes,'selection':selection,'sessions':copy.deepcopy(sessions),
        'config':{'budget':256,'max_witnesses':2000000,'max_search_states':5000000,
                  'model_entries':50000000,'samples':128,'historical_hint_alpha':.9,'arms':VALIDATION_ARMS},'intent_audit':intent_audit()}
    key=content_key(descriptor); store=ArtifactStore(output)
    if store.validate(key): return key
    with store.claim(key) as claim:
        (claim.stage/'selection.json').write_text(json.dumps(selection,sort_keys=True,indent=2))
        (claim.stage/'queries').mkdir()
        candidate_cells={}; count=0
        for session in sessions:
            for slot in session['slots']:
                if slot.get('unsupported'):
                    candidate_cells[slot['query_id']]=[]; continue
                query=enumerate_slot(graph,slot)
                filename=f'queries/{count:03d}.json.gz'; count+=1
                with gzip.open(claim.stage/filename,'wt',encoding='utf8') as stream:
                    json.dump(query_to_dict(query),stream,separators=(',',':'))
                slot['query_file']=filename
                slot['structurally_empty']=not query.witnesses and query.metadata.get('enumeration_complete',False)
                if query.metadata.get('computationally_unsupported'):
                    slot['unsupported']='enumeration_limit'; candidate_cells[slot['query_id']]=None
                else: candidate_cells[slot['query_id']]=list(query.cells)
        frozen={'key':key,'descriptor':descriptor,'sessions':sessions,
                'overlap_components':overlap_components(sessions,candidate_cells,graph.node_ids),
                'usable_overlap_components':usable_components(sessions,overlap_components(sessions,candidate_cells,graph.node_ids)),
                'candidate_cells':candidate_cells,'preparation_cpu_seconds':time.process_time()-started,
                'preparation_wall_seconds':time.monotonic()-wall,
                'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
        (claim.stage/'freeze.json').write_text(json.dumps(frozen,sort_keys=True))
        claim.complete()
    return key

def load_cohort(root,key):
    import json
    from .runtime import ArtifactStore,content_key
    store=ArtifactStore(root)
    if not store.validate(key): raise ValueError('invalid frozen cohort artifact')
    frozen=json.loads((store.path(key)/'freeze.json').read_text())
    if content_key(frozen['descriptor'])!=key or frozen['key']!=key: raise ValueError('cohort identity mismatch')
    return frozen

def load_session_queries(root,key,session):
    import gzip,json
    from pathlib import Path
    from .core import query_from_dict
    from .runtime import ArtifactStore
    path=ArtifactStore(root).path(key); result={}
    for slot in session['slots']:
        if slot.get('unsupported'): continue
        relative=Path(slot['query_file'])
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('query file outside artifact')
        with gzip.open(path/relative,'rt',encoding='utf8') as stream: query=query_from_dict(json.load(stream))
        if query.query_id!=slot['query_id'] or query.metadata.get('computationally_unsupported') or not query.metadata.get('enumeration_complete'):
            raise ValueError('unsupported or mismatched query artifact')
        result[query.query_id]=query
    return result

def cached_predictions(node_ids,label_count,factory,root,identity):
    """Heavy feature/model construction happens only on a cache miss."""
    from .predictors import export_predictions
    class LazyPrior:
        def __init__(self): self.label_count=label_count; self.base=None
        def probabilities(self,cells,observed):
            if self.base is None: self.base=factory()
            return self.base.probabilities(cells,observed)
    return export_predictions(LazyPrior(),node_ids,root,identity)

class WitnessFixedPolicy:
    """Freeze shared exact witness priority once; skip only logically dead cells."""
    def __init__(self,*,max_probability_entries=50000000):
        self.cap=max_probability_entries
    def begin_query(self,query,observed):
        self.order=None
        self.diagnostics={'fallback_count':0,'fallback_reason':None,'order_name':'witness_first'}
    def choose_next(self,query,observed,prior,rng):
        from collections import Counter
        from .core import active_cells,live_witnesses
        from .acquisition import witness_first_order
        from .priors import ImpossibleObservation
        active=set(active_cells(query,observed))
        if self.order is None:
            reason=None
            if len(active)*prior.label_count>self.cap: reason='probability_entry_budget'
            else:
                try: self.order=witness_first_order(query,observed,prior)
                except ImpossibleObservation: reason='zero_model_mass'
            if reason:
                counts=Counter(cell for witness in live_witnesses(query,observed) for cell,_ in witness.requirements if cell in active)
                self.order=tuple(sorted(active,key=lambda cell:(-counts[cell],cell)))
                self.diagnostics.update(fallback_count=1,fallback_reason=reason)
        return next((cell for cell in self.order if cell in active),None)

def make_arms(base_priors,provisional,*,max_entries=50000000,n_samples=128):
    from .predictors import HistoricalHintPrior
    from .priors import UniformProductPrior
    from .acquisition import FixedOrderPolicy,CertificateGreedy,SampledEC2Policy,AdaptiveWitnessPolicy
    k=base_priors['nn'].label_count
    if any(base_priors[name].label_count!=k for name in ('nn','ntac','frequency')): raise ValueError('prior vocabulary mismatch')
    bases={**base_priors,'uniform':UniformProductPrior(k)}
    mixed={name:HistoricalHintPrior(base,provisional,.9) for name,base in bases.items()}
    result={}
    for arm in VALIDATION_ARMS:
        name=arm.split('_')[0]
        if arm=='nn_witness_fixed':
            factory=lambda:WitnessFixedPolicy(max_probability_entries=max_entries)
        elif arm.endswith('strict_fixed'):
            factory=lambda:FixedOrderPolicy(n_samples=n_samples,logical_skip=False,max_sample_entries=max_entries)
        elif arm.endswith('skip_fixed'):
            factory=lambda:FixedOrderPolicy(n_samples=n_samples,logical_skip=True,max_sample_entries=max_entries)
        elif arm.endswith('ec2'):
            factory=lambda:SampledEC2Policy(n_samples=n_samples,max_sample_entries=max_entries)
        elif arm=='nn_one_step_ablation':
            factory=lambda:CertificateGreedy(max_probability_entries=max_entries)
        else:
            factory=lambda:AdaptiveWitnessPolicy(max_probability_entries=max_entries)
        result[arm]=(factory,mixed[name])
    return result

RESOURCE_FALLBACKS=frozenset(('probability_entry_budget','sample_entry_budget'))
ADMISSION_CHECKS=('code_data_frozen','resource_forecast_fits','remaining_budget_available','zero_invalid_certificates')

def authorize_evaluation(gate,cohort_key,predictor_key,dataset,seed):
    if gate.get('allow_evaluation') is not True or gate.get('dataset')!=dataset or gate.get('cohort_key')!=cohort_key:
        raise ValueError('sealed evaluation not authorized')
    if predictor_key not in gate.get('predictor_bundle_keys',[]): raise ValueError('predictor bundle not admitted')
    if any(gate.get('admission',{}).get(key) is not True for key in ADMISSION_CHECKS):
        raise ValueError('external admission checks incomplete')
    if dataset=='malecns':
        if seed!=0: raise ValueError('validation permits seed zero only')
    elif dataset=='flywire':
        expected={'descriptive_seed0':[0],'superiority_confirmation':[0,1,2]}.get(gate.get('branch'))
        if expected is None or gate.get('allowed_training_seeds')!=expected or seed not in expected:
            raise ValueError('target branch/seed not authorized')
    else: raise ValueError('unknown evaluation dataset')

def usable_components(sessions,components):
    informative={s['session_id'] for s in sessions if any(not slot.get('unsupported') for slot in s['slots'])}
    return [component for component in components if informative.intersection(component)]

def pooled_resolution(rows):
    planned=sum(row['metrics']['planned_slots'] for row in rows)
    certified=sum(row['metrics']['certified_slots'] for row in rows)
    if planned<=0 or not 0<=certified<=planned: raise ValueError('invalid pooled slot counts')
    return certified/planned

def validation_gate(results,admission):
    """Frozen continuation rule; favorable cost cannot authorize resource gates."""
    reasons=[key for key in ADMISSION_CHECKS if admission.get(key) is not True]
    grouped={arm:[r for r in results if r['arm']==arm] for arm in VALIDATION_ARMS}
    domains=[{r['session_id'] for r in rows} for rows in grouped.values()]
    complete=(len(results)==12*len(VALIDATION_ARMS) and all(len(rows)==12 and len(domain)==12 for rows,domain in zip(grouped.values(),domains))
              and all(domain==domains[0] for domain in domains) and all(r['dataset']=='malecns' and r.get('seed',0)==0 for r in results))
    if not complete:
        return {'operational_pass':False,'superiority_pass':False,'branch':'stop','allowed_training_seeds':[],
                'reasons':reasons+['incomplete_validation_matrix']}
    for row in results:
        values=(row['metrics']['C_B'],row['metrics']['certificate_fraction'],row['process_cpu_seconds'])
        if not np.isfinite(values).all() or not 0<=values[0]<=256 or not 0<=values[1]<=1 or values[2]<0:
            raise ValueError('invalid validation metrics')
    for arm in MANDATORY_ARMS:
        rows=grouped[arm]; slots=[slot for row in rows for slot in row['slot_outcomes']]
        if len(slots)!=32 or sum(s['comparator_supported'] for s in slots)<.75*32: reasons.append(arm+':overall_coverage')
        for family,expected in (('CX',12),('olfactory',8),('mushroom',12)):
            selected=[slot for row in rows if row['family']==family for slot in row['slot_outcomes']]
            if len(selected)!=expected or sum(s['comparator_supported'] for s in selected)<.5*expected:
                reasons.append(arm+':'+family+':coverage')
    completed={row['session_id'] for row in results if row['arm'] in MANDATORY_ARMS and row['metrics']['fully_certified']}
    if len(completed)<6: reasons.append('endpoint_feasibility')
    fixed=choose_strong_fixed(results)
    def mean(arm,key): return float(np.mean([r['metrics'][key] for r in grouped[arm]]))
    superiority=True
    for comparator in (fixed,'ntac_adaptive'):
        baseline=mean(comparator,'C_B')
        superiority &= baseline>0 and mean('nn_adaptive','C_B')<=.9*baseline
        superiority &= pooled_resolution(grouped['nn_adaptive'])>=pooled_resolution(grouped[comparator])
        for family in SESSION_FAMILIES:
            count=lambda arm:sum(r['metrics']['fully_certified'] for r in grouped[arm] if r['family']==family)
            superiority &= count('nn_adaptive')>=count(comparator)-1
    operational=not reasons; superiority=bool(superiority and operational)
    branch='stop' if not operational else ('superiority_confirmation' if superiority else 'descriptive_seed0')
    return {'operational_pass':operational,'superiority_pass':superiority,'branch':branch,'strong_fixed':fixed,
            'allowed_training_seeds':[] if not operational else ([0,1,2] if superiority else [0]),'reasons':reasons,
            'comparison_means':{arm:mean(arm,'C_B') for arm in ('nn_adaptive',fixed,'ntac_adaptive')}}

def slot_outcomes(result,slots):
    """Keep proof resolution separate from intended-comparator coverage."""
    rows={r['query_id']:r for r in result['queries']}; output=[]
    for slot in slots:
        row=rows.get(slot['query_id'])
        fallbacks=result.get('fallbacks_by_query',{}).get(slot['query_id'],[])
        if slot.get('unsupported'): reason=slot['unsupported']
        elif row is None: reason='not_exposed_resource_stop' if result.get('resource_stopped') else 'missing_result'
        elif row.get('stop_reason')=='resource_stop': reason='resource_stop'
        elif RESOURCE_FALLBACKS.intersection(fallbacks): reason='model_workspace_fallback'
        else: reason=None
        output.append({**slot,'status':row['status'] if row else 'unresolved',
                       'certificate_valid':bool(row and row.get('certificate_valid')),
                       'comparator_supported':reason is None,'coverage_reason':reason,'fallback_reasons':fallbacks,
                       'reads':row.get('reads',0) if row else 0})
    return output

def summarize_session(result,slots,budget=256):
    if budget!=256: raise ValueError('held-out endpoint is frozen at B=256')
    rows={row['query_id']:row for row in result['queries']}
    if len(rows)!=len(result['queries']) or not set(rows)<=set(slot['query_id'] for slot in slots): raise ValueError('invalid result query IDs')
    resolved=sum(bool(rows.get(slot['query_id'],{}).get('certificate_valid')) and not slot.get('unsupported') for slot in slots)
    cost=result['cost']
    if not 0<=cost<=budget: raise ValueError('invalid actual read count')
    return {'C_B':cost if resolved==len(slots) and slots else budget,'actual_reads':cost,
            'planned_slots':len(slots),'certified_slots':resolved,'certificate_fraction':resolved/len(slots) if slots else 0.,
            'fully_certified':bool(slots) and resolved==len(slots),
            'positive':sum(r.get('status')=='positive' and r.get('certificate_valid',False) for r in rows.values()),
            'negative':sum(r.get('status')=='negative' and r.get('certificate_valid',False) for r in rows.values()),
            'unresolved':len(slots)-resolved,'unsupported':sum(bool(s.get('unsupported')) for s in slots)}

def overlap_components(sessions,candidate_cells,public_node_ids):
    """Purchasable witnesses plus anchors; None means unknown full universe."""
    parent=list(range(len(sessions)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    owner={}
    for i,session in enumerate(sessions):
        cells=set()
        for slot in session['slots']:
            selected=candidate_cells.get(slot['query_id'],())
            cells.update(public_node_ids if selected is None else selected)
            if slot.get('anchor') is not None: cells.add(slot['anchor'])
        for cell in cells:
            if cell in owner: parent[find(i)]=find(owner[cell])
            else: owner[cell]=i
    groups={}
    for i,session in enumerate(sessions): groups.setdefault(find(i),[]).append(session['session_id'])
    return sorted((sorted(group) for group in groups.values()),key=lambda group:group[0])

def choose_strong_fixed(results):
    rows=[row for row in results if row['arm'] in FIXED_ARMS]
    if any(row['dataset']!='malecns' for row in rows): raise ValueError('fixed comparator selection requires validation only')
    grouped={arm:[r for r in rows if r['arm']==arm] for arm in FIXED_ARMS}
    domains=[{r['session_id'] for r in group} for group in grouped.values()]
    if any(len(group)!=12 or len(domain)!=12 for group,domain in zip(grouped.values(),domains)) or any(domain!=domains[0] for domain in domains):
        raise ValueError('complete matched twelve-session fixed matrix required')
    def score(arm):
        group=grouped[arm]
        return (np.mean([r['metrics']['C_B'] for r in group]),-pooled_resolution(group),
                np.mean([r['process_cpu_seconds'] for r in group]),arm)
    return min(FIXED_ARMS,key=score)

class ResourceStop(RuntimeError):
    pass

def certificate_query_identity(query):
    """Streaming identity; callers cache it once per immutable public query."""
    import json
    digest=hashlib.sha256()
    def add(value):
        digest.update(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()); digest.update(b'\n')
    add({'query_id':query.query_id,'metadata':dict(query.metadata)})
    for witness in query.witnesses:
        add((witness.witness_id,witness.requirements,witness.vertices))
    return {'query_id':query.query_id,'query_sha256':digest.hexdigest(),'witness_count':len(query.witnesses)}

def compact_certificates(result,queries,identities,witness_ids=None):
    """Replace redundant negative witness-ID lists, retaining every proof fact."""
    for row in result['queries']:
        certificate=row.get('certificate')
        if certificate is None: continue
        if certificate['status']=='negative':
            query=queries[row['query_id']]
            expected=tuple(sorted(w.witness_id for w in query.witnesses)) if witness_ids is None else witness_ids[row['query_id']]
            if tuple(certificate['blocked_witnesses'])!=expected:
                raise ValueError('negative certificate does not list the immutable query witnesses')
            certificate.pop('blocked_witnesses')
            certificate['format']='cq-v32-negative-certificate-1'
            certificate['query']=identities[row['query_id']]
    return result

def replay_compact_certificate(query,payload,*,identity=None,witness_ids=None):
    from .core import Verification,replay_certificate
    if payload.get('format')!='cq-v32-negative-certificate-1' or payload.get('status')!='negative': return False
    expected=certificate_query_identity(query) if identity is None else identity
    if payload.get('query')!=expected: return False
    ids=tuple(sorted(w.witness_id for w in query.witnesses)) if witness_ids is None else witness_ids
    try:
        certificate=Verification(payload['status'],payload['witness_id'],tuple(tuple(f) for f in payload['facts']),ids)
        return replay_certificate(query,certificate)
    except (KeyError,TypeError,ValueError): return False

def audit_session_result(result,journal_path,session,queries,reference,certificate_contexts=None):
    """Recompute payment provenance and proof validity from the durable log."""
    import json
    from .core import Verification,replay_certificate
    if result.get('global_budget_stop'):
        if result['cost']!=0 or result['queries'] or result['events'] or not result.get('resource_stopped'):
            raise ValueError('invalid global-budget placeholder')
        return summarize_session(result,session['slots'])
    expected=[slot['query_id'] for slot in session['slots'] if not slot.get('unsupported')]
    paid={}; attempted=set(); pending=None; current=None; beginnings=[]; endings={}; reads={}; fallbacks={}; finished=False
    with open(journal_path) as stream:
        for line in stream:
            if not line.endswith('\n'): raise ValueError('incomplete durable journal')
            event=json.loads(line)
            if finished: raise ValueError('journal continues after finish')
            kind=event['kind']
            if kind=='begin':
                qid=event['query_id']
                if pending is not None or len(beginnings)>=len(expected) or qid!=expected[len(beginnings)]: raise ValueError('invalid query order')
                if event['attempt_count']!=len(attempted) or {int(k):v for k,v in event['observed'].items()}!=paid: raise ValueError('invalid paid cache snapshot')
                if current is not None: endings[current]=dict(paid)
                current=qid; beginnings.append(qid); reads[qid]=0
            elif kind=='attempt':
                cell=event['cell']
                if current is None or pending is not None or event['query_id']!=current or event['index']!=len(attempted): raise ValueError('invalid attempt chronology')
                if cell in attempted or cell not in queries[current].cells or len(attempted)>=256: raise ValueError('illegal or repeated paid cell')
                attempted.add(cell); reads[current]+=1; pending=cell
            elif kind=='response':
                cell=event['cell']; label=event['label']
                if cell!=pending or label!=reference[cell]: raise ValueError('oracle response differs from sealed reference')
                paid[cell]=label; pending=None
            elif kind=='fallback':
                if event['query_id']!=current: raise ValueError('fallback outside current query')
                fallbacks.setdefault(current,set()).add(event['reason'])
            elif kind=='finished':
                if event['cost']!=len(attempted) or event['resource_stopped']!=result.get('resource_stopped'): raise ValueError('invalid finished accounting')
                finished=True
            else: raise ValueError('unknown journal event')
    if not finished: raise ValueError('journal lacks durable completion')
    if current is not None: endings[current]=dict(paid)
    if pending is not None and not result.get('resource_stopped'): raise ValueError('unfinished response without interruption')
    if result['cost']!=len(attempted) or result.get('budget')!=256: raise ValueError('forged cost or budget')
    if {int(k):v for k,v in result.get('observed',{}).items()}!=paid: raise ValueError('result cache differs from journal')
    if [row['query_id'] for row in result['queries']]!=beginnings: raise ValueError('result query order differs from journal')
    if result.get('fallbacks_by_query',{})!={qid:sorted(values) for qid,values in fallbacks.items()}: raise ValueError('fallback report differs from journal')
    for row in result['queries']:
        qid=row['query_id']; proof=row.get('certificate')
        if row['reads']!=reads[qid]: raise ValueError('forged per-query read count')
        if row['status']=='unresolved':
            if row.get('certificate_valid'): raise ValueError('unresolved query claims proof')
            continue
        if result.get('resource_stopped') and qid==current: raise ValueError('interrupted current query claims proof')
        if not proof or proof['status']!=row['status']: raise ValueError('missing or inconsistent proof')
        for cell,label in proof['facts']:
            if cell not in endings[qid] or endings[qid][cell] is None or endings[qid][cell]!=label: raise ValueError('unpaid proof fact')
        if proof.get('format')=='cq-v32-negative-certificate-1':
            context={} if certificate_contexts is None else certificate_contexts[qid]
            valid=replay_compact_certificate(queries[qid],proof,**context)
        else: valid=replay_certificate(queries[qid],Verification(proof['status'],proof['witness_id'],tuple(tuple(f) for f in proof['facts']),tuple(proof.get('blocked_witnesses',()))))
        if not valid or row.get('certificate_valid') is not True: raise ValueError('proof does not replay')
    return summarize_session(result,session['slots'])

def load_result(root,key,queries):
    """Check stored bytes and independently replay every claimed proof."""
    import gzip,json
    from .runtime import ArtifactStore
    from .core import Verification,replay_certificate
    store=ArtifactStore(root)
    if not store.validate(key): raise ValueError('invalid result artifact')
    with gzip.open(store.path(key)/'result.json.gz','rt',encoding='utf8') as stream: result=json.load(stream)
    for row in result['queries']:
        if not row.get('certificate_valid'): continue
        query=queries[row['query_id']]; proof=row['certificate']
        if proof.get('status')!=row['status']: raise ValueError('certificate/outcome status mismatch')
        if proof.get('format')=='cq-v32-negative-certificate-1':
            valid=replay_compact_certificate(query,proof)
        else:
            valid=replay_certificate(query,Verification(proof['status'],proof['witness_id'],
                tuple(tuple(f) for f in proof['facts']),tuple(proof.get('blocked_witnesses',()))))
        if not valid: raise ValueError('certificate failed independent replay')
    return result

@contextmanager
def _journal_transaction():
    import signal
    previous=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGPROF})
    try: yield
    finally: signal.pthread_sigmask(signal.SIG_SETMASK,previous)

def _atomic_journal(method):
    def wrapped(*args,**kwargs):
        with _journal_transaction(): return method(*args,**kwargs)
    return wrapped

class _Journal:
    def __init__(self,stream):
        self.stream=stream; self.attempts=[]; self.responses={}; self.begins=[]; self.fallbacks={}

    def record(self,value):
        import json,os
        with _journal_transaction():
            self.stream.write(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')
            self.stream.flush(); os.fsync(self.stream.fileno())

    @_atomic_journal
    def begin(self,query,observed):
        self.begins.append((query,dict(observed),len(self.attempts)))
        self.record({'kind':'begin','query_id':query.query_id,'attempt_count':len(self.attempts),'observed':dict(observed)})

    @_atomic_journal
    def attempt(self,cell):
        event={'kind':'attempt','cell':int(cell),'query_id':self.begins[-1][0].query_id,'index':len(self.attempts)}
        self.attempts.append(event); self.record(event)

    @_atomic_journal
    def response(self,cell,label):
        self.record({'kind':'response','cell':int(cell),'label':None if label is None else int(label)})
        self.responses[int(cell)]=None if label is None else int(label)

    def recover(self,seed):
        from dataclasses import asdict
        from .core import evaluate,replay_certificate
        rows=[]
        for i,(query,observed,start_reads) in enumerate(self.begins):
            if i+1<len(self.begins):
                snapshot=self.begins[i+1][1]; end_reads=self.begins[i+1][2]
                verdict=evaluate(query,snapshot)
                valid=verdict.status!='unresolved' and replay_certificate(query,verdict)
                row={'query_id':query.query_id,'status':verdict.status,'certificate_valid':valid,'certificate':asdict(verdict),
                     'reads':end_reads-start_reads,'stop_reason':'recovered_before_timeout'}
            else:
                row={'query_id':query.query_id,'status':'unresolved','certificate_valid':False,
                     'reads':len(self.attempts)-start_reads,'stop_reason':'resource_stop'}
            rows.append(row)
        return {'schema':'cq-annotation-session-v32','seed':seed,'budget':256,'cost':len(self.attempts),
                'queries':rows,'events':self.attempts,'observed':{str(k):v for k,v in self.responses.items()},
                'query_count':len(rows),'resolved_queries':sum(r['certificate_valid'] for r in rows)}

def budget_curve(result,slots,points=(0,32,64,128,256)):
    rows=[]; cumulative=0
    for row in result['queries']:
        start=cumulative; cumulative+=row['reads']; rows.append((start,cumulative,row))
    return {point:{'actual_reads':min(point,result['cost']),
        'exposed_slots':sum(start<=point for start,end,row in rows),
        'certified_slots':sum(end<=point and bool(row.get('certificate_valid')) for start,end,row in rows),
        'planned_slots':len(slots)} for point in points}

def run_heldout_session(session,queries,policy,oracle,prior,journal_path,*,seed=0,cpu_limit=120):
    """Durable attempts, current-query-only policy access, conservative timeout.

    Native code may delay CPU timer delivery; a hard outer limit is required.
    The journal must live outside disposable artifact stages.
    """
    import signal,time
    from pathlib import Path
    from .session import run_session
    path=Path(journal_path); path.parent.mkdir(parents=True,exist_ok=True)
    start=time.process_time(); wall=time.monotonic()
    def stop(signum,frame): raise ResourceStop('CPU deadline')
    if cpu_limit is not None and (cpu_limit<=0 or signal.getitimer(signal.ITIMER_PROF)[0]>0):
        raise ValueError('positive allowance and no nested CPU timer required')
    with path.open('x') as stream:
        journal=_Journal(stream)
        class JournalPolicy:
            @property
            def diagnostics(self): return getattr(policy,'diagnostics',{})
            def begin_query(self,query,observed):
                journal.begin(query,observed)
                result=policy.begin_query(query,observed)
                self.track(query)
                return result
            def choose_next(self,query,observed,model,rng):
                result=policy.choose_next(query,observed,model,rng)
                self.track(query)
                return result
            def track(self,query):
                reason=getattr(policy,'diagnostics',{}).get('fallback_reason')
                if reason and reason not in journal.fallbacks.get(query.query_id,set()):
                    with _journal_transaction():
                        journal.fallbacks.setdefault(query.query_id,set()).add(reason)
                        journal.record({'kind':'fallback','query_id':query.query_id,'reason':reason})
        def paid(cell):
            journal.attempt(cell)  # fsync precedes oracle access/return.
            label=oracle(cell)
            journal.response(cell,label)
            return label
        old_handler=signal.getsignal(signal.SIGPROF); interrupted=False
        try:
            if cpu_limit is not None:
                signal.signal(signal.SIGPROF,stop); signal.setitimer(signal.ITIMER_PROF,cpu_limit)
            result=run_session((queries[s['query_id']] for s in session['slots'] if not s.get('unsupported')),
                               JournalPolicy(),paid,prior,budget=256,seed=seed)
        except ResourceStop:
            interrupted=True
            result=journal.recover(seed)
        finally:
            if cpu_limit is not None:
                signal.setitimer(signal.ITIMER_PROF,0); signal.signal(signal.SIGPROF,old_handler)
        result.update(cost=len(journal.attempts),resource_stopped=interrupted,
            process_cpu_seconds=time.process_time()-start,wall_seconds=time.monotonic()-wall,
            requested_cpu_limit=cpu_limit,attempts_without_response=sum(e['cell'] not in journal.responses for e in journal.attempts))
        result['fallbacks_by_query']={key:sorted(value) for key,value in journal.fallbacks.items()}
        journal.record({'kind':'finished','cost':result['cost'],'resource_stopped':interrupted})
    return result
