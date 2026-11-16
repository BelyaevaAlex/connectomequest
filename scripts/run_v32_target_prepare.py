#!/usr/bin/env python3
"""Conditionally meter source-only fitting and public FlyWire preparation."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from connectomequest.annotation_v32.runtime import ArtifactStore,content_key,execute,hash_file

MALE_COHORT='aa6e5ac51470538676e46701d527e606a38c272b6a256b365c93f9868796c994'
SOURCE_FIT='0e5c3ab567527b5fcf1374581aa048cf57462d7a767f0216a90a0d1db5553af5'
MLP='59cd92072106eb705ce5ae288aa891a79ee47de74bbba3cc5dce30ae4fc38ea3'
SAGE='773829936906404bcbe055a55f8c09cebe2561bd0b50542bda80309dbb9561d8'


def load_admitted_decision(root,key):
    store=ArtifactStore(root)
    if not store.validate(key): raise ValueError('invalid MaleCNS decision artifact')
    row=json.loads((store.path(key)/'decision.json').read_text())
    if content_key(row)!=key: raise ValueError('decision key mismatch')
    if row.get('operational_pass') is not True or row.get('independently_audited_runs')!=132 or row.get('invalid_certificates')!=0:
        raise ValueError('MaleCNS decision has not passed operational audit')
    branch=row.get('branch'); seeds=row.get('allowed_training_seeds')
    valid=(branch=='descriptive_seed0' and row.get('superiority_pass') is False and seeds==[0]) or (branch=='superiority_confirmation' and row.get('superiority_pass') is True and seeds==[0,1,2])
    keys=row.get('run_keys',[])
    if not valid or len(keys)!=132 or len(set(keys))!=132 or any(not isinstance(k,str) or len(k)!=64 for k in keys):
        raise ValueError('inconsistent decision branch or incomplete run identities')
    return row


def validate_male_matrix(root,decision):
    from connectomequest.annotation_v32 import heldout as h
    if decision.get('cohort_key')!=MALE_COHORT: raise ValueError('unexpected MaleCNS cohort')
    frozen=h.load_cohort(root/'outputs/v32/heldout/cohorts',MALE_COHORT)
    if frozen['descriptor']['dataset']!='malecns': raise ValueError('decision is not MaleCNS')
    runs=ArtifactStore(root/'outputs/v32/heldout/results/runs'); pairs=[]
    for key in decision['run_keys']:
        if not runs.validate(key): raise ValueError('invalid audited MaleCNS run')
        path=runs.path(key); identity=json.loads((path/'identity.json').read_text()); summary=json.loads((path/'summary.json').read_text())
        if content_key(identity)!=key or identity.get('cohort_key')!=MALE_COHORT or summary.get('dataset')!='malecns':
            raise ValueError('MaleCNS run identity mismatch')
        pairs.append((identity['session_id'],identity['arm']))
    expected={(s['session_id'],a) for s in frozen['sessions'] for a in h.VALIDATION_ARMS}
    if len(expected)!=132 or len(pairs)!=132 or set(pairs)!=expected: raise ValueError('incomplete MaleCNS matrix')


def phase_command(phase,*,cohort_key=None,ntac_key=None,reference_sha256=None,authorization=None):
    base=['scripts/evaluate_annotation_v32.py']
    if phase=='freeze':
        if not isinstance(reference_sha256,str) or len(reference_sha256)!=64 or any(c not in '0123456789abcdef' for c in reference_sha256): raise ValueError('sealed reference SHA-256 required')
        return base+['freeze','--source','inputs/v32/prepared-common-v1/hemibrain','--prepared','inputs/v32/prepared-common-v1/flywire','--dataset','flywire','--reference-sha256',reference_sha256,'--features','outputs/v32/predictors/_features','--output','outputs/v32/heldout-target/cohorts'],3600
    if phase=='ntac':
        return ['scripts/fit_transfer_annotation_v32_ntac.py','--source','inputs/v32/prepared-common-v1/hemibrain','--evaluation','inputs/v32/prepared-common-v1/flywire','--dataset','flywire','--output','outputs/v32/ntac-transfer/flywire','--features','outputs/v32/predictors/_features','--source-fit-output','outputs/v32/ntac-source','--source-fit-key',SOURCE_FIT],5400
    if phase=='predictors':
        if not cohort_key or not ntac_key or not authorization: raise ValueError('cohort, NTAC key and authorization required')
        return base+['prepare-predictors','--cohorts','outputs/v32/heldout-target/cohorts','--cohort-key',cohort_key,'--models','outputs/v32/predictors','--mlp-key',MLP,'--graphsage-key',SAGE,'--ntac-output','outputs/v32/ntac-transfer/flywire','--ntac-fit-key',ntac_key,'--features','outputs/v32/predictors/_features','--output','outputs/v32/heldout-target/predictors','--authorization',str(authorization)],1800
    raise ValueError('unknown preparation phase')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('freeze','ntac','predictors')); parser.add_argument('--decision-key',required=True)
    parser.add_argument('--cohort-key');parser.add_argument('--ntac-key');parser.add_argument('--reference-sha256')
    parser.add_argument('--authorization',type=Path);parser.add_argument('--attempt',default='0');parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args(argv)
    # No target inputs or target artifacts are opened before this gate.
    decision=load_admitted_decision(ROOT/'outputs/v32/heldout/decisions',args.decision_key)
    validate_male_matrix(ROOT,decision)
    if args.phase=='predictors':
        if args.authorization is None: parser.error('predictors requires --authorization')
        auth=json.loads(args.authorization.read_text())
        if auth.get('allow_prediction_preparation') is not True or auth.get('cohort_key')!=args.cohort_key or auth.get('allowed_training_seeds')!=[0]: raise ValueError('target seed-zero prediction preparation is not authorized')
    command,cap=phase_command(args.phase,cohort_key=args.cohort_key,ntac_key=args.ntac_key,reference_sha256=args.reference_sha256,authorization=args.authorization)
    identity={'phase':args.phase,'decision_key':args.decision_key,'decision_branch':decision['branch'],
              'command':command,'cpu_cap_seconds':cap,'launcher_sha256':hash_file(__file__),'entry_sha256':hash_file(ROOT/command[0])}
    key=content_key(identity)
    bootstrap=('import os,resource,runpy,sys;'
               'cap=int(sys.argv.pop(1));resource.setrlimit(resource.RLIMIT_CPU,(cap,cap));'
               'script=os.path.abspath(sys.argv.pop(1));sys.path.insert(0,os.path.dirname(script));'
               'sys.argv[0]=script;runpy.run_path(script,run_name="__main__")')
    bounded=['timeout','--signal=TERM','--kill-after=30s',str(cap*2)+'s',sys.executable,'-c',bootstrap,str(cap),*command]
    if args.dry_run:
        print(json.dumps({'run_key':key,'command':bounded,'identity':identity},sort_keys=True)); return 0
    os.chdir(ROOT)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'): os.environ[name]='1'
    os.environ['PYTHONPATH']='src:.v32-deps'
    print('preparation_run_key='+key,flush=True)
    return execute(ROOT/'outputs/v32/target-preparation-usage',key,0,bounded,attempt=args.phase+'-'+args.attempt)

if __name__=='__main__': raise SystemExit(main())
