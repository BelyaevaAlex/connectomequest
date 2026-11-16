#!/usr/bin/env python3
"""Derive immutable NTAC artifacts after public hint correction, without fitting.

Only source supervision is hashed. Evaluation private files are never opened.
Feature identities remain compatible with the frozen cached_features reader;
copy provenance is stored separately in derivation.json.
"""
import argparse
import copy
import errno
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import resource
import shutil
import time

import numpy as np

from connectomequest.annotation_v32.data import load_public_graph, ontology_manifest
from connectomequest.annotation_v32.predictors import structural_features
from connectomequest.annotation_v32.runtime import ArtifactStore, content_key, hash_file
from correct_v32_historical_hints import public_hashes, public_equivalence

ROOT = Path(__file__).resolve().parents[1]
CONFIG = dict(split_seed=0, steps=3, lr=.3, topk=1, feature_weight=1.,
              numba_threads=1, workspace_cap_bytes=192000000000)
COMMIT = '04a8e4607993266315e8c36d089057e928116483'
DEPENDENCIES = ('numpy', 'scipy', 'torch', 'numba', 'bottleneck', 'pyarrow')
ADAPTER = 'scripts/correct_v32_historical_hints.py'
REBINDER = 'scripts/rebind_v32_hint_predictions.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


def producer_code(source=False):
    names = ['scripts/fit_annotation_v32_ntac.py' if source else 'scripts/fit_transfer_annotation_v32_ntac.py',
             'scripts/train_annotation_v32.py']
    names += ['src/connectomequest/annotation_v32/' + name for name in
              ('__init__.py', 'data.py', 'predictors.py', 'priors.py', 'training.py', 'runtime.py')]
    names += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / '.v32-deps/ntac').rglob('*.py'))]
    return {name: hash_file(ROOT / name) for name in names}


def artifact(root, key):
    require(root.is_dir(), 'missing parent artifact store: ' + str(root))
    store = ArtifactStore(root)
    require(store.validate(key), 'invalid artifact: ' + key)
    path = store.path(key)
    identity = read_json(path / 'identity.json')
    require(content_key(identity) == key, 'artifact identity/key mismatch')
    return path, identity


def source_hashes(source):
    return {p.relative_to(source).as_posix(): hash_file(p)
            for folder in ('public', 'private') for p in sorted((source / folder).iterdir()) if p.is_file()}


def validate_feature(path, identity, graph, public, dependencies):
    expected = dict(public=public, implementation=inspect.getsource(structural_features),
                    dependencies={k: dependencies[k] for k in ('numpy', 'scipy')})
    require(identity == expected, 'feature public identity/implementation/dependencies changed')
    with np.load(path / 'features.npz', allow_pickle=False) as values:
        require(set(values.files) == {'node_ids', 'features'}, 'unexpected feature payload')
        require(np.array_equal(values['node_ids'], graph.node_ids), 'feature node IDs differ')
        matrix = values['features']
        require(matrix.shape == (len(graph.node_ids), 12) and matrix.dtype == np.float32,
                'feature matrix shape/dtype mismatch')
        require(np.isfinite(matrix).all(), 'invalid feature values')


def validate_prediction(path, descriptor, graph):
    require(descriptor.get('schema') == 'v32-probability-f32-1', 'unsupported prediction schema')
    ids = np.load(path / 'node_ids.npy', mmap_mode='r', allow_pickle=False)
    values = np.load(path / 'probabilities.npy', mmap_mode='r', allow_pickle=False)
    require(ids.dtype == np.int64 and np.array_equal(ids, graph.node_ids), 'prediction node IDs differ')
    require(descriptor['node_ids_sha256'] == hashlib.sha256(ids.tobytes()).hexdigest(), 'prediction row hash differs')
    require(descriptor['label_count'] == len(graph.vocabulary), 'prediction vocabulary differs')
    require(values.dtype == np.float32 and values.shape == (len(ids), len(graph.vocabulary)),
            'prediction matrix shape/dtype mismatch')
    for start in range(0, len(ids), 1024):
        block = values[start:start + 1024]
        require(np.isfinite(block).all() and np.all(block >= 0)
                and np.allclose(block.sum(axis=1, dtype=np.float64), 1., rtol=1e-5, atol=1e-6),
                'invalid probability matrix')


def link_or_copy(source, destination, expected):
    """Create a new name; never chmod or write an existing parent inode."""
    try:
        os.link(source, destination)
        method = 'hardlink'
    except OSError as error:
        if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES, errno.EMLINK, errno.ENOTSUP):
            raise
        shutil.copyfile(source, destination)
        method = 'copy'
    require(hash_file(destination) == expected, 'payload changed during derivation')
    return method


def rebind(*, parent_prepared, prepared, source, source_fit_output, parent_output,
           parent_fit_key, parent_prediction_key, features, output, dry_run=False):
    started, cpu_started = time.monotonic(), time.process_time()
    parent_prepared, prepared, source, source_fit_output, parent_output, features, output = map(
        lambda p: Path(p).resolve(), (parent_prepared, prepared, source, source_fit_output, parent_output, features, output))
    for protected in (parent_prepared, prepared, source, source_fit_output, parent_output, features):
        require(not output.is_relative_to(protected) and not protected.is_relative_to(output),
                'output must be separate from input stores')
    require(parent_prepared != prepared, 'corrected dataset must have a new root')
    fit_path, parent_identity = artifact(parent_output / 'fits', parent_fit_key)
    prediction_path, descriptor = artifact(parent_output / 'predictions', parent_prediction_key)
    report = read_json(fit_path / 'report.json')
    require(parent_identity.get('schema') == 'v32-official-ntac-reference-transfer-1'
            and parent_identity.get('dataset') == 'flywire', 'unsupported parent fit')
    require(parent_identity['config'] == CONFIG, 'parent NTAC config changed')
    require(parent_identity['ntac_commit'] == COMMIT == (ROOT / '.v32-deps/NTAC-COMMIT').read_text().strip(),
            'NTAC commit changed')
    dependencies = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
    require(parent_identity['dependencies'] == dependencies, 'parent dependencies changed')
    code = producer_code()
    require(parent_identity['code'] == code, 'parent producer code changed')
    for relative, digest in code.items():
        archived = fit_path / 'code' / relative
        require(archived.is_file() and hash_file(archived) == digest, 'producer code archive mismatch')
    parent_public, derived_public = public_hashes(parent_prepared), public_hashes(prepared)
    require(parent_identity['evaluation_public_inputs'] == parent_public, 'original public inputs changed')
    parent_graph, graph = load_public_graph(parent_prepared), load_public_graph(prepared)
    require(parent_graph.metadata == parent_identity['evaluation_metadata'], 'parent evaluation metadata mismatch')
    require(parent_graph.metadata.get('dataset') == 'flywire783'
            and parent_graph.metadata.get('hint_release') == 'flywire:v2.1.0'
            and parent_graph.metadata.get('min_synapses') == 5, 'unexpected public release/threshold')
    metadata = copy.deepcopy(graph.metadata)
    correction = metadata.pop('historical_hint_correction', None)
    require(metadata == parent_graph.metadata and isinstance(correction, dict), 'non-hint metadata changed')
    require(correction.get('schema') == 'v32-old-hint-correction-1'
            and correction.get('parent_public_inputs') == parent_public
            and correction.get('adapter_sha256') == hash_file(ROOT / ADAPTER), 'hint correction lineage mismatch')
    proof = public_equivalence(parent_prepared, prepared)
    require(parent_identity['vocabulary_sha256'] == proof['vocabulary_sha256'], 'parent vocabulary mismatch')
    require(graph.metadata['ontology'] == ontology_manifest(), 'ontology implementation changed')
    source_graph = load_public_graph(source)
    require(source_graph.metadata.get('dataset', '').startswith('hemibrain:'), 'source must be hemibrain')
    inputs = source_hashes(source)
    require(inputs == parent_identity['source_inputs'], 'source inputs changed')
    require(source_graph.metadata == parent_identity['source_metadata']
            and source_graph.vocabulary == graph.vocabulary, 'source metadata/vocabulary changed')
    source_fit_key = parent_identity['source_fit_key']
    source_fit_path, source_identity = artifact(source_fit_output / 'fits', source_fit_key)
    require(source_identity.get('schema') == 'v32-source-official-ntac-1'
            and source_identity['inputs'] == inputs, 'source fit/input mismatch')
    source_code = producer_code(source=True)
    require(source_identity['code'] == source_code, 'source producer code changed')
    require(source_identity['config'] == {**CONFIG, 'workspace_cap_bytes': 64000000000}
            and source_identity['ntac_commit'] == COMMIT, 'source fit protocol changed')
    require(source_identity['dependencies'] == {k: dependencies[k] for k in DEPENDENCIES if k != 'pyarrow'},
            'source fit dependencies changed')
    require(report['fit_key'] == parent_fit_key and report['prediction_key'] == parent_prediction_key
            and report['source_fit_key'] == source_fit_key and report['dataset'] == 'flywire'
            and report['evaluation_rows'] == len(graph.node_ids), 'parent fit report mismatch')
    require(np.isfinite(report['temperature']) and report['temperature'] > 0, 'invalid parent temperature')
    require(descriptor['identity'] == dict(fit_key=parent_fit_key, temperature=report['temperature'],
            source_feature_key=report['source_feature_key'], evaluation_feature_key=report['evaluation_feature_key'],
            vocabulary_sha256=parent_identity['vocabulary_sha256']), 'parent prediction lineage mismatch')
    validate_prediction(prediction_path, descriptor, graph)
    feature_path, feature_identity = artifact(features, report['evaluation_feature_key'])
    source_feature_path, source_feature_identity = artifact(features, report['source_feature_key'])
    validate_feature(feature_path, feature_identity, parent_graph, parent_public, dependencies)
    validate_feature(source_feature_path, source_feature_identity, source_graph, public_hashes(source), dependencies)

    # All mathematical inputs are exact; public hints and their provenance differ.
    adapter_code = {ADAPTER: hash_file(ROOT / ADAPTER), REBINDER: hash_file(ROOT / REBINDER)}
    proof.update(node_ids_sha256=descriptor['node_ids_sha256'],
                 numerical_dependency_argument='structural_features uses weighted CSR only; reference_union discards all hints; NTAC uses source labels, fixed config and structural features only.')
    watched_artifacts = [fit_path, prediction_path, source_fit_path, feature_path, source_feature_path]
    parent_manifests = {str(p): hash_file(p / 'manifest.json') for p in watched_artifacts}
    lineage = dict(schema='v32-ntac-hint-only-derivation-1', no_refit=True,
                   parent_fit_key=parent_fit_key, parent_prediction_key=parent_prediction_key,
                   parent_fit_manifest_sha256=hash_file(fit_path / 'manifest.json'),
                   parent_fit_identity_sha256=hash_file(fit_path / 'identity.json'),
                   parent_fit_report_sha256=hash_file(fit_path / 'report.json'),
                   parent_prediction_manifest_sha256=hash_file(prediction_path / 'manifest.json'),
                   parent_prediction_identity_sha256=hash_file(prediction_path / 'identity.json'),
                   parent_feature_key=report['evaluation_feature_key'],
                   parent_feature_manifest_sha256=hash_file(feature_path / 'manifest.json'),
                   source_fit_manifest_sha256=hash_file(source_fit_path / 'manifest.json'),
                   source_feature_manifest_sha256=hash_file(source_feature_path / 'manifest.json'),
                   parent_public_inputs=parent_public, derived_public_inputs=derived_public,
                   equality_proof=proof, adapter_code=adapter_code,
                   payload_sha256={n: hash_file(prediction_path / n) for n in ('probabilities.npy', 'node_ids.npy')})
    new_feature_identity = {**feature_identity, 'public': derived_public}
    feature_key = content_key(new_feature_identity)
    require(feature_key != report['evaluation_feature_key'], 'public identity did not change')
    feature_lineage = dict(schema='v32-structural-feature-hint-derivation-1', no_recompute=True,
                           parent_key=report['evaluation_feature_key'], equality_proof=proof,
                           adapter_code=adapter_code, features_sha256=hash_file(feature_path / 'features.npz'))
    identity = copy.deepcopy(parent_identity)
    identity.update(schema='v32-official-ntac-reference-transfer-hint-derivation-1',
                    evaluation_public_inputs=derived_public, evaluation_metadata=graph.metadata,
                    code={**code, **source_code, **adapter_code}, derivation=lineage)
    fit_key = content_key(identity)
    new_descriptor = copy.deepcopy(descriptor)
    new_descriptor['identity'].update(fit_key=fit_key, evaluation_feature_key=feature_key,
                                      derivation=dict(no_refit=True, parent_prediction_key=parent_prediction_key,
                                                      parent_fit_key=parent_fit_key))
    prediction_key = content_key(new_descriptor)
    result = dict(fit_key=fit_key, prediction_key=prediction_key, evaluation_feature_key=feature_key)
    if dry_run:
        return {**result, 'dry_run': True, 'identity': identity}

    def unchanged():
        require(public_hashes(parent_prepared) == parent_public and public_hashes(prepared) == derived_public,
                'public inputs changed during derivation')
        require(source_hashes(source) == inputs, 'source changed during derivation')
        require(all(hash_file(ROOT / p) == digest for p, digest in identity['code'].items()),
                'code changed during derivation')
        require({n: importlib.metadata.version(n) for n in DEPENDENCIES} == dependencies,
                'dependencies changed during derivation')
        for p in watched_artifacts:
            require(hash_file(p / 'manifest.json') == parent_manifests[str(p)]
                    and ArtifactStore(p.parent).validate(p.name), 'parent artifact changed during derivation')

    stores = dict(features=ArtifactStore(features), predictions=ArtifactStore(output / 'predictions'), fits=ArtifactStore(output / 'fits'))
    for store, key in ((stores['features'], feature_key), (stores['predictions'], prediction_key), (stores['fits'], fit_key)):
        require(not store.path(key).exists() or store.validate(key), 'existing derived artifact is invalid')
    if stores['features'].path(feature_key).exists():
        fp = stores['features'].path(feature_key)
        require(read_json(fp / 'identity.json') == new_feature_identity
                and read_json(fp / 'derivation.json') == feature_lineage
                and hash_file(fp / 'features.npz') == feature_lineage['features_sha256'], 'existing feature derivation mismatch')
    else:
        with stores['features'].claim(feature_key) as claim:
            shutil.copyfile(feature_path / 'features.npz', claim.stage / 'features.npz')
            require(hash_file(claim.stage / 'features.npz') == feature_lineage['features_sha256'], 'feature copy changed')
            write_json(claim.stage / 'identity.json', new_feature_identity)
            write_json(claim.stage / 'derivation.json', feature_lineage)
            claim.complete()
    if not stores['predictions'].path(prediction_key).exists():
        with stores['predictions'].claim(prediction_key) as claim:
            methods = {n: link_or_copy(prediction_path / n, claim.stage / n, digest)
                       for n, digest in lineage['payload_sha256'].items()}
            write_json(claim.stage / 'identity.json', new_descriptor)
            write_json(claim.stage / 'derivation.json', {**lineage, 'payload_transfer': methods})
            claim.complete()
    else:
        pp = stores['predictions'].path(prediction_key)
        require(read_json(pp / 'identity.json') == new_descriptor
                and all(hash_file(pp / n) == digest for n, digest in lineage['payload_sha256'].items()),
                'existing prediction derivation mismatch')
    unchanged()
    if not stores['fits'].path(fit_key).exists():
        with stores['fits'].claim(fit_key) as claim:
            for relative, digest in identity['code'].items():
                destination = claim.stage / 'code' / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
                require(hash_file(destination) == digest, 'code changed during archive')
            write_json(claim.stage / 'identity.json', identity)
            write_json(claim.stage / 'parent_identity.json', parent_identity)
            write_json(claim.stage / 'report.json', dict(
                **result, schema='v32-ntac-hint-derivation-report-1', no_refit=True,
                source_fit_key=source_fit_key, dataset='flywire', source_feature_key=report['source_feature_key'],
                temperature=report['temperature'], evaluation_rows=len(graph.node_ids),
                derivation_wall_seconds_until_report=time.monotonic() - started,
                derivation_cpu_seconds_until_report=time.process_time() - cpu_started,
                process_peak_rss_kib_at_report=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                parent_fit_report=report))
            claim.complete()
    else:
        require(read_json(stores['fits'].path(fit_key) / 'identity.json') == identity,
                'existing fit derivation identity mismatch')
    require(all(store.validate(key) for store, key in
                ((stores['features'], feature_key), (stores['predictions'], prediction_key), (stores['fits'], fit_key))),
            'derived artifact verification failed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('parent-prepared', 'prepared', 'source', 'source-fit-output', 'parent-output', 'features', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--parent-fit-key', required=True)
    parser.add_argument('--parent-prediction-key', required=True)
    parser.add_argument('--dry-run', action='store_true')
    print(json.dumps(rebind(**vars(parser.parse_args())), sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
