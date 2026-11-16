#!/usr/bin/env python3
"""Official NTAC reference transfer; evaluation annotations are never loaded."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import time

import numpy as np

from connectomequest.annotation_v32.data import load_prepared_dataset, load_public_graph
from connectomequest.annotation_v32.predictors import reference_union, fit_ntac, ntac_class_scores, NtacPrior, fit_temperature, export_predictions
from connectomequest.annotation_v32.training import source_split
from connectomequest.annotation_v32.runtime import ArtifactStore, content_key, hash_file
from train_annotation_v32 import cached_features


def available_memory():
    fields = {line.split(':')[0]: line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()}
    available = int(fields['MemAvailable'].split()[0]) * 1024
    maximum = Path('/sys/fs/cgroup/memory.max')
    current = Path('/sys/fs/cgroup/memory.current')
    if maximum.exists() and current.exists() and maximum.read_text().strip() != 'max':
        limit, used = int(maximum.read_text()), int(current.read_text())
        statistics = Path('/sys/fs/cgroup/memory.stat')
        stats = dict(line.split() for line in statistics.read_text().splitlines()) if statistics.exists() else {}
        # memory.current includes page/slab caches. Allow clean inactive pages
        # and half of reclaimable slab; retain active cache and dirty pages.
        # The caller additionally reserves 30% of this estimate. Counters:
        # https://docs.kernel.org/admin-guide/cgroup-v2.html#memory-interface-files
        clean_inactive = max(0, int(stats.get('inactive_file', 0))
                             - int(stats.get('file_dirty', 0)) - int(stats.get('file_writeback', 0)))
        reclaimable = min(used, clean_inactive + int(stats.get('slab_reclaimable', 0)) // 2)
        available = min(available, max(0, min(limit, limit - used + reclaimable)))
    return available


def validate_dataset_tag(dataset, tag):
    valid = (dataset == 'malecns' and tag.startswith('male-cns:')) or (dataset == 'flywire' and tag == 'flywire783')
    if not valid:
        raise ValueError('evaluation dataset manifest mismatch')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--dataset', required=True, choices=('malecns', 'flywire'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--source-fit-key', required=True)
    parser.add_argument('--source-fit-output', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source_manifest = json.loads((args.source / 'public/manifest.json').read_text())
    evaluation_manifest = json.loads((args.evaluation / 'public/manifest.json').read_text())
    if not source_manifest.get('metadata', {}).get('dataset', '').startswith('hemibrain:'):
        raise ValueError('source must be hemibrain')
    validate_dataset_tag(args.dataset, evaluation_manifest.get('metadata', {}).get('dataset', ''))
    source_inputs = {str(p.relative_to(args.source)): hash_file(p) for folder in ('public', 'private') for p in (args.source / folder).iterdir() if p.is_file()}
    evaluation_inputs = {str(p.relative_to(args.evaluation)): hash_file(p) for p in (args.evaluation / 'public').iterdir() if p.is_file()}
    source_store = ArtifactStore(args.source_fit_output / 'fits')
    if not source_store.validate(args.source_fit_key):
        raise ValueError('source NTAC fit artifact invalid')
    source_identity = json.loads((source_store.path(args.source_fit_key) / 'identity.json').read_text())
    if source_identity['inputs'] != source_inputs:
        raise ValueError('source fit and transfer input identities differ')
    code = [Path(__file__), root / 'scripts/train_annotation_v32.py']
    code += [root / 'src/connectomequest/annotation_v32' / name for name in ('__init__.py', 'data.py', 'predictors.py', 'priors.py', 'training.py', 'runtime.py')]
    code += sorted((root / '.v32-deps/ntac').rglob('*.py'))
    dependencies = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'torch', 'numba', 'bottleneck', 'pyarrow')}
    identity = {'schema': 'v32-official-ntac-reference-transfer-1', 'dataset': args.dataset,
                'source_inputs': source_inputs, 'evaluation_public_inputs': evaluation_inputs,
                'source_fit_key': args.source_fit_key, 'code': {str(p.relative_to(root)): hash_file(p) for p in code},
                'dependencies': dependencies, 'ntac_commit': (root / '.v32-deps/NTAC-COMMIT').read_text().strip(),
                'vocabulary_sha256': content_key(source_manifest['vocabulary']),
                'source_metadata': source_manifest['metadata'], 'evaluation_metadata': evaluation_manifest['metadata'],
                'config': {'split_seed': 0, 'steps': 3, 'lr': .3, 'topk': 1, 'feature_weight': 1.,
                           'numba_threads': 1, 'workspace_cap_bytes': 192000000000}}
    key = content_key(identity)
    store = ArtifactStore(args.output / 'fits')
    if args.dry_run:
        print(json.dumps({'key': key, 'identity': identity}, sort_keys=True, indent=2)); return
    if store.validate(key):
        print(store.path(key)); return
    start = time.monotonic()
    with store.claim(key) as claim:
        # Preserve the exact execution source beside its digests, before fitting.
        for path in code:
            destination = claim.stage / 'code' / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        (claim.stage / 'identity.json').write_text(json.dumps(identity, sort_keys=True, indent=2))
        source, private = load_prepared_dataset(args.source)
        evaluation = load_public_graph(args.evaluation)
        train, calibration, _ = source_split(private.labels, 0)
        source_features, source_feature_key = cached_features(source, args.features,
            {k: v for k, v in source_inputs.items() if k.startswith('public/')}, dependencies)
        evaluation_features, evaluation_feature_key = cached_features(evaluation, args.features, evaluation_inputs, dependencies)
        n = len(source.node_ids) + len(evaluation.node_ids)
        estimate = 8 * n * (3 * len(train) + 10 * len(set(private.labels[c] for c in train)) + 64) + 64 * (len(source.indices) + len(evaluation.indices))
        available = available_memory()
        if estimate > 192000000000 or estimate > .7 * available:
            raise MemoryError(f'NTAC workspace admission failed: estimated={estimate}, available={available}')
        print(json.dumps({'phase': 'fit', 'dataset': args.dataset, 'estimate_bytes': estimate, 'available_bytes': available, 'fit_key': key}), flush=True)
        union, source_map, evaluation_map = reference_union(source, evaluation)
        features = np.concatenate((source_features, evaluation_features))
        references = {source_map[cell]: private.labels[cell] for cell in train}
        model = fit_ntac(union, references, features=features, steps=3, max_workspace_bytes=192000000000)
        fit_seconds = time.monotonic() - start
        scores = ntac_class_scores(model, np.array([source_map[cell] - 1 for cell in calibration]), len(source.vocabulary))
        temperature = fit_temperature(scores, np.array([private.labels[cell] for cell in calibration]))
        prior = NtacPrior(evaluation.node_ids, np.array([evaluation_map[int(cell)] - 1 for cell in evaluation.node_ids]), model, len(source.vocabulary), temperature=temperature)
        prediction_key = export_predictions(prior, evaluation.node_ids, args.output / 'predictions',
            {'fit_key': key, 'temperature': temperature, 'source_feature_key': source_feature_key,
             'evaluation_feature_key': evaluation_feature_key, 'vocabulary_sha256': identity['vocabulary_sha256']})
        report = {'fit_key': key, 'source_fit_key': args.source_fit_key, 'prediction_key': prediction_key,
                  'dataset': args.dataset, 'source_feature_key': source_feature_key, 'evaluation_feature_key': evaluation_feature_key,
                  'temperature': temperature, 'fit_wall_seconds': fit_seconds, 'total_wall_seconds': time.monotonic() - start,
                  'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  'estimated_workspace_bytes': estimate, 'available_memory_at_admission_bytes': available,
                  'training_rows': len(train), 'evaluation_rows': len(evaluation.node_ids), 'numba_threads': 1}
        (claim.stage / 'report.json').write_text(json.dumps(report, sort_keys=True, indent=2))
        print(claim.complete())


if __name__ == '__main__':
    main()
