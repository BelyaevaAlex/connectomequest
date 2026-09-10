# Data provenance

The exact processed HemiBrain-KG, MANC-KG, H01-KG files used by the reference
MICCAI 2025 paper are not publicly linked. The paper page lists the dataset as
N/A. ConnectomeQuest therefore creates BWKG-v2 snapshots from official releases
and records every source version, source URL, filter, file size, and checksum.

Reference paper:
https://papers.miccai.org/miccai-2025/0110-Paper4041.html

## MANC / Male CNS

Source: https://male-cns.janelia.org/download/

Version: male-cns:v1.0, CC-BY.

Downloaded artifacts are curated neuron annotations (about 14.5 MB) and the
weighted segment connection graph (about 1.05 GB). The raw graph contains
151,856,684 fragment pairs. BWKG-v2 retains connections between neurons with a
non-null mancType. The current weight-1 snapshot has:

- 22,744 curated neurons;
- 3,893 type entities;
- 3,874,922 directed wiring pairs;
- inverse edges and has_type edges.

The minimum synapse weight is configurable and must be reported as an ablation.

## H01

Source: https://h01-release.storage.googleapis.com/data.html

Version: C3 segmentation and 20210729 synaptic export, CC BY 4.0.

The public prefix contains 332 objects totaling 148.01 GiB: 166 binary Avro
objects (30.60 GiB) and 166 parallel JSON exports (117.41 GiB). The pipeline
records the complete inventory but selects only Avro, avoiding duplicate
downloads. It:

1. enumerates labels from the official cell-body mesh layer;
2. samples the official C3 segmentation at each soma mesh centroid;
3. samples the official cortical-layer volume at each soma centroid;
4. retains nonzero C3 IDs and explicit L1--L6/WM labels;
5. streams binary Avro shards through local scratch;
6. counts synapses whose endpoints are both soma-bearing C3 segments;
7. deletes each raw shard after reduction;
8. writes the compact graph and provenance manifest.

Soma mapping and Avro reduction both write one atomic checkpoint per source
shard. Interrupted 30.60 GiB scans resume without downloading completed objects.
Cell-body meshes are read only while computing centroids; raw meshes and Avro
objects are not retained in the processed benchmark. The soma-to-segment mapping
and compact edge-count reductions are retained with checksums.

## HemiBrain

Source: https://neuprint.janelia.org/

Version: hemibrain:v1.2.1, CC BY 4.0.

neuPrint requires a personal token. Set NEUPRINT_TOKEN to either the token itself
or a file containing it. The export keeps traced, non-cropped neurons and reads
weighted ConnectsTo edges in resumable source-ID chunks. Semantic entities
include type, instance, cell-body fiber, and ROI.

Never commit a token. Token files and .env are ignored.

## Reference-code licensing

The reference GitHub repository does not declare a source-code license. This
project does not copy its implementation. Torus operations are a clean
implementation from the published mathematical description.
