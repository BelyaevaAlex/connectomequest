import json

from scripts.build_artifact import ROOT, sealed_source_provenance


def test_source_provenance_is_exact_for_timing_confirmation() -> None:
    protocol = json.loads(
        (
            ROOT
            / "outputs/nemo/strong-accept/amortized-revalidation-v40/confirmation/protocol.json"
        ).read_text()
    )
    carried = sealed_source_provenance()
    index = json.loads(carried["provenance/SOURCE_INDEX.json"])
    graph_protocol = json.loads(
        (ROOT / "outputs/nemo/strong-accept/fresh-contract-test-v1/PROTOCOL.json").read_text()
    )
    expected = dict(protocol["source_hashes"])
    expected.update(
        {name: digest for name, digest in graph_protocol["inputs"].items() if name.endswith(".py")}
    )
    attribution_protocol = json.loads(
        (ROOT / "outputs/nemo/strong-accept/fresh-attribution-v35/PROTOCOL.json").read_text()
    )
    expected.update(attribution_protocol["source_sha256"])
    assert index == expected
    for digest in index.values():
        assert f"provenance/source/{digest}.txt" in carried


def test_artifact_does_not_package_its_own_secret_scanner() -> None:
    from scripts.build_artifact import publication_files

    names = {str(path.relative_to(ROOT)) for path in publication_files()}
    assert "scripts/build_artifact.py" not in names
    assert "scripts/build_submission.py" not in names


def test_artifact_contains_only_compact_graph_snapshot_files() -> None:
    from scripts.build_artifact import publication_files

    data_names = {path.name for path in publication_files() if "data/processed" in str(path)}
    assert data_names == {"edges.parquet", "nodes.parquet", "manifest.json"}


def test_archived_source_verification_prefers_indexed_snapshot(tmp_path) -> None:
    import hashlib
    import zipfile

    from connectomequest.embodied.protocol import verify_archived_source_hashes

    historical_path = "src/connectomequest/env.py"
    sealed = b"sealed source bytes\n"
    digest = hashlib.sha256(sealed).hexdigest()
    archive_path = tmp_path / "artifact.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(historical_path, b"current source bytes\n")
        archive.writestr("provenance/SOURCE_INDEX.json", json.dumps({historical_path: digest}))
        archive.writestr(f"provenance/source/{digest}.txt", sealed)

    verify_archived_source_hashes(archive_path, {historical_path: digest})


def test_private_material_scan_distinguishes_text_from_binary() -> None:
    from scripts.build_artifact import contains_private_material

    email = b"person" + b"@" + b"example.org"
    token_prefix = b"eyJ" + b"hbGci"
    assert contains_private_material("record.json", b'{"email": "' + email + b'"}')
    assert not contains_private_material("edges.parquet", email)
    assert contains_private_material("edges.parquet", token_prefix + b"OiJIUzI1NiJ9")
