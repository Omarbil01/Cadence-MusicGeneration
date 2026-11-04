import csv
import json
import stat
import zipfile

import pytest

from musicgen.generation.runtime.midi import Note, write_midi
from musicgen.generation.runtime.tokenizer import GENRES
from musicgen.training.colab import build_bundle, extract_bundle, validate_bundle
from musicgen.training.manifest import digest, load_approved


@pytest.fixture
def bundle_inputs(tmp_path):
    project, dataset = tmp_path / "project", tmp_path / "source_midi"
    package = project / "src" / "musicgen"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text('"""Minimal fixture project."""\n')
    (project / "pyproject.toml").write_text('[project]\nname="bundle-test"\nversion="0.1.0"\n')
    (project / "README.md").write_text("# Bundle test\n")
    (project / ".env").write_text("PRIVATE_VALUE=must-not-package\n")
    (project / "PDMX.csv").write_text("private,unselected,source\n")
    (package / "__pycache__").mkdir()
    (package / "__pycache__" / "cached.pyc").write_bytes(b"not source")
    rows = []
    for genre_id, genre in enumerate(GENRES):
        for split_id, split in enumerate(("train", "val", "test")):
            source = dataset / "mid" / genre / f"{split}.mid"
            source.parent.mkdir(parents=True, exist_ok=True)
            write_midi(source, [Note(0, 1, 60 + genre_id, split_id)], end_beat=1)
            rows.append(
                {
                    "candidate_id": f"{genre}_{split}",
                    "sha256": digest(source),
                    "midi_path": source.relative_to(dataset).as_posix(),
                    "genre": genre,
                    "genre_id": str(genre_id),
                    "composition_group": f"{genre}_{split}",
                    "split": split,
                    "training_status": "APPROVED",
                    "approved_for_training": "true",
                    "genre_status": "PROVISIONAL",
                    "rights_status": "UNVERIFIED",
                }
            )
    manifest = tmp_path / "selected.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"project": project, "dataset": dataset, "manifest": manifest, "rows": rows}


def _build(inputs, output):
    return build_bundle(
        inputs["manifest"],
        inputs["dataset"],
        output,
        project_root=inputs["project"],
        min_per_genre=3,
        max_per_genre=3,
    )


@pytest.fixture
def private_bundle(bundle_inputs, tmp_path):
    path = tmp_path / "private.zip"
    _build(bundle_inputs, path)
    return path


def _rewrite(original, destination, *, changed_name=None, changed_data=None, extra=None, mode=None):
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(destination, "w") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            if info.filename == changed_name:
                if changed_data is not None:
                    data = changed_data(data)
                if mode is not None:
                    info.external_attr = mode << 16
            target.writestr(info, data)
        if extra is not None:
            target.writestr(extra, b"unexpected member")


def test_bundle_roundtrip_preserves_manifest_hashes_and_excludes_private_files(
    bundle_inputs, tmp_path
):
    archive = tmp_path / "one.zip"
    summary = _build(bundle_inputs, archive)
    record = validate_bundle(archive)
    assert summary["selected_files"] == record["audit"]["rows"] == 9
    assert record["audit"]["by_genre"] == dict.fromkeys(GENRES, 3)
    assert summary["archive_sha256"] == digest(archive)
    assert record["audit"]["genre_status_counts"] == {"PROVISIONAL": 9}
    with zipfile.ZipFile(archive) as handle:
        assert handle.read("dataset/selected.csv") == bundle_inputs["manifest"].read_bytes()
        names = set(handle.namelist())
    assert len(names) == 14  # Three tiny project files, manifest, nine MIDIs and bundle.json.
    assert not any("PDMX.csv" in name or ".env" in name or "__pycache__" in name for name in names)
    extracted = extract_bundle(archive, tmp_path / "unpacked")
    rows = load_approved(
        extracted["manifest_path"],
        midi_root=extracted["dataset_root"],
        min_per_genre=3,
        max_per_genre=3,
    )
    assert rows == bundle_inputs["rows"]
    assert (extracted["project_root"] / "pyproject.toml").is_file()
    assert extracted["record"] == record
    with pytest.raises(FileExistsError):
        _build(bundle_inputs, archive)
    with pytest.raises(FileExistsError):
        extract_bundle(archive, tmp_path / "unpacked")
    assert digest(archive) == summary["archive_sha256"]
    repeated = _build(bundle_inputs, tmp_path / "two.zip")
    assert repeated["archive_sha256"] == summary["archive_sha256"]


def test_bundle_build_rechecks_source_hash_and_does_not_publish_on_failure(bundle_inputs, tmp_path):
    source = bundle_inputs["dataset"] / bundle_inputs["rows"][0]["midi_path"]
    source.write_bytes(source.read_bytes() + b"changed")
    output = tmp_path / "bad.zip"
    with pytest.raises(ValueError, match="Source hash mismatch"):
        _build(bundle_inputs, output)
    assert not output.exists()


def test_bundle_accepts_expanded_selection_count_contract(bundle_inputs, tmp_path):
    output = tmp_path / "expanded.zip"
    build_bundle(
        bundle_inputs["manifest"],
        bundle_inputs["dataset"],
        output,
        project_root=bundle_inputs["project"],
        min_per_genre=3,
        max_per_genre=50000,
    )
    record = validate_bundle(output)
    assert record["max_per_genre"] == 50000
    assert record["audit"]["rows"] == 9


@pytest.mark.parametrize(
    "unsafe", ["../outside.py", "/tmp/absolute.py", "project/src/musicgen/../../escape.py"]
)
def test_extraction_rejects_unsafe_names_before_writing(private_bundle, tmp_path, unsafe):
    altered = tmp_path / "unsafe.zip"
    _rewrite(private_bundle, altered, extra=unsafe)
    with pytest.raises(ValueError, match="Unsafe archive path"):
        extract_bundle(altered, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
    assert not list(tmp_path.glob(".cadence-extract-*"))
    assert not (tmp_path / "outside.py").exists()


def test_extraction_rejects_symlinks_and_undeclared_payload(private_bundle, tmp_path):
    symlink = tmp_path / "symlink.zip"
    _rewrite(
        private_bundle,
        symlink,
        changed_name="project/src/musicgen/__init__.py",
        mode=stat.S_IFLNK | 0o777,
    )
    with pytest.raises(ValueError, match="unsafe archive member"):
        validate_bundle(symlink)
    extra = tmp_path / "extra.zip"
    _rewrite(private_bundle, extra, extra="project/src/musicgen/unlisted.py")
    with pytest.raises(ValueError, match="inventory does not match"):
        validate_bundle(extra)


def test_extraction_checks_streamed_payload_hash_and_cleans_partial_data(private_bundle, tmp_path):
    changed = tmp_path / "changed.zip"
    _rewrite(
        private_bundle,
        changed,
        changed_name="dataset/mid/classical/train.mid",
        changed_data=lambda data: data[:-1] + bytes([data[-1] ^ 1]),
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        extract_bundle(changed, tmp_path / "rejected")
    assert not (tmp_path / "rejected").exists()
    assert not list(tmp_path.glob(".cadence-extract-*"))


@pytest.mark.parametrize(
    "limits", [{"max_total_bytes": 1}, {"max_member_bytes": 1}, {"max_members": 1}]
)
def test_extraction_enforces_size_and_count_limits(private_bundle, tmp_path, limits):
    with pytest.raises(ValueError, match="limit"):
        extract_bundle(private_bundle, tmp_path / "too_large", **limits)
    assert not (tmp_path / "too_large").exists()


def test_bundle_metadata_cannot_hide_missing_manifest_file(private_bundle, tmp_path):
    altered = tmp_path / "inventory.zip"

    def change_record(data):
        record = json.loads(data)
        del record["files"]["dataset/mid/classical/train.mid"]
        return json.dumps(record).encode()

    _rewrite(private_bundle, altered, changed_name="bundle.json", changed_data=change_record)
    with pytest.raises(ValueError, match="inventory does not match"):
        validate_bundle(altered)
