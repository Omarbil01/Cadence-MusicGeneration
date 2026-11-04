import copy
import csv
import json
import sqlite3
from types import SimpleNamespace

import pytest

from musicgen.generation.runtime.midi import Note, write_midi
from musicgen.generation.runtime.tokenizer import GENRES, VOCAB_SHA
from musicgen.training import preprocessing
from musicgen.training.cache import SQLiteWindows, prepare_sqlite_dataset
from musicgen.training.manifest import digest, load_approved


@pytest.fixture
def selected(tmp_path):
    rows = []
    for genre_id, genre in enumerate(GENRES):
        for split_id, split in enumerate(("train", "val", "test")):
            path = tmp_path / "mid" / f"{genre}_{split}.mid"
            path.parent.mkdir(exist_ok=True)
            notes = [
                Note(i / 2, (i + 1) / 2, 60 + i % 5, genre_id + split_id * 16) for i in range(20)
            ]
            write_midi(path, notes, end_beat=10)
            rows.append(
                {
                    "candidate_id": path.stem,
                    "sha256": digest(path),
                    "midi_path": str(path.relative_to(tmp_path)),
                    "genre": genre,
                    "genre_id": str(genre_id),
                    "composition_group": path.stem,
                    "split": split,
                    "training_status": "APPROVED",
                    "approved_for_training": "true",
                }
            )
    manifest = tmp_path / "selected.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"root": tmp_path, "manifest": manifest, "cache": tmp_path / "cache", "rows": rows}


def _prepare(selected, **options):
    rows = load_approved(
        selected["manifest"], midi_root=selected["root"], min_per_genre=3, max_per_genre=10000
    )
    identity = {
        "pipeline_version": "test-cache-v1",
        "vocab_sha256": VOCAB_SHA,
        "context": 32,
        "manifest_sha256": digest(selected["manifest"]),
        "preprocessing_sha256": digest(preprocessing.__file__),
        **options,
    }
    return prepare_sqlite_dataset(
        selected["manifest"],
        selected["root"],
        selected["cache"],
        rows,
        identity,
        context=32,
        min_per_genre=3,
        max_per_genre=10000,
        **options,
    )


def test_sqlite_sequences_are_lazy_portable_and_preserve_windows(selected, monkeypatch):
    prepared = _prepare(selected)
    windows = prepared["splits"]["train"]
    assert isinstance(windows, SQLiteWindows)
    actual = list(windows)
    assert actual == windows[:] == windows[0 : len(windows)]
    assert windows[-1] == actual[-1]
    assert copy.deepcopy(windows)[0] == actual[0]
    with pytest.raises(IndexError):
        windows[len(windows)]
    assert prepared["metadata"]["program_counts_train"] == {
        genre: {str(index): 20} for index, genre in enumerate(GENRES)
    }
    assert prepared["report"]["retained_files"] == 9
    assert len(windows.take_per_genre(2)) == 6

    def unexpected_encoding(*args, **kwargs):
        raise AssertionError("Valid SQLite caches must not re-encode MIDI sources")

    monkeypatch.setattr(preprocessing, "encode_midi", unexpected_encoding)
    reused = _prepare(selected)
    assert reused["dataset_sha256"] == prepared["dataset_sha256"]
    assert list(reused["splits"]["train"]) == actual
    for split in prepared["splits"].values():
        split.close()


def test_sqlite_reuse_rejects_database_and_descriptor_tampering(selected):
    prepared = _prepare(selected)
    database = prepared["splits"]["train"].database
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE windows SET payload=? WHERE split='train' AND idx=0", (b"bad",))
    with pytest.raises(ValueError, match="database checksum changed"):
        _prepare(selected)
    metadata = database.parent / "metadata.json"
    descriptor = json.loads(metadata.read_text())
    descriptor["report"]["retained_files"] = 100
    metadata.write_text(json.dumps(descriptor))
    with pytest.raises(ValueError, match="metadata checksum changed"):
        _prepare(selected)


def test_sqlite_reuse_revalidates_original_source_hash(selected):
    _prepare(selected)
    source = selected["root"] / selected["rows"][0]["midi_path"]
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Source hash mismatch"):
        _prepare(selected)


def test_sqlite_interrupted_build_has_no_published_or_temporary_cache(selected, monkeypatch):
    encode = preprocessing.encode_midi
    calls = 0

    def interrupt_after_first_file(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("Interrupted preprocessing")
        return encode(*args, **kwargs)

    monkeypatch.setattr(preprocessing, "encode_midi", interrupt_after_first_file)
    with pytest.raises(KeyboardInterrupt, match="Interrupted preprocessing"):
        _prepare(selected)
    assert not list(selected["cache"].iterdir())


def test_sqlite_rejected_source_is_reported_without_partial_publication(selected, monkeypatch):
    encode = preprocessing.encode_midi

    def reject_source(path, **options):
        if path.name == "classical_train.mid":
            raise ValueError("Deliberately unusable fixture")
        return encode(path, **options)

    monkeypatch.setattr(preprocessing, "encode_midi", reject_source)
    with pytest.raises(ValueError, match="retained 2 files"):
        _prepare(selected)
    report = json.loads((selected["cache"] / "preparation_report.json").read_text())
    assert report["retained_files"] == 8
    assert report["rejected_files"][0]["candidate_id"] == "classical_train"
    assert not list(selected["cache"].glob("*.sqlite"))
    assert not list(selected["cache"].glob(".sqlite-build-*"))


def test_public_sqlite_preparation_matches_json_and_samples_without_payload_reads(
    selected, monkeypatch
):
    pytest.importorskip("torch")
    from musicgen.training import training

    options = {"context": 32, "min_per_genre": 3, "max_per_genre": 10000}
    baseline = training.prepare_dataset(
        selected["manifest"], selected["root"], selected["root"] / "json", **options
    )
    disk = training.prepare_dataset(
        selected["manifest"], selected["root"], selected["cache"], storage="sqlite", **options
    )
    for split in ("train", "val", "test"):
        assert list(disk["splits"][split]) == baseline["splits"][split]
    config = SimpleNamespace(seed=42, max_windows_per_file=2, progress_bins=0)
    expected = training._epoch_indices(baseline["splits"]["train"], config, 0)

    def no_payload_reads(*args, **kwargs):
        raise AssertionError("Epoch sampling must use only SQLite metadata")

    monkeypatch.setattr(SQLiteWindows, "_decode", no_payload_reads)
    assert training._epoch_indices(disk["splits"]["train"], config, 0) == expected
    assert training._epoch_indices(disk["splits"]["train"], config, 0) == expected
    config.progress_bins = 16
    selected_indices = training._epoch_indices(disk["splits"]["train"], config, 0)
    endings = {indices[-1] for _, indices in disk["splits"]["train"].file_indices()}
    assert endings <= set(selected_indices)


def test_sqlite_preserves_new_progress_conditioning_without_changing_legacy(selected):
    legacy = _prepare(selected)
    progress = _prepare(selected, normalize_tempo=True, progress_bins=16)
    assert legacy["dataset_sha256"] != progress["dataset_sha256"]
    assert "progress_ids" not in legacy["splits"]["train"][0]
    window = progress["splits"]["train"][0]
    assert len(window["progress_ids"]) == len(window["input_ids"]) == 32
    assert all(0 <= value < 16 for value in window["progress_ids"])
    assert progress["identity"]["progress_bins"] == 16
