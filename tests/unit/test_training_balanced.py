"""Balanced selection preserves holdouts and replenishes unusable candidates."""

import csv
import json
from collections import Counter, defaultdict

import pytest

from musicgen.generation.runtime.midi import Note, write_midi
from musicgen.training import balanced
from musicgen.training.manifest import digest


def _selection(tmp_path, counts, *, duplicate_families=False):
    rows, identity = [], 0
    for genre_id, genre in enumerate(("classical", "romantic", "rock_pop")):
        for split, count in counts.items():
            for index in range(count):
                name = f"{genre}_{split}_{index}"
                path = tmp_path / "mid" / f"{name}.mid"
                write_midi(path, [Note(0, 1 + identity / 480, 60 + genre_id, 0)])
                identity += 1
                family = index // 2 if duplicate_families else index
                rows.append(
                    {
                        "candidate_id": name,
                        "sha256": digest(path),
                        "midi_path": str(path.relative_to(tmp_path)),
                        "genre": genre,
                        "genre_id": str(genre_id),
                        "composition_group": f"{genre}_{split}_family{family}",
                        "split": split,
                        "training_status": "APPROVED",
                        "approved_for_training": "true",
                        "canonical_composer": f"composer{index % 5}" if genre_id < 2 else "",
                        "composer": f"Composer {index % 5}",
                        "artist": f"Band {index % 7}",
                        "genre_status": "PROVISIONAL",
                        "rights_status": "UNVERIFIED",
                        "rules_version": "cadence-pdmx-expanded-v3",
                        "baseline_groups": f"original_{genre}_{split}_{family}",
                        "baseline_splits": split,
                        "custom_provenance": f"retained metadata {name}",
                    }
                )
    path = tmp_path / "curated.csv"
    _write(path, rows)
    return path, rows


def _write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_defaults_select_exact_750_and_preserve_all_provenance(tmp_path, monkeypatch):
    manifest, source = _selection(tmp_path, {"train": 177, "val": 52, "test": 27})
    calls = 0
    original_encode = balanced.preprocessing.encode_midi

    def encode(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert kwargs == {"normalize_tempo": True, "progress_bins": 16}
        return original_encode(*args, **kwargs)

    monkeypatch.setattr(balanced.preprocessing, "encode_midi", encode)
    output = tmp_path / "balanced"
    report = balanced.build_balanced_manifest(manifest, tmp_path, output)
    selected = _read(output / "selected.csv")
    expected = {
        (genre, split): count
        for genre in ("classical", "romantic", "rock_pop")
        for split, count in {"train": 175, "val": 50, "test": 25}.items()
    }
    assert Counter((row["genre"], row["split"]) for row in selected) == expected
    assert len(selected) == calls == report["encoded_candidates"] == 750
    assert calls < len(source)
    assert len({row["sha256"] for row in selected}) == 750
    assert len({row["composition_group"] for row in selected}) == 750
    original = {row["candidate_id"]: row for row in source}
    for row in selected:
        assert all(row[key] == value for key, value in original[row["candidate_id"]].items())
        assert row["selected_from_manifest_sha256"] == digest(manifest)
        assert row["selection_policy"] == "cadence-balanced250-v1"
        assert row["baseline_splits"] == row["split"]
    assert all(row["true_piece_endings"] >= 1 for row in report["file_reports"])
    assert report["selected_manifest_sha256"] == digest(output / "selected.csv")
    assert json.loads((output / "report.json").read_text()) == report
    with pytest.raises(FileExistsError):
        balanced.build_balanced_manifest(manifest, tmp_path, output)


def test_selection_is_deterministic_and_round_robin_diverse(tmp_path):
    manifest, _ = _selection(tmp_path, {"train": 10, "val": 10, "test": 10})
    options = {"per_genre": 15, "split_counts": {"train": 5, "val": 5, "test": 5}, "seed": 42}
    first = balanced.build_balanced_manifest(manifest, tmp_path, tmp_path / "one", **options)
    second = balanced.build_balanced_manifest(manifest, tmp_path, tmp_path / "two", **options)
    assert first == second
    assert (tmp_path / "one/selected.csv").read_bytes() == (
        tmp_path / "two/selected.csv"
    ).read_bytes()
    by_bucket = defaultdict(Counter)
    for row in _read(tmp_path / "one/selected.csv"):
        by_bucket[row["genre"], row["split"]][row["selection_stratum"]] += 1
    assert all(len(counts) == 5 and set(counts.values()) == {1} for counts in by_bucket.values())


def test_only_one_file_per_family_is_selected(tmp_path):
    manifest, _ = _selection(tmp_path, {"train": 8, "val": 8, "test": 8}, duplicate_families=True)
    report = balanced.build_balanced_manifest(
        manifest,
        tmp_path,
        tmp_path / "balanced",
        per_genre=9,
        split_counts={"train": 3, "val": 3, "test": 3},
    )
    selected = _read(tmp_path / "balanced/selected.csv")
    assert len(selected) == len({row["composition_group"] for row in selected}) == 27
    assert report["duplicate_family_candidates_skipped"] >= 1


def test_fresh_encoding_rejection_is_replenished_within_its_original_split(tmp_path, monkeypatch):
    manifest, _ = _selection(tmp_path, {"train": 5, "val": 5, "test": 5})
    original = balanced.preprocessing.encode_midi
    calls = 0

    def reject_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("Deliberately unusable source fixture")
        return original(*args, **kwargs)

    monkeypatch.setattr(balanced.preprocessing, "encode_midi", reject_first)
    report = balanced.build_balanced_manifest(
        manifest,
        tmp_path,
        tmp_path / "balanced",
        per_genre=6,
        split_counts={"train": 2, "val": 2, "test": 2},
    )
    assert report["selected_files"] == 18
    assert report["encoded_candidates"] == 19
    rejected = report["rejected_candidates"][0]
    assert (rejected["genre"], rejected["split"], rejected["stage"]) == (
        "classical",
        "train",
        "fresh_encoding",
    )
    assert rejected["candidate_id"] not in {
        row["candidate_id"] for row in _read(tmp_path / "balanced/selected.csv")
    }


def test_known_empty_midi_is_excluded_and_replaced_without_reencoding(tmp_path, monkeypatch):
    manifest, rows = _selection(tmp_path, {"train": 5, "val": 5, "test": 5})
    bad = rows[0]
    path = tmp_path / bad["midi_path"]
    write_midi(path, [])
    bad["sha256"] = digest(path)
    _write(manifest, rows)
    known = tmp_path / "metadata.json"
    known.write_text(
        json.dumps(
            {
                "identity": {"manifest_sha256": digest(manifest)},
                "report": {
                    "rejected_files": [{"candidate_id": bad["candidate_id"], "error": "No notes"}]
                },
            }
        )
    )
    original = balanced.preprocessing.encode_midi

    def encode(candidate, **kwargs):
        assert candidate != path
        return original(candidate, **kwargs)

    monkeypatch.setattr(balanced.preprocessing, "encode_midi", encode)
    report = balanced.build_balanced_manifest(
        manifest,
        tmp_path,
        tmp_path / "balanced",
        preparation_report=known,
        per_genre=6,
        split_counts={"train": 2, "val": 2, "test": 2},
    )
    assert report["selected_files"] == 18
    assert report["rejected_candidates"][0]["stage"] == "known_preparation_rejection"
    assert bad["candidate_id"] not in {
        row["candidate_id"] for row in _read(tmp_path / "balanced/selected.csv")
    }


def test_partial_sequences_without_a_piece_ending_are_replenished(tmp_path, monkeypatch):
    manifest, _ = _selection(tmp_path, {"train": 4, "val": 4, "test": 4})
    original = balanced.preprocessing.encode_midi
    calls = 0

    def missing_ending(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 1:
            for sequence in result["sequences"]:
                sequence["is_piece_end"] = False
        return result

    monkeypatch.setattr(balanced.preprocessing, "encode_midi", missing_ending)
    report = balanced.build_balanced_manifest(
        manifest,
        tmp_path,
        tmp_path / "balanced",
        per_genre=3,
        split_counts={"train": 1, "val": 1, "test": 1},
    )
    assert report["selected_files"] == 9
    assert report["rejected_candidates"][0]["reason"] == "No retained true piece-ending sequence"


def test_insufficient_families_fail_without_publishing_output(tmp_path):
    manifest, _ = _selection(tmp_path, {"train": 2, "val": 2, "test": 2}, duplicate_families=True)
    with pytest.raises(ValueError, match="Insufficient usable composition families"):
        balanced.build_balanced_manifest(
            manifest,
            tmp_path,
            tmp_path / "balanced",
            per_genre=6,
            split_counts={"train": 2, "val": 2, "test": 2},
        )
    assert not (tmp_path / "balanced").exists()
    assert not list(tmp_path.glob(".balanced-manifest-*"))


def test_changed_source_hash_fails_before_selection(tmp_path):
    manifest, rows = _selection(tmp_path, {"train": 2, "val": 2, "test": 2})
    path = tmp_path / rows[0]["midi_path"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Source hash mismatch"):
        balanced.build_balanced_manifest(
            manifest,
            tmp_path,
            tmp_path / "balanced",
            per_genre=3,
            split_counts={"train": 1, "val": 1, "test": 1},
        )
    assert not (tmp_path / "balanced").exists()


@pytest.mark.parametrize(
    "counts",
    [
        {"train": 2, "val": 1, "test": 0},
        {"train": 1, "val": 1, "test": 2},
        {"train": True, "val": 1, "test": 1},
        {"train": 1, "val": 2},
    ],
)
def test_invalid_split_quotas_are_rejected_before_reading_source(tmp_path, counts):
    with pytest.raises(ValueError, match="split_counts"):
        balanced.build_balanced_manifest(
            "missing.csv", tmp_path, tmp_path / "out", per_genre=3, split_counts=counts
        )
