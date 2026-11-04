import csv

import pytest

from musicgen.generation.runtime.tokenizer import GENRES
from musicgen.training.manifest import (
    audit_manifest,
    build_review_candidates,
    digest,
    load_approved,
    source_path,
)


def write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def selection(tmp_path):
    rows = []
    for genre_id, genre in enumerate(GENRES):
        for split in ("train", "val", "test"):
            source = tmp_path / "mid" / f"{genre}_{split}.mid"
            source.parent.mkdir(exist_ok=True)
            source.write_bytes(b"MThd" + f"{genre}_{split}".encode())
            rows.append(
                {
                    "candidate_id": f"{genre}_{split}",
                    "sha256": digest(source),
                    "midi_path": str(source.relative_to(tmp_path)),
                    "genre": genre,
                    "genre_id": str(genre_id),
                    "composition_group": f"{genre}_{split}",
                    "split": split,
                    "training_status": "APPROVED",
                    "approved_for_training": "true",
                    "genre_status": "PROVISIONAL",
                    "rights_status": "UNVERIFIED",
                    "copyright_clearance_confirmed": "false",
                }
            )
    manifest = tmp_path / "selected.csv"
    write_manifest(manifest, rows)
    return tmp_path, manifest, rows


def test_manifest_audit_preserves_decisions_and_reports_counts(selection):
    root, manifest, rows = selection
    audited = load_approved(manifest, midi_root=root, min_per_genre=3, max_per_genre=3)
    assert audited == rows
    report = audit_manifest(manifest, root, min_per_genre=3, max_per_genre=3)
    assert report["rows"] == report["verified_files"] == 9
    assert report["by_genre"] == dict.fromkeys(GENRES, 3)
    assert report["by_genre_split"]["romantic"] == {"train": 1, "val": 1, "test": 1}
    assert report["genre_status_counts"] == {"PROVISIONAL": 9}
    assert report["rights_status_counts"] == {"UNVERIFIED": 9}
    assert report["manifest_sha256"] == digest(manifest)
    assert report["verified_bytes"] > 0
    assert report["warnings"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_id", " ", "Empty required"),
        ("genre", "jazz", "Invalid genre"),
        ("genre_id", "2", "genre_id"),
        ("split", "validation", "Invalid genre or split"),
        ("sha256", "abc", "Invalid source SHA"),
        ("composition_group", "", "Empty required"),
        ("approved_for_training", "false", "not approved"),
        ("training_status", "REVIEW_REQUIRED", "not approved"),
    ],
)
def test_manifest_rejects_incomplete_or_invalid_decisions(selection, field, value, message):
    root, manifest, rows = selection
    rows[0][field] = value
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match=message):
        load_approved(manifest, midi_root=root, min_per_genre=3, max_per_genre=3)


@pytest.mark.parametrize("field", ["candidate_id", "sha256", "midi_path"])
def test_manifest_rejects_duplicate_sources(selection, field):
    _, manifest, rows = selection
    rows[1][field] = rows[0][field]
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match=f"Duplicate {field}"):
        load_approved(manifest, min_per_genre=3, max_per_genre=3)


@pytest.mark.parametrize("other_row", [1, 3])
def test_manifest_rejects_groups_crossing_splits_or_genres(selection, other_row):
    _, manifest, rows = selection
    rows[other_row]["composition_group"] = rows[0]["composition_group"]
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="Composition group crosses"):
        load_approved(manifest, min_per_genre=3, max_per_genre=3)


def test_manifest_requires_all_genre_splits_and_count_bounds(selection):
    _, manifest, rows = selection
    with pytest.raises(ValueError, match="Expected 200"):
        load_approved(manifest)
    rows[0]["split"] = "val"
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="No selected files for classical/train"):
        load_approved(manifest, min_per_genre=3, max_per_genre=3)


def test_manifest_rejects_changed_or_missing_source(selection):
    root, manifest, rows = selection
    source = root / rows[0]["midi_path"]
    source.write_bytes(b"MThdchanged")
    with pytest.raises(ValueError, match="Source hash mismatch"):
        load_approved(manifest, midi_root=root, min_per_genre=3, max_per_genre=3)
    source.unlink()
    with pytest.raises(ValueError, match="Missing MIDI file"):
        load_approved(manifest, midi_root=root, min_per_genre=3, max_per_genre=3)


@pytest.mark.parametrize(
    "relative", ["../escape.mid", "/tmp/escape.mid", "C:\\escape.mid", "x.csv"]
)
def test_source_path_rejects_unsafe_paths(tmp_path, relative):
    with pytest.raises(ValueError):
        source_path(tmp_path, relative)


def test_source_path_rejects_escaping_symlink(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.mid"
    outside.write_bytes(b"MThd")
    (root / "link.mid").symlink_to(outside)
    with pytest.raises(ValueError, match="outside dataset root"):
        source_path(root, "link.mid")


def test_manifest_rejects_missing_columns_and_invalid_midi_header(selection):
    root, manifest, rows = selection
    for row in rows:
        del row["genre_id"]
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="missing columns: genre_id"):
        load_approved(manifest, min_per_genre=3, max_per_genre=3)
    for row in rows:
        row["genre_id"] = str(GENRES.index(row["genre"]))
    source = root / rows[0]["midi_path"]
    source.write_bytes(b"not midi")
    rows[0]["sha256"] = digest(source)
    write_manifest(manifest, rows)
    with pytest.raises(ValueError, match="no MIDI header"):
        load_approved(manifest, midi_root=root, min_per_genre=3, max_per_genre=3)


def test_review_shortlist_is_deterministic_and_never_approved(tmp_path):
    rows = []
    for genre in ("classical", "pop", "rock"):
        for index in range(3):
            source = tmp_path / f"{genre}_{index}.mid"
            source.write_bytes(b"MThd" + f"{genre}_{index}".encode())
            rows.append(
                {
                    "mid": "./" + source.name,
                    "genres": genre,
                    "license_conflict": "False",
                    "is_best_unique_arrangement": "True",
                    "best_unique_arrangement": source.stem + ".json",
                    "title": f"Title {genre} {index}",
                }
            )
    pdmx = tmp_path / "pdmx.csv"
    write_manifest(pdmx, rows)
    first, second = tmp_path / "one.csv", tmp_path / "two.csv"
    report = build_review_candidates(pdmx, tmp_path, first, per_bucket=2, seed=42)
    write_manifest(pdmx, list(reversed(rows)))
    repeated = build_review_candidates(pdmx, tmp_path, second, per_bucket=2, seed=42)
    assert digest(first) == digest(second) == report["output_sha256"]
    assert report["by_suggested_genre"] == {"classical_or_romantic": 2, "rock_pop": 2}
    assert repeated["approved_for_training"] == 0
    with first.open(newline="") as handle:
        candidates = list(csv.DictReader(handle))
    assert all(row["approved_for_training"] == "false" for row in candidates)
    assert all(not row["genre"] and not row["genre_id"] and not row["split"] for row in candidates)
    with pytest.raises(ValueError):
        load_approved(first, midi_root=tmp_path, min_per_genre=1, max_per_genre=250)
    with pytest.raises(FileExistsError):
        build_review_candidates(pdmx, tmp_path, first)
