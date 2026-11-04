"""PDMX expansion keeps provisional labels and protects baseline holdouts."""

import csv
import json
import shutil

import pytest

from musicgen.generation.runtime.midi import Note, write_midi
from musicgen.training.manifest import digest
from musicgen.training.pdmx import build_manifest, classify


def _source(root, name, identity):
    path = root / "mid" / f"{name}.mid"
    write_midi(path, [Note(0, 1 + identity / 480, 60, 0)], end_beat=2 + identity / 480)
    return path


def _row(name, *, composer="Wolfgang Amadeus Mozart", artist=None, genre="classical", work=None):
    return {
        "mid": f"./mid/{name}.mid",
        "path": f"./data/{name}.json",
        "best_path": f"./data/{work or name}.json",
        "best_arrangement": f"./data/{work or name}.json",
        "best_unique_arrangement": f"./data/{work or name}.json",
        "metadata": f"./metadata/{name}.json",
        "genres": genre,
        "composer_name": composer,
        "artist_name": artist if artist is not None else composer,
        "title": f"Distinct work {name}",
        "song_name": f"Distinct work {name}",
        "is_best_unique_arrangement": "True",
        "subset:all_valid": "True",
        "is_draft": "False",
        "license_conflict": "False",
        "license": "cc-zero",
        "license_url": "https://example.org/source-claim",
    }


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize(
    ("composer", "artist", "expected"),
    [
        ("Johann Sebastian Bach", "Johann Sebastian Bach", "classical"),
        ("Carl Philipp Emanuel Bach", "Carl Philipp Emanuel Bach", "classical"),
        ("Mozart", "Wolfgang Amadeus Mozart", "classical"),
        ("Composed by Joseph Haydn", "Joseph Haydn", "classical"),
        ("NA", "Frédéric Chopin", "romantic"),
        ("César Franck", "César Franck", "romantic"),
        ("Melchior Franck", "Melchior Franck", ""),
        ("Franck", "NA", ""),
        ("Claude Debussy", "Claude Debussy", ""),
        ("Ludwig van Beethoven", "Ludwig van Beethoven", ""),
        ("Unknown modern arranger", "Johann Sebastian Bach", ""),
        ("Robert Schumann", "Wolfgang Amadeus Mozart", ""),
        ("Leopold Mozart", "Leopold Mozart", ""),
        ("NA", "Mozart L'opera Rock", ""),
    ],
)
def test_period_evidence_is_conservative_and_person_specific(composer, artist, expected):
    row = _row("example", composer=composer, artist=artist)
    genre, _, _, reason = classify(row)
    assert genre == expected
    assert bool(reason) == (not bool(expected))


def test_period_scope_and_beethoven_policy_are_explicit():
    bach = _row("bach", composer="J. S. Bach")
    assert classify(bach)[0] == "classical"
    assert classify(bach, classical_scope="classical")[3] == "composer_outside_period_policy"
    beethoven = _row("beethoven", composer="Ludwig van Beethoven")
    assert classify(beethoven, beethoven_policy="classical")[0] == "classical"
    assert classify(beethoven, beethoven_policy="romantic")[0] == "romantic"
    with pytest.raises(ValueError):
        classify(bach, classical_scope="anything")


@pytest.mark.parametrize(
    "name",
    [
        "Gabriel Fauré",
        "Sergei Rachmaninoff",
        "Giacomo Puccini",
        "Gustav Mahler",
        "Niccolò Paganini",
        "John Field",
        "Isaac Albéniz",
        "Mikhail Glinka",
        "Alexander Glazunov",
        "Jean Sibelius",
        "Enrique Granados",
        "Gioachino Rossini",
        "Carl Maria von Weber",
        "Gaetano Donizetti",
        "Vincenzo Bellini",
    ],
)
def test_expanded_romantic_composers_have_explicit_evidence(name):
    genre, person, evidence, reason = classify(_row("piece", composer=name))
    assert genre == "romantic" and person and evidence.startswith("composer_alias:")
    assert not reason


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Frédérick Chopin", "romantic"),
        ("Peter Tchaikovsky", "romantic"),
        ("Edward Grieg", "romantic"),
        ("Antonio Lucio Vivaldi", "classical"),
        ("Bach Johann Sebastian", "classical"),
        ("G.F. Handel1685-1759", "classical"),
        ("Antonín DvoÅák", "romantic"),
    ],
)
def test_observed_full_name_variants_are_explicitly_supported(name, expected):
    assert classify(_row("piece", composer=name))[0] == expected


def test_ambiguous_surname_needs_a_corresponding_identified_artist():
    assert classify(_row("unknown", composer="Field", artist="NA"))[0] == ""
    assert classify(_row("unknown", composer="Franck", artist="NA"))[0] == ""
    assert classify(_row("known", composer="Franck", artist="César Franck"))[0] == "romantic"
    assert classify(_row("known", composer="Faure", artist="Gabriel Fauré"))[0] == "romantic"
    assert classify(_row("known", composer="Franck", artist="Melchior Franck"))[0] == ""
    assert classify(_row("review", composer="Maurice Ravel"))[0] == ""
    assert classify(_row("review", composer="Claude Debussy"))[0] == ""


def test_generic_classical_and_mixed_tags_do_not_become_historical_labels():
    assert (
        classify(_row("unknown", composer="NA", artist="Unknown author"))[3]
        == "unresolved_classical_period"
    )
    assert classify(_row("mixed", genre="classical-rock"))[3] == "cross_genre_tags"
    assert classify(_row("pop", genre="pop", composer="New artist"))[0] == "rock_pop"
    assert (
        classify(_row("cover", genre="pop", composer="Franz Schubert"))[3] == "cross_genre_composer"
    )


def test_expansion_has_no_250_file_cap_and_is_reproducible(tmp_path):
    rows = []
    for index in range(251):
        name = f"mozart_{index:03}"
        _source(tmp_path, name, index)
        rows.append(_row(name))
    metadata = tmp_path / "PDMX.csv"
    _csv(metadata, rows)
    first = build_manifest(metadata, tmp_path, tmp_path / "first")
    second = build_manifest(metadata, tmp_path, tmp_path / "second")
    assert first["selected_by_genre"]["classical"] == 251
    assert (tmp_path / "first/expanded.csv").read_bytes() == (
        tmp_path / "second/expanded.csv"
    ).read_bytes()
    assert first == second
    selected = _read(tmp_path / "first/expanded.csv")
    assert {r["split"] for r in selected} == {"train", "val", "test"}
    assert all(
        r["genre_status"] == "PROVISIONAL" and r["rights_status"] == "UNVERIFIED" for r in selected
    )
    assert all(r["approved_for_training"] == "true" for r in selected)
    assert first["expanded_manifest_sha256"] == digest(tmp_path / "first/expanded.csv")
    with pytest.raises(FileExistsError):
        build_manifest(metadata, tmp_path, tmp_path / "first")


@pytest.fixture
def baseline(tmp_path):
    rows, metadata = [], []
    composers = ("Wolfgang Amadeus Mozart", "Frédéric Chopin", "Modern Pop Artist")
    for genre_id, genre in enumerate(("classical", "romantic", "rock_pop")):
        for index, split in enumerate(("train", "val", "test")):
            name = f"{genre}_{split}"
            path = _source(tmp_path, name, genre_id * 100 + index)
            rows.append(
                {
                    "candidate_id": name,
                    "sha256": digest(path),
                    "midi_path": str(path.relative_to(tmp_path)),
                    "genre": genre,
                    "genre_id": str(genre_id),
                    "composition_group": name,
                    "split": split,
                    "training_status": "APPROVED",
                    "approved_for_training": "true",
                    "composer": composers[genre_id],
                    "artist": composers[genre_id],
                    "title": f"Distinct work {name}",
                }
            )
            metadata.append(
                _row(
                    name,
                    composer=composers[genre_id],
                    genre="pop" if genre == "rock_pop" else "classical",
                )
            )
    path = tmp_path / "baseline.csv"
    _csv(path, rows)
    return path, rows, metadata


def test_baseline_holdouts_propagate_to_related_pdmx_and_title_families(tmp_path, baseline):
    original, _, rows = baseline
    _source(tmp_path, "alternate_arrangement", 500)
    rows.append(_row("alternate_arrangement", work="classical_val"))
    _source(tmp_path, "alternate_title_source", 501)
    titled = _row("alternate_title_source")
    titled["title"] = "Distinct work romantic_test"
    titled["composer_name"] = titled["artist_name"] = "Frédéric Chopin"
    rows.append(titled)
    # Byte-identical MIDI under a new path must not bypass the original test split.
    copied = tmp_path / "mid/copied_test.mid"
    shutil.copyfile(tmp_path / "mid/classical_test.mid", copied)
    rows.append(_row("copied_test"))
    metadata = tmp_path / "PDMX.csv"
    _csv(metadata, rows)
    report = build_manifest(metadata, tmp_path, tmp_path / "expanded", baseline_manifest=original)
    selected = {r["midi_path"]: r for r in _read(tmp_path / "expanded/expanded.csv")}
    assert selected["mid/alternate_arrangement.mid"]["split"] == "val"
    assert selected["mid/alternate_title_source.mid"]["split"] == "test"
    assert selected["mid/classical_test.mid"]["split"] == "test"
    assert "mid/copied_test.mid" not in selected
    assert report["review_reasons"]["duplicate_source_hash"] == 1
    for row in selected.values():
        assert row["split_origin"] == "baseline_family"
    assert selected["mid/alternate_arrangement.mid"]["baseline_groups"] == "classical_val"


def test_conflicting_baseline_splits_quarantine_entire_related_family(tmp_path, baseline):
    original, _, rows = baseline
    # Source IDs reveal these previously separate train/test files are related.
    for row in rows:
        if row["mid"] in {"./mid/classical_train.mid", "./mid/classical_test.mid"}:
            row["best_unique_arrangement"] = "./data/shared_work.json"
    _source(tmp_path, "new_related", 600)
    rows.append(_row("new_related", work="shared_work"))
    metadata = tmp_path / "PDMX.csv"
    _csv(metadata, rows)
    report = build_manifest(metadata, tmp_path, tmp_path / "expanded", baseline_manifest=original)
    assert report["review_reasons"]["baseline_split_conflict"] == 3
    selected = _read(tmp_path / "expanded/expanded.csv")
    assert all(
        row["midi_path"]
        not in {"mid/classical_train.mid", "mid/classical_test.mid", "mid/new_related.mid"}
        for row in selected
    )
    quarantine = report["quarantined_families"][0]
    assert quarantine["baseline_splits"] == ["test", "train"]


def test_source_and_metadata_failures_are_reported_without_approving(tmp_path):
    rows = [
        _row("missing"),
        _row("broken"),
        _row("unmatched", composer="NA", artist="NA"),
        _row("not_best"),
        _row("license_conflict"),
        _row("draft"),
    ]
    (tmp_path / "mid").mkdir()
    (tmp_path / "mid/broken.mid").write_bytes(b"not-midi")
    rows[3]["is_best_unique_arrangement"] = "False"
    rows[4]["license_conflict"] = "True"
    rows[5]["is_draft"] = "True"
    metadata = tmp_path / "PDMX.csv"
    _csv(metadata, rows)
    report = build_manifest(metadata, tmp_path, tmp_path / "expanded")
    assert report["counts"]["source_errors"] == 2
    assert report["counts"]["metadata_gate_excluded"] == 3
    assert report["review_reasons"]["unresolved_classical_period"] == 1
    assert _read(tmp_path / "expanded/expanded.csv") == []
    assert all(
        row["approved_for_training"] == "false" for row in _read(tmp_path / "expanded/review.csv")
    )
    assert json.loads((tmp_path / "expanded/report.json").read_text()) == report


def test_broad_pdmx_clusters_do_not_merge_unrelated_composers(tmp_path):
    rows = [
        _row("mozart"),
        _row("chopin", composer="Frédéric Chopin"),
    ]
    for index, row in enumerate(rows):
        _source(tmp_path, ("mozart", "chopin")[index], index)
        row["best_path"] = row["best_arrangement"] = "./data/overbroad_cluster.json"
    metadata = tmp_path / "PDMX.csv"
    _csv(metadata, rows)
    report = build_manifest(metadata, tmp_path, tmp_path / "expanded")
    assert report["counts"]["selected_files"] == 2
    assert report["quarantined_families"] == []
    groups = {row["composition_group"] for row in _read(tmp_path / "expanded/expanded.csv")}
    assert len(groups) == 2


def test_failed_input_does_not_publish_partial_output(tmp_path):
    source = tmp_path / "broken.csv"
    source.write_text("mid,genres\nexample.mid,classical\n")
    with pytest.raises(ValueError, match="missing required"):
        build_manifest(source, tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".pdmx-manifest-*"))
