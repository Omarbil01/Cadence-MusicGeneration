"""Build an expanded, provisional-label PDMX training manifest from local MIDI.

Training inclusion records the project's authorization to expand its dataset.
Composer and source-tag matches remain provisional genre evidence, and PDMX
license claims are preserved without declaring source rights verified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import struct
import tempfile
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from musicgen.generation.runtime.tokenizer import GENRES
from musicgen.training.manifest import digest, load_approved, source_path

RULES_VERSION = "cadence-pdmx-expanded-v3"

# Full-person aliases take precedence over surname aliases. A one-word alias
# matches an entire attribution field, never an arbitrary substring in it.
COMPOSERS = {
    "mozart_wolfgang_amadeus": (
        "classical",
        "Wolfgang Amadeus Mozart",
        "W A Mozart",
        "W A Mozart",
        "Mozart",
    ),
    "haydn_joseph": ("classical", "Joseph Haydn", "Franz Joseph Haydn", "J Haydn", "Haydn"),
    "haydn_michael": ("classical", "Johann Michael Haydn", "Michael Haydn", "M Haydn"),
    "bach_carl_philipp_emanuel": (
        "classical",
        "Carl Philipp Emanuel Bach",
        "C P E Bach",
        "CPE Bach",
    ),
    "bach_johann_christian": ("classical", "Johann Christian Bach", "J C Bach"),
    "clementi_muzio": ("classical", "Muzio Clementi", "M Clementi", "Clementi"),
    "boccherini_luigi": ("classical", "Luigi Boccherini", "L Boccherini", "Boccherini"),
    "gluck_christoph_willibald": (
        "classical",
        "Christoph Willibald Gluck",
        "Christof Willibald Gluck",
        "C W Gluck",
        "Gluck",
    ),
    "salieri_antonio": ("classical", "Antonio Salieri", "A Salieri", "Salieri"),
    "dittersdorf_carl": ("classical", "Carl Ditters von Dittersdorf", "Dittersdorf"),
    "stamitz_carl": ("classical", "Carl Philipp Stamitz", "Carl Stamitz", "Karl Stamitz"),
    "pleyel_ignaz": ("classical", "Ignaz Pleyel", "Ignace Pleyel", "I Pleyel", "Pleyel"),
    "bach_johann_sebastian": ("baroque", "Johann Sebastian Bach", "J S Bach", "JS Bach", "Bach"),
    "handel_george_frideric": (
        "baroque",
        "George Frideric Handel",
        "Georg Friedrich Handel",
        "G F Handel",
        "Handel",
    ),
    "vivaldi_antonio": ("baroque", "Antonio Vivaldi", "A Vivaldi", "Vivaldi"),
    "corelli_arcangelo": ("baroque", "Arcangelo Corelli", "A Corelli", "Corelli"),
    "telemann_georg_philipp": ("baroque", "Georg Philipp Telemann", "G P Telemann", "Telemann"),
    "scarlatti_domenico": ("baroque", "Domenico Scarlatti", "D Scarlatti"),
    "scarlatti_alessandro": ("baroque", "Alessandro Scarlatti", "A Scarlatti"),
    "purcell_henry": ("baroque", "Henry Purcell", "H Purcell", "Purcell"),
    "couperin_francois": ("baroque", "Francois Couperin", "F Couperin"),
    "rameau_jean_philippe": ("baroque", "Jean Philippe Rameau", "J P Rameau", "Rameau"),
    "pachelbel_johann": ("baroque", "Johann Pachelbel", "J Pachelbel", "Pachelbel"),
    "chopin_frederic": ("romantic", "Frederic Chopin", "Fryderyk Chopin", "F Chopin", "Chopin"),
    "schubert_franz": ("romantic", "Franz Schubert", "F Schubert", "Schubert"),
    "schumann_robert": ("romantic", "Robert Schumann", "R Schumann", "Schumann"),
    "schumann_clara": ("romantic", "Clara Schumann", "Clara Wieck", "C Schumann"),
    "liszt_franz": ("romantic", "Franz Liszt", "F Liszt", "Liszt"),
    "brahms_johannes": ("romantic", "Johannes Brahms", "J Brahms", "Brahms"),
    "mendelssohn_felix": (
        "romantic",
        "Felix Mendelssohn",
        "Felix Mendelssohn Bartholdy",
        "F Mendelssohn",
        "Mendelssohn",
    ),
    "tchaikovsky_pyotr": (
        "romantic",
        "Pyotr Ilyich Tchaikovsky",
        "Peter Ilyich Tchaikovsky",
        "Piotr Ilitch Tchaikovsky",
        "P I Tchaikovsky",
        "Tchaikovsky",
        "Chaykovsky",
    ),
    "grieg_edvard": ("romantic", "Edvard Grieg", "E Grieg", "Grieg"),
    "dvorak_antonin": ("romantic", "Antonin Dvorak", "A Dvorak", "Dvorak"),
    "smetana_bedrich": ("romantic", "Bedrich Smetana", "B Smetana", "Smetana"),
    "berlioz_hector": ("romantic", "Hector Berlioz", "H Berlioz", "Berlioz"),
    "bruckner_anton": ("romantic", "Anton Bruckner", "A Bruckner", "Bruckner"),
    "wagner_richard": ("romantic", "Richard Wagner", "R Wagner", "Wagner"),
    "verdi_giuseppe": ("romantic", "Giuseppe Verdi", "G Verdi", "Verdi"),
    "bizet_georges": ("romantic", "Georges Bizet", "G Bizet", "Bizet"),
    "saint_saens_camille": ("romantic", "Camille Saint Saens", "C Saint Saens", "Saint Saens"),
    "rimsky_korsakov_nikolai": (
        "romantic",
        "Nikolai Rimsky Korsakov",
        "Nikolay Rimsky Korsakov",
        "Rimsky Korsakov",
    ),
    "borodin_alexander": ("romantic", "Alexander Borodin", "Aleksandr Borodin", "Borodin"),
    "mussorgsky_modest": ("romantic", "Modest Mussorgsky", "Mussorgsky"),
    "balakirev_mily": ("romantic", "Mily Balakirev", "Mili Balakirev", "Balakirev"),
    "franck_cesar": ("romantic", "Cesar Franck", "C Franck"),
    "gounod_charles": ("romantic", "Charles Gounod", "Gounod"),
    "beethoven_ludwig": ("transitional", "Ludwig van Beethoven", "L van Beethoven", "Beethoven"),
    "debussy_claude": ("review", "Claude Debussy", "Debussy"),
    "ravel_maurice": ("review", "Maurice Ravel", "Ravel"),
    "franck_melchior": ("review", "Melchior Franck"),
    "mozart_leopold": ("review", "Leopold Mozart"),
    "mozart_franz_xaver": ("review", "Franz Xaver Wolfgang Mozart", "Franz Xaver Mozart"),
}

# Additional full-name spellings observed in the source, and established
# Romantic/late-Romantic composers. Debussy and Ravel remain review entries:
# this policy does not silently merge Impressionism into the Romantic label.
_VARIANTS = {
    "haydn_joseph": ("F J Haydn", "Franz Josef Haydn"),
    "bach_johann_sebastian": ("Bach Johann Sebastian", "JSBach"),
    "handel_george_frideric": (
        "George Frederic Handel",
        "Georg Friedrich Haendel",
        "George F Handel",
        "Handel George F",
        "Handel George Frideric",
        "Georg Frideric Handel",
    ),
    "vivaldi_antonio": ("Antonio Lucio Vivaldi",),
    "chopin_frederic": (
        "Frederick Chopin",
        "Frederic Francois Chopin",
        "Fryderyk Franciszek Chopin",
    ),
    "tchaikovsky_pyotr": (
        "Peter Tchaikovsky",
        "P I Tsjaikovski",
        "Peter Iljitsch Tschaikowsky",
        "Pyotr Ilyich Tschaikovsky",
        "Pjotr Iljitsch Tschaikowski",
        "P I Tschaikovsky",
    ),
    "grieg_edvard": ("Edward Grieg", "Edvard Hagerup Grieg"),
    "dvorak_antonin": ("Antonín DvoÅák",),
    "smetana_bedrich": ("BedÅich Smetana",),
    "mussorgsky_modest": ("Modest Petrowitsch Mussorgski", "Modest Petrovich Mussorgsky"),
    "gounod_charles": ("Ch Gounod",),
}
for _person, _names in _VARIANTS.items():
    COMPOSERS[_person] += _names
COMPOSERS.update(
    {
        "faure_gabriel": ("romantic", "Gabriel Faure", "Gabriel Urbain Faure", "G Faure"),
        "rachmaninoff_sergei": (
            "romantic",
            "Sergei Rachmaninoff",
            "Sergey Rachmaninoff",
            "Sergei Rachmaninov",
            "Sergey Rachmaninov",
            "Sergei Vasilyevich Rachmaninoff",
            "S Rachmaninoff",
            "S Rachmaninov",
            "Rachmaninoff",
            "Rachmaninov",
        ),
        "puccini_giacomo": ("romantic", "Giacomo Puccini", "G Puccini", "Puccini"),
        "mahler_gustav": ("romantic", "Gustav Mahler", "G Mahler", "Mahler"),
        "paganini_niccolo": ("romantic", "Niccolo Paganini", "N Paganini", "Paganini"),
        "field_john": ("romantic", "John Field", "J Field"),
        "albeniz_isaac": ("romantic", "Isaac Albeniz", "I Albeniz", "Albeniz"),
        "granados_enrique": ("romantic", "Enrique Granados", "E Granados", "Granados"),
        "glinka_mikhail": (
            "romantic",
            "Mikhail Glinka",
            "Mikhail Ivanovich Glinka",
            "M Glinka",
            "Glinka",
        ),
        "glazunov_alexander": (
            "romantic",
            "Alexander Glazunov",
            "Aleksandr Glazunov",
            "Alexander Konstantinovich Glazunov",
            "A Glazunov",
            "Glazunov",
        ),
        "weber_carl_maria": (
            "romantic",
            "Carl Maria von Weber",
            "Carl Maria Weber",
            "C M von Weber",
        ),
        "rossini_gioachino": (
            "romantic",
            "Gioachino Rossini",
            "Gioacchino Rossini",
            "G Rossini",
            "Rossini",
        ),
        "donizetti_gaetano": ("romantic", "Gaetano Donizetti", "G Donizetti", "Donizetti"),
        "bellini_vincenzo": ("romantic", "Vincenzo Bellini", "V Bellini", "Bellini"),
        "rubinstein_anton": ("romantic", "Anton Rubinstein", "A Rubinstein"),
        "rubinstein_nikolai": (
            "romantic",
            "Nikolai Rubinstein",
            "Nikolay Rubinstein",
            "N Rubinstein",
        ),
        "sibelius_jean": (
            "romantic",
            "Jean Sibelius",
            "Johan Julius Christian Sibelius",
            "J Sibelius",
            "Sibelius",
        ),
    }
)

FIELDS = (
    "candidate_id",
    "sha256",
    "midi_path",
    "genre",
    "genre_id",
    "composition_group",
    "split",
    "training_status",
    "approved_for_training",
    "source",
    "source_csv_sha256",
    "title",
    "song_name",
    "composer",
    "artist",
    "canonical_composer",
    "source_genres",
    "genre_basis",
    "genre_status",
    "rights_status",
    "license_claim",
    "license_url_claim",
    "license_conflict",
    "source_metadata",
    "source_work_ids",
    "rules_version",
    "classical_scope",
    "beethoven_policy",
    "split_origin",
    "authorization_basis",
    "baseline_groups",
    "baseline_splits",
    "reason",
)


def _normal(value):
    text = "".join(
        char
        for char in unicodedata.normalize("NFKD", value.lower())
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _missing(value):
    return _normal(value) in {"", "na", "nan", "unknown", "composer", "anonymous"}


@lru_cache(maxsize=50000)
def _matches(value):
    normalized = _normal(value)
    matches = set()
    for person, (_, *aliases) in COMPOSERS.items():
        for alias in aliases:
            alias = _normal(alias)
            if normalized == alias or (
                " " in alias
                and re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", normalized)
            ):
                matches.add(person)
                break
    # A more specific full name can contain another person's full-name alias.
    if "mozart_franz_xaver" in matches:
        matches.discard("mozart_wolfgang_amadeus")
    return frozenset(matches)


def classify(row, *, classical_scope="baroque_classical", beethoven_policy="review"):
    """Return provisional (genre, canonical person, evidence, review reason)."""
    if classical_scope not in {"baroque_classical", "classical"}:
        raise ValueError("classical_scope must be baroque_classical or classical")
    if beethoven_policy not in {"review", "classical", "romantic"}:
        raise ValueError("beethoven_policy must be review, classical or romantic")
    tags = set(re.split(r"[-,;|\s]+", row.get("genres", "").lower())) - {"", "na"}
    rock = bool(tags & {"rock", "pop"})
    if rock and "classical" in tags:
        return "", "", "source_tags", "cross_genre_tags"
    composer, artist = row.get("composer_name", ""), row.get("artist_name", "")
    composers, artists = _matches(composer), _matches(artist)
    shorthand = False
    if not composers and len(artists) == 1:
        identified = next(iter(artists))
        surname = _normal(COMPOSERS[identified][1]).split()[-1]
        if _normal(composer) == surname:
            # An otherwise ambiguous surname is evidence only when the artist
            # field independently names one supported full person.
            composers, shorthand = artists, True
    people = composers | artists
    text = _normal(" ".join(row.get(key, "") for key in ("title", "composer_name", "artist_name")))
    if re.search(r"\b(medley|mashup|pastiche|inspired by|style of|opera rock)\b", text):
        return "", "", "attribution_text", "ambiguous_arrangement_or_inspiration"
    if len(people) > 1:
        return "", "", "composer_artist_aliases", "conflicting_composer_attributions"
    if rock:
        if people:
            return "", next(iter(people)), "composer_and_source_tags", "cross_genre_composer"
        return "rock_pop", "", "PDMX_rock_or_pop_tag", ""
    if not people:
        return (
            "",
            "",
            "source_tags",
            "unresolved_classical_period" if "classical" in tags else "unmatched_style",
        )
    person = next(iter(people))
    if not composers and not _missing(composer):
        return "", person, "artist_alias_only", "unrecognized_composer_with_known_artist"
    period = COMPOSERS[person][0]
    evidence = (
        "composer_surname_confirmed_by_artist:"
        if shorthand
        else "composer_alias:"
        if composers
        else "artist_alias:"
    ) + person
    if tags - {"classical"}:
        return "", person, evidence, "conflicting_source_style_tags"
    if period == "transitional":
        return (
            ("", person, evidence, "transitional_composer")
            if beethoven_policy == "review"
            else (beethoven_policy, person, evidence, "")
        )
    if period == "review" or (period == "baroque" and classical_scope == "classical"):
        return "", person, evidence, "composer_outside_period_policy"
    return ("romantic" if period == "romantic" else "classical"), person, evidence, ""


class _Families:
    def __init__(self):
        self.parent, self.index = [], {}

    def find(self, node):
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def add(self, keys):
        node = len(self.parent)
        self.parent.append(node)
        for key in keys:
            if key in self.index:
                a, b = self.find(node), self.find(self.index[key])
                self.parent[max(a, b)] = min(a, b)
            else:
                self.index[key] = node
        return node


def _work_keys(row, person):
    keys = set()
    # Broad PDMX best_path/best_arrangement clusters can connect unrelated works.
    # Use the unique-arrangement representative, plus independently recorded work
    # and hash evidence, rather than treating every source cluster as identity.
    for field in ("path", "best_unique_arrangement"):
        value = row.get(field, "").strip().removeprefix("./")
        if value and value.lower() != "na":
            keys.add("pdmx:" + value)
    if person:
        for field in ("song_name", "title"):
            title = _normal(row.get(field, ""))
            if len(title) >= 8 and title not in {
                "nocturne",
                "romance",
                "prelude",
                "piano sonata",
                "sonatina",
                "symphony",
                "untitled",
            }:
                keys.add(f"work:{person}:{title}")
    return keys


def _record(row, source_csv_sha, scope, beethoven, *, genre, person, evidence, reason=""):
    relative = row.get("mid", "").strip().removeprefix("./")
    return {
        "candidate_id": hashlib.sha256(relative.encode()).hexdigest()[:20],
        "sha256": "",
        "midi_path": relative,
        "genre": genre,
        "genre_id": str(GENRES.index(genre)) if genre else "",
        "composition_group": "",
        "split": "",
        "training_status": "REVIEW_REQUIRED" if reason else "APPROVED",
        "approved_for_training": "false" if reason else "true",
        "source": "PDMX",
        "source_csv_sha256": source_csv_sha,
        "title": row.get("title", ""),
        "song_name": row.get("song_name", ""),
        "composer": row.get("composer_name", ""),
        "artist": row.get("artist_name", ""),
        "canonical_composer": person,
        "source_genres": row.get("genres", ""),
        "genre_basis": evidence,
        "genre_status": "PROVISIONAL",
        "rights_status": "UNVERIFIED",
        "license_claim": row.get("license", ""),
        "license_url_claim": row.get("license_url", ""),
        "license_conflict": row.get("license_conflict", ""),
        "source_metadata": row.get("metadata", ""),
        "source_work_ids": ";".join(sorted(_work_keys(row, ""))),
        "rules_version": RULES_VERSION,
        "classical_scope": scope,
        "beethoven_policy": beethoven,
        "split_origin": "",
        "authorization_basis": "user_authorized_expanded_training_selection",
        "reason": reason,
    }


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(
    pdmx_csv,
    midi_root,
    output_dir,
    *,
    baseline_manifest=None,
    classical_scope="baroque_classical",
    beethoven_policy="review",
    seed=42,
    split_ratios=(0.7, 0.2, 0.1),
):
    """Stream PDMX metadata and build uncapped, hash-verified source manifests.

    Baseline files and entire related families retain their recorded split.
    Conflicting baseline splits or genre attributions quarantine the family.
    New families receive a deterministic split; no model training is started.
    """
    classify({}, classical_scope=classical_scope, beethoven_policy=beethoven_policy)
    if (
        len(split_ratios) != 3
        or any(not math.isfinite(n) or n <= 0 for n in split_ratios)
        or not math.isclose(sum(split_ratios), 1)
    ):
        raise ValueError("split_ratios must be three positive fractions summing to one")
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    pdmx_csv, midi_root = Path(pdmx_csv), Path(midi_root)
    baseline = (
        []
        if baseline_manifest is None
        else load_approved(baseline_manifest, min_per_genre=1, max_per_genre=1_000_000_000)
    )
    baseline_paths = {row["midi_path"]: row for row in baseline}
    baseline_hashes = {row["sha256"]: row for row in baseline}
    families, anchors = _Families(), []
    for row in baseline:
        keys = {
            "sha:" + row["sha256"],
            "source:" + row["midi_path"],
            "baseline:" + row["composition_group"],
        }
        attribution = {
            "composer_name": row.get("composer", ""),
            "artist_name": row.get("artist", ""),
        }
        _, person, _, _ = classify(
            attribution, classical_scope=classical_scope, beethoven_policy=beethoven_policy
        )
        keys |= _work_keys({"title": row.get("title", "")}, person)
        anchors.append((families.add(keys), row))
    source_csv_sha, counts = digest(pdmx_csv), Counter()
    candidates, reviews = [], []
    required = {
        "mid",
        "genres",
        "composer_name",
        "artist_name",
        "title",
        "is_best_unique_arrangement",
        "subset:all_valid",
        "license_conflict",
        "is_draft",
    }
    with pdmx_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required <= set(reader.fieldnames or []):
            raise ValueError("PDMX metadata is missing required columns")
        for raw in reader:
            counts["source_rows"] += 1
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"Malformed PDMX row {counts['source_rows'] + 1}")
            row = {key: value.strip() for key, value in raw.items()}
            relative = row["mid"].removeprefix("./")
            genre, person, evidence, reason = classify(
                row, classical_scope=classical_scope, beethoven_policy=beethoven_policy
            )
            if relative in baseline_paths:
                anchor = baseline_paths[relative]
                keys = _work_keys(row, person) | {"source:" + relative, "sha:" + anchor["sha256"]}
                families.add(keys)
                counts["baseline_rows_found_in_pdmx"] += 1
            if not (
                row["subset:all_valid"].lower() == "true"
                and row["is_best_unique_arrangement"].lower() == "true"
                and row["is_draft"].lower() != "true"
                and row["license_conflict"].lower() == "false"
            ):
                counts["metadata_gate_excluded"] += 1
                continue
            counts["metadata_eligible"] += 1
            if reason == "unmatched_style":
                counts[reason] += 1
                continue
            record = _record(
                row,
                source_csv_sha,
                classical_scope,
                beethoven_policy,
                genre=genre,
                person=person,
                evidence=evidence,
                reason=reason,
            )
            if reason:
                reviews.append(record)
                counts["genre_review"] += 1
                continue
            try:
                path = source_path(midi_root, relative)
                with path.open("rb") as source:
                    header = source.read(14)
                if len(header) != 14 or header[:4] != b"MThd":
                    raise ValueError("Missing MIDI header")
                size, fmt, tracks, ppq = struct.unpack(">IHHH", header[4:14])
                if size < 6 or fmt not in (0, 1) or not tracks or not ppq or ppq & 0x8000:
                    raise ValueError("Unsupported MIDI header/timing")
                checksum = digest(path)
                if relative in baseline_paths and checksum != baseline_paths[relative]["sha256"]:
                    raise ValueError("Baseline source hash changed")
            except (OSError, ValueError, struct.error) as exc:
                record.update(
                    training_status="REVIEW_REQUIRED",
                    approved_for_training="false",
                    reason="source_error:" + str(exc),
                )
                reviews.append(record)
                counts["source_errors"] += 1
                continue
            record["sha256"] = checksum
            keys = _work_keys(row, person) | {"sha:" + checksum, "source:" + relative}
            if checksum in baseline_hashes:
                keys.add("baseline:" + baseline_hashes[checksum]["composition_group"])
            candidates.append((families.add(keys), record))
    by_family, anchor_families = defaultdict(list), defaultdict(list)
    for node, record in candidates:
        by_family[families.find(node)].append(record)
    for node, row in anchors:
        anchor_families[families.find(node)].append(row)
    selected, family_report = [], []
    for family, records in sorted(by_family.items()):
        original = anchor_families[family]
        splits = {row["split"] for row in original}
        labels = {row["genre"] for row in original} | {row["genre"] for row in records}
        for record in records:
            record["baseline_groups"] = ";".join(
                sorted({row["composition_group"] for row in original})
            )
            record["baseline_splits"] = ";".join(sorted(splits))
        fingerprint = "|".join(
            sorted(
                {row["sha256"] for row in records}
                | {"baseline:" + row["composition_group"] for row in original}
            )
        )
        group = "pdmx_" + hashlib.sha256(fingerprint.encode()).hexdigest()[:24]
        conflict = (
            "baseline_split_conflict"
            if len(splits) > 1
            else "family_genre_conflict"
            if len(labels) > 1
            else ""
        )
        if conflict:
            for record in records:
                record.update(
                    composition_group=group,
                    training_status="REVIEW_REQUIRED",
                    approved_for_training="false",
                    reason=conflict,
                )
                reviews.append(record)
            counts[conflict + "_files"] += len(records)
            family_report.append(
                {
                    "group": group,
                    "reason": conflict,
                    "baseline_splits": sorted(splits),
                    "genres": sorted(labels),
                    "files": len(records),
                }
            )
            continue
        if splits:
            split, origin = next(iter(splits)), "baseline_family"
        else:
            draw = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), 16) / 2**256
            split = (
                "train"
                if draw < split_ratios[0]
                else "val"
                if draw < sum(split_ratios[:2])
                else "test"
            )
            origin = "deterministic_family_hash"
        unique = {}
        for record in sorted(
            records, key=lambda r: (r["midi_path"] not in baseline_paths, r["midi_path"])
        ):
            if record["sha256"] in unique:
                record.update(
                    composition_group=group,
                    split=split,
                    training_status="REVIEW_REQUIRED",
                    approved_for_training="false",
                    reason="duplicate_source_hash",
                )
                reviews.append(record)
                counts["duplicate_source_hash"] += 1
                continue
            unique[record["sha256"]] = record
            record.update(composition_group=group, split=split, split_origin=origin)
            selected.append(record)
    selected.sort(key=lambda r: (r["genre"], r["split"], r["midi_path"]))
    reviews.sort(key=lambda r: (r["reason"], r["midi_path"]))
    counts["selected_files"], counts["review_files"] = len(selected), len(reviews)
    report = {
        "rules_version": RULES_VERSION,
        "builder_sha256": digest(__file__),
        "source_csv_sha256": source_csv_sha,
        "baseline_manifest_sha256": digest(baseline_manifest) if baseline_manifest else None,
        "classical_scope": classical_scope,
        "beethoven_policy": beethoven_policy,
        "seed": seed,
        "split_ratios": list(split_ratios),
        "counts": dict(counts),
        "selected_by_genre": {g: sum(r["genre"] == g for r in selected) for g in GENRES},
        "selected_by_genre_split": {
            g: {
                s: sum(r["genre"] == g and r["split"] == s for r in selected)
                for s in ("train", "val", "test")
            }
            for g in GENRES
        },
        "baseline_family_files": sum(r["split_origin"] == "baseline_family" for r in selected),
        "quarantined_families": family_report,
        "review_reasons": dict(Counter(r["reason"] for r in reviews)),
        "composer_rules": {key: list(value) for key, value in COMPOSERS.items()},
        "genre_policy": (
            "Baroque/Classical or Classical-era composer aliases; Romantic aliases; explicit "
            "rock/pop tags. Generic classical tags, conflicting attributions, transitional "
            "and out-of-policy composers are reviewed."
        ),
        "period_boundary_policy": (
            "The Romantic label includes the explicitly listed Romantic/late-Romantic "
            "composers. Debussy and Ravel are kept for review rather than assigning "
            "Impressionist works to Romantic by default. Beethoven uses its explicit policy."
        ),
        "rights_policy": (
            "Source claims preserved; license_conflict=False is a metadata gate, "
            "not independent rights clearance."
        ),
        "split_policy": (
            "Preserve baseline splits by hashes, recorded groups and linked PDMX/work families; "
            "quarantine conflicting families. New family splits use deterministic hashes."
        ),
        "limitations": [
            "Metadata labels remain provisional.",
            "Work keys and hashes do not prove absence of unrecorded musical near-duplicates.",
            "MIDI headers and hashes are checked; full parsing and window quality checks "
            "happen during preparation.",
        ],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pdmx-manifest-", dir=output_dir.parent))
    try:
        _write_csv(staging / "expanded.csv", selected)
        _write_csv(staging / "review.csv", reviews)
        report["expanded_manifest_sha256"] = digest(staging / "expanded.csv")
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report
