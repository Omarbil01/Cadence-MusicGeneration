"""Select a reproducible, technically usable 250-file-per-genre experiment.

Selection samples within the curated manifest's existing splits. One source per
composition family prevents alternate arrangements filling the smaller dataset;
composer/artist strata add diversity without using model or held-out scores.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict, deque
from pathlib import Path

from musicgen.generation.runtime.tokenizer import GENRES
from musicgen.training import preprocessing
from musicgen.training.manifest import digest, load_approved, source_path

SELECTION_POLICY = "cadence-balanced250-v1"
SPLITS = ("train", "val", "test")
EXTRA_FIELDS = ("selection_policy", "selected_from_manifest_sha256", "selection_stratum")


def _usable_name(value):
    return value.strip().lower() not in {"", "na", "nan", "unknown", "anonymous"}


def _stratum(row):
    if _usable_name(row.get("canonical_composer", "")):
        return row["canonical_composer"].strip().casefold()
    # Performer identity is more informative than a transcriber's composer field
    # for the source-tagged Rock/Pop arrangements.
    fields = ("artist", "composer") if row["genre"] == "rock_pop" else ("composer", "artist")
    for field in fields:
        if _usable_name(row.get(field, "")):
            return row[field].strip().casefold()
    return "unidentified"


def _order(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _known_exclusions(path, source_manifest_sha):
    if path is None:
        return {}, None
    path = Path(path)
    if path.is_dir():
        path /= "metadata.json"
    document = json.loads(path.read_text())
    identity = document.get("identity", {})
    if identity.get("manifest_sha256", source_manifest_sha) != source_manifest_sha:
        raise ValueError("Preparation report belongs to a different curated manifest")
    report = document.get("report", document)
    excluded = {}
    for row in report.get("rejected_files", []):
        if row.get("candidate_id"):
            excluded[row["candidate_id"]] = row.get("error", "Known technical exclusion")
    return excluded, digest(path)


def _quotas(per_genre, split_counts):
    if isinstance(per_genre, bool) or not isinstance(per_genre, int) or per_genre < 3:
        raise ValueError("per_genre must be an integer of at least three")
    if split_counts is None:
        train, val = round(per_genre * 0.7), round(per_genre * 0.2)
        split_counts = {"train": train, "val": val, "test": per_genre - train - val}
    if (
        set(split_counts) != set(SPLITS)
        or any(type(value) is not int or value < 1 for value in split_counts.values())
        or sum(split_counts.values()) != per_genre
    ):
        raise ValueError("Positive train/val/test split_counts must sum to per_genre")
    return dict(split_counts)


def build_balanced_manifest(
    curated_manifest,
    midi_root,
    output_dir,
    *,
    preparation_report=None,
    per_genre=250,
    split_counts=None,
    seed=42,
):
    """Verify, diversify and publish an exact balanced selection into a new directory.

    Each candidate is encoded lazily with normalized tempo and 16 progress bins.
    Retained notes and a genuine piece-ending sequence are required. Rejections
    are replenished within their existing genre/split; no source is reassigned.
    The function never starts training, changes source files or evaluates a model.
    """
    quotas = _quotas(per_genre, split_counts)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Selection output already exists: {output_dir}")
    rows = load_approved(
        curated_manifest,
        midi_root=midi_root,
        min_per_genre=1,
        max_per_genre=1_000_000_000,
        verify_hashes=True,
    )
    source_manifest_sha = digest(curated_manifest)
    excluded, known_report_sha = _known_exclusions(preparation_report, source_manifest_sha)
    buckets = defaultdict(list)
    rejected = []
    for row in rows:
        if row["candidate_id"] in excluded:
            rejected.append(
                {
                    "candidate_id": row["candidate_id"],
                    "genre": row["genre"],
                    "split": row["split"],
                    "reason": excluded[row["candidate_id"]],
                    "stage": "known_preparation_rejection",
                }
            )
            continue
        buckets[row["genre"], row["split"]].append(row)
    selected, selected_reports = [], []
    used_families, used_hashes = set(), set()
    skipped_family, encoded_candidates = 0, 0
    for genre in GENRES:
        for split in SPLITS:
            strata = defaultdict(list)
            for row in buckets[genre, split]:
                strata[_stratum(row)].append(row)
            queues = {
                name: deque(
                    sorted(
                        candidates,
                        key=lambda row: _order(seed, row["sha256"] + ":" + row["candidate_id"]),
                    )
                )
                for name, candidates in strata.items()
            }
            names = sorted(queues, key=lambda name: _order(seed, f"{genre}:{split}:{name}"))
            count, target = 0, quotas[split]
            while count < target:
                progressed = False
                for name in names:
                    if count == target:
                        break
                    queue = queues[name]
                    while queue:
                        progressed = True
                        row = queue.popleft()
                        if row["composition_group"] in used_families:
                            skipped_family += 1
                            continue
                        if row["sha256"] in used_hashes:
                            raise ValueError("Curated selection contains a duplicate source hash")
                        encoded_candidates += 1
                        try:
                            encoded = preprocessing.encode_midi(
                                source_path(midi_root, row["midi_path"]),
                                normalize_tempo=True,
                                progress_bins=16,
                            )
                            if not encoded["sequences"] or not encoded["report"].get(
                                "retained_notes", 0
                            ):
                                raise ValueError("No usable encoded notes or sequences")
                            endings = sum(
                                sequence.get("is_piece_end", False)
                                for sequence in encoded["sequences"]
                            )
                            if not endings:
                                raise ValueError("No retained true piece-ending sequence")
                        except (OSError, ValueError) as exc:
                            rejected.append(
                                {
                                    "candidate_id": row["candidate_id"],
                                    "genre": genre,
                                    "split": split,
                                    "reason": str(exc),
                                    "stage": "fresh_encoding",
                                }
                            )
                            continue
                        selected.append(
                            {
                                **row,
                                "selection_policy": SELECTION_POLICY,
                                "selected_from_manifest_sha256": source_manifest_sha,
                                "selection_stratum": name,
                            }
                        )
                        selected_reports.append(
                            {
                                "candidate_id": row["candidate_id"],
                                "genre": genre,
                                "split": split,
                                "selection_stratum": name,
                                "true_piece_endings": endings,
                                **encoded["report"],
                            }
                        )
                        used_families.add(row["composition_group"])
                        used_hashes.add(row["sha256"])
                        count += 1
                        break
                if not progressed:
                    raise ValueError(
                        f"Insufficient usable composition families for {genre}/{split}: "
                        f"selected {count}, required {target}; {len(rejected)} exclusions recorded"
                    )
    selected.sort(
        key=lambda row: (
            GENRES.index(row["genre"]),
            SPLITS.index(row["split"]),
            row["candidate_id"],
        )
    )
    fields = list(rows[0]) + [field for field in EXTRA_FIELDS if field not in rows[0]]
    report = {
        "selection_policy": SELECTION_POLICY,
        "selector_sha256": digest(__file__),
        "selected_from_manifest_sha256": source_manifest_sha,
        "preparation_report_sha256": known_report_sha,
        "preprocessing_sha256": digest(preprocessing.__file__),
        "seed": seed,
        "files_per_genre": per_genre,
        "split_counts_per_genre": quotas,
        "source_pool_files": len(rows),
        "selected_files": len(selected),
        "encoded_candidates": encoded_candidates,
        "unique_composition_families": len(used_families),
        "duplicate_family_candidates_skipped": skipped_family,
        "selected_by_genre_split": {
            genre: {
                split: sum(row["genre"] == genre and row["split"] == split for row in selected)
                for split in SPLITS
            }
            for genre in GENRES
        },
        "selected_strata_by_genre": {
            genre: dict(
                Counter(row["selection_stratum"] for row in selected if row["genre"] == genre)
            )
            for genre in GENRES
        },
        "rejected_candidates": rejected,
        "file_reports": selected_reports,
        "split_policy": "Keep existing splits and groups; one file per composition family.",
        "diversity_policy": (
            "Deterministic round-robin canonical composers, then artist/composer metadata."
        ),
        "encoding_policy": {
            "normalize_tempo": True,
            "progress_bins": 16,
            "require_true_piece_end": True,
        },
        "interpretation": "Source usability and metadata diversity do not measure musical quality.",
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".balanced-manifest-", dir=output_dir.parent))
    try:
        with (staging / "selected.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(selected)
        report["selected_manifest_sha256"] = digest(staging / "selected.csv")
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"Selection output already exists: {output_dir}")
        staging.rename(output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report
