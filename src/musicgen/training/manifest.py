"""Validate selected MIDI manifests without changing approval or genre decisions."""

import csv
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from musicgen.generation.runtime.tokenizer import GENRES

SPLITS = ("train", "val", "test")
REQUIRED = (
    "candidate_id",
    "sha256",
    "midi_path",
    "genre",
    "genre_id",
    "composition_group",
    "split",
    "training_status",
    "approved_for_training",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def digest(path: str | Path) -> str:
    """Return a streaming SHA-256 digest of a file."""
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _relative_midi_path(relative: str) -> PurePosixPath:
    if not relative or "\\" in relative or ":" in relative:
        raise ValueError("MIDI paths must be relative POSIX paths")
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or any(ord(c) < 32 for c in relative):
        raise ValueError("MIDI path escapes the dataset root")
    if path.suffix.lower() not in {".mid", ".midi"}:
        raise ValueError("Expected a .mid or .midi source file")
    return path


def source_path(midi_root: str | Path, relative: str) -> Path:
    """Resolve an existing MIDI file, rejecting traversal and escaping symlinks."""
    path = _relative_midi_path(relative)
    root = Path(midi_root).resolve()
    if not root.is_dir():
        raise ValueError(f"MIDI root is not a directory: {root}")
    candidate = (root / path).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ValueError(f"Missing MIDI file or path outside dataset root: {relative}")
    return candidate


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields):
            raise ValueError("Manifest has duplicate column names")
        missing = set(REQUIRED) - set(fields)
        if missing:
            raise ValueError("Manifest is missing columns: " + ", ".join(sorted(missing)))
        rows = []
        for line, raw in enumerate(reader, start=2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"Malformed CSV row at line {line}")
            row = {key: value.strip() for key, value in raw.items()}
            empty = [key for key in REQUIRED if not row[key]]
            if empty:
                raise ValueError(f"Empty required fields at line {line}: {', '.join(empty)}")
            rows.append(row)
    if not rows:
        raise ValueError("Manifest contains no selected files")
    return rows


def load_approved(
    path: str | Path,
    *,
    midi_root: str | Path | None = None,
    min_per_genre: int = 200,
    max_per_genre: int = 250,
    verify_hashes: bool = True,
) -> list[dict[str, str]]:
    """Validate selected rows; optionally verify source files against their hashes.

    Approval here means the manifest records an existing training decision. Rights
    and genre-review fields remain unchanged and are not promoted to verified.
    Without ``midi_root``, only manifest structure and lexical paths are checked.
    A composition group may contain related files, but cannot cross splits or labels.
    """
    if not 1 <= min_per_genre <= max_per_genre:
        raise ValueError("Expected 1 <= min_per_genre <= max_per_genre")
    rows = _read_rows(path)
    seen: dict[str, set[str]] = {key: set() for key in ("candidate_id", "sha256", "midi_path")}
    groups: dict[str, tuple[str, str]] = {}
    counts: Counter = Counter()
    split_counts: Counter = Counter()
    for row in rows:
        identifier = row["candidate_id"]
        if row["training_status"] != "APPROVED" or row["approved_for_training"].lower() != "true":
            raise ValueError(f"Row is not approved for training: {identifier}")
        genre, split = row["genre"], row["split"]
        if genre not in GENRES or split not in SPLITS:
            raise ValueError(f"Invalid genre or split: {identifier}")
        if row["genre_id"] != str(GENRES.index(genre)):
            raise ValueError(f"genre_id does not match genre: {identifier}")
        row["sha256"] = row["sha256"].lower()
        if not SHA256.fullmatch(row["sha256"]):
            raise ValueError(f"Invalid source SHA-256: {identifier}")
        row["midi_path"] = str(_relative_midi_path(row["midi_path"]))
        for key, values in seen.items():
            if row[key] in values:
                raise ValueError(f"Duplicate {key}: {identifier}")
            values.add(row[key])
        group = row["composition_group"]
        if group in groups and groups[group] != (genre, split):
            raise ValueError(f"Composition group crosses splits or genre labels: {group}")
        groups[group] = genre, split
        counts[genre] += 1
        split_counts[genre, split] += 1
        if midi_root is not None:
            source = source_path(midi_root, row["midi_path"])
            if verify_hashes and digest(source) != row["sha256"]:
                raise ValueError(f"Source hash mismatch: {identifier}")
            with source.open("rb") as handle:
                if handle.read(4) != b"MThd":
                    raise ValueError(f"Source file has no MIDI header: {identifier}")
    for genre in GENRES:
        if not min_per_genre <= counts[genre] <= max_per_genre:
            raise ValueError(
                f"Expected {min_per_genre}–{max_per_genre} selected files for {genre}, "
                f"got {counts[genre]}"
            )
        for split in SPLITS:
            if not split_counts[genre, split]:
                raise ValueError(f"No selected files for {genre}/{split}")
    return rows


def audit_manifest(
    path: str | Path,
    midi_root: str | Path,
    *,
    min_per_genre: int = 200,
    max_per_genre: int = 250,
    verify_hashes: bool = True,
) -> dict:
    """Return a JSON-compatible audit, preserving unresolved review statuses."""
    rows = load_approved(
        path,
        midi_root=midi_root,
        min_per_genre=min_per_genre,
        max_per_genre=max_per_genre,
        verify_hashes=verify_hashes,
    )
    warnings = []
    genre_status = Counter(row.get("genre_status", "UNSPECIFIED") for row in rows)
    rights_status = Counter(row.get("rights_status", "UNSPECIFIED") for row in rows)
    if any(status.upper() != "VERIFIED" for status in genre_status):
        warnings.append("Genre metadata is not fully verified; labels were preserved unchanged.")
    if any(status.upper() != "VERIFIED" for status in rights_status):
        warnings.append("Training approval does not independently verify source rights.")
    warnings.append(
        "Group/hash isolation is checked against recorded identifiers; "
        "unrecorded related works and musical near-duplicates require separate review."
    )
    return {
        "manifest_sha256": digest(path),
        "rows": len(rows),
        "by_genre": {genre: sum(r["genre"] == genre for r in rows) for genre in GENRES},
        "by_genre_split": {
            genre: {
                split: sum(r["genre"] == genre and r["split"] == split for r in rows)
                for split in SPLITS
            }
            for genre in GENRES
        },
        "unique_composition_groups": len({row["composition_group"] for row in rows}),
        "verified_files": len(rows) if verify_hashes else 0,
        "verified_bytes": sum(source_path(midi_root, r["midi_path"]).stat().st_size for r in rows),
        "hashes_verified": verify_hashes,
        "approval_status_counts": dict(Counter(row["training_status"] for row in rows)),
        "genre_status_counts": dict(genre_status),
        "rights_status_counts": dict(rights_status),
        "copyright_clearance_counts": dict(
            Counter(row.get("copyright_clearance_confirmed", "UNSPECIFIED") for row in rows)
        ),
        "warnings": warnings,
    }


def build_review_candidates(
    pdmx_csv: str | Path,
    midi_root: str | Path,
    output_path: str | Path,
    *,
    per_bucket: int = 250,
    seed: int = 17,
) -> dict:
    """Write a deterministic *unapproved* PDMX shortlist for manual review.

    Source tags support ``rock_pop`` or an ambiguous ``classical_or_romantic``
    bucket, not a defensible historical-period assignment. No training split or
    approved genre is assigned. The output cannot pass ``load_approved``.
    """
    if per_bucket < 1:
        raise ValueError("per_bucket must be positive")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Review output already exists: {output}")
    pools: dict[str, list] = defaultdict(list)
    scanned = 0
    with Path(pdmx_csv).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        needed = {"mid", "genres", "license_conflict", "is_best_unique_arrangement"}
        if not needed <= set(reader.fieldnames or []):
            raise ValueError("PDMX CSV is missing candidate-selection metadata")
        for raw in reader:
            scanned += 1
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"Malformed PDMX row: {scanned + 1}")
            if raw["license_conflict"].lower() != "false":
                continue
            if raw["is_best_unique_arrangement"].lower() != "true":
                continue
            if raw.get("is_draft", "false").lower() == "true":
                continue
            if raw.get("subset:all_valid", "true").lower() != "true":
                continue
            tags = set(re.split(r"[-,;|\s]+", raw["genres"].lower()))
            if tags & {"rock", "pop"}:
                bucket = "rock_pop"
            elif "classical" in tags:
                bucket = "classical_or_romantic"
            else:
                continue
            try:
                relative = str(_relative_midi_path(raw["mid"].strip()))
            except ValueError:
                continue
            rank = int(hashlib.sha256(f"{seed}:{relative}".encode()).hexdigest(), 16)
            # A bounded pool prevents the 215 MB source table being loaded into memory.
            entry = (-rank, relative, json.dumps(raw, sort_keys=True))
            pool = pools[bucket]
            if len(pool) < per_bucket * 8:
                heapq.heappush(pool, entry)
            elif entry > pool[0]:
                heapq.heapreplace(pool, entry)
    candidates = []
    seen_hashes: set[str] = set()
    seen_paths: set[str] = set()
    by_bucket = Counter()
    for bucket in sorted(pools):
        for _, relative, payload in sorted(pools[bucket], reverse=True):
            if by_bucket[bucket] >= per_bucket:
                break
            if relative in seen_paths:
                continue
            try:
                source = source_path(midi_root, relative)
                sha = digest(source)
                with source.open("rb") as handle:
                    valid_header = handle.read(4) == b"MThd"
            except (OSError, ValueError):
                continue
            if sha in seen_hashes or not valid_header:
                continue
            raw = json.loads(payload)
            row = dict.fromkeys(REQUIRED, "")
            row.update(
                candidate_id=hashlib.sha256(relative.encode()).hexdigest()[:16],
                sha256=sha,
                midi_path=relative,
                training_status="REVIEW_REQUIRED",
                approved_for_training="false",
                composition_group="pdmx_"
                + hashlib.sha256(raw.get("best_unique_arrangement", relative).encode()).hexdigest()[
                    :20
                ],
                suggested_genre=bucket,
                source="PDMX",
                source_genres=raw["genres"],
                title=raw.get("title", ""),
                composer=raw.get("composer_name", ""),
                artist=raw.get("artist_name", ""),
                license_claim=raw.get("license", ""),
                rights_status="UNVERIFIED",
                genre_status="PROVISIONAL",
                copyright_clearance_confirmed="false",
                review_note=(
                    "Review genre, related works, rights, MIDI quality and splits before use."
                ),
            )
            candidates.append(row)
            seen_hashes.add(sha)
            seen_paths.add(relative)
            by_bucket[bucket] += 1
    if not candidates:
        raise ValueError("No existing local MIDI candidates passed source-metadata screening")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(candidates)
    return {
        "source_rows_scanned": scanned,
        "review_candidates": len(candidates),
        "by_suggested_genre": dict(by_bucket),
        "approved_for_training": 0,
        "output_sha256": digest(output),
        "warnings": [
            "No genres, training approvals or splits were assigned.",
            "Source license flags are metadata claims, not independent rights clearance.",
            "Bounded candidate pools can return fewer than the requested count.",
        ],
    }
