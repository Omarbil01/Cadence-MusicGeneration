"""Immutable SQLite window caches with lazy token access and verified provenance."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import zlib
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path

from musicgen.generation.runtime.tokenizer import GENRES
from musicgen.training.manifest import audit_manifest, digest, source_path

SPLITS = ("train", "val", "test")
CACHE_FORMAT = "cadence-sqlite-windows-v1"


def _identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _payload(row) -> bytes:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class SQLiteWindows(Sequence):
    """Read one compressed window at a time; sampling reads only indexed metadata.

    Connections are opened lazily, read-only, and excluded from pickle/deepcopy.
    Call ``close`` when a notebook no longer needs a prepared dataset.
    """

    def __init__(self, database, split, count):
        if split not in SPLITS or not isinstance(count, int) or count < 0:
            raise ValueError("Invalid SQLite split or window count")
        self.database = Path(database).resolve()
        self.split = split
        self.count = count
        self._connection = None
        self._pid = None

    def _connect(self):
        if self._connection is None or self._pid != os.getpid():
            self.close()
            self._connection = sqlite3.connect(
                self.database.as_uri() + "?mode=ro&immutable=1",
                uri=True,
                check_same_thread=False,
            )
            self._connection.execute("PRAGMA cache_size=-8192")
            self._connection.execute("PRAGMA temp_store=FILE")
            self._pid = os.getpid()
        return self._connection

    def close(self):
        if self._connection is not None:
            self._connection.close()
        self._connection = None
        self._pid = None

    def __getstate__(self):
        return {**self.__dict__, "_connection": None, "_pid": None}

    def __len__(self):
        return self.count

    def _decode(self, record):
        if record is None:
            raise ValueError("SQLite cache is missing a requested window")
        blob, expected, candidate, group, genre = record
        try:
            encoded = zlib.decompress(blob)
            if hashlib.sha256(encoded).hexdigest() != expected:
                raise ValueError("SQLite window checksum mismatch")
            row = json.loads(encoded)
        except (zlib.error, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("SQLite cache has a corrupt window payload") from exc
        if (row.get("candidate_id"), row.get("composition_group"), row.get("genre_id")) != (
            candidate,
            group,
            genre,
        ):
            raise ValueError("SQLite window metadata differs from its token payload")
        return row

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(self.count))]
        if not isinstance(index, int):
            raise TypeError("Window indices must be integers or slices")
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError("Window index is outside this split")
        record = (
            self._connect()
            .execute(
                "SELECT payload,payload_sha256,candidate_id,composition_group,genre_id "
                "FROM windows WHERE split=? AND idx=?",
                (self.split, index),
            )
            .fetchone()
        )
        return self._decode(record)

    def __iter__(self) -> Iterator[dict]:
        cursor = self._connect().execute(
            "SELECT payload,payload_sha256,candidate_id,composition_group,genre_id "
            "FROM windows WHERE split=? ORDER BY idx",
            (self.split,),
        )
        for record in cursor:
            yield self._decode(record)

    def file_indices(self):
        """Yield file genre and contiguous index ranges without loading token blobs."""
        cursor = self._connect().execute(
            "SELECT genre_id,MIN(idx),MAX(idx),COUNT(*) FROM windows "
            "WHERE split=? GROUP BY candidate_id,genre_id ORDER BY MIN(idx)",
            (self.split,),
        )
        for genre, first, last, count in cursor:
            if last - first + 1 != count:
                raise ValueError("SQLite cache file windows are not contiguous")
            yield genre, range(first, last + 1)

    def take_per_genre(self, limit):
        """Fetch a bounded smoke subset using only indexed genre lookups."""
        result = []
        for genre in range(len(GENRES)):
            cursor = self._connect().execute(
                "SELECT idx FROM windows WHERE split=? AND genre_id=? ORDER BY idx LIMIT ?",
                (self.split, genre, limit),
            )
            result.extend(self[index] for (index,) in cursor)
        return result


def _load(directory, identity):
    descriptor = json.loads((directory / "metadata.json").read_text())
    checksum = descriptor.pop("content_sha256", None)
    if checksum != _identity(descriptor):
        raise ValueError("SQLite cache metadata checksum changed")
    descriptor["content_sha256"] = checksum
    if descriptor.get("identity") != identity or descriptor.get("format") != CACHE_FORMAT:
        raise ValueError("SQLite cache identity does not match this preparation")
    database = directory / "windows.sqlite3"
    if digest(database) != descriptor["database_sha256"]:
        raise ValueError("SQLite cache database checksum changed; prepare a new cache")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        if connection.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ValueError("SQLite cache failed its integrity check")
        counts = dict(connection.execute("SELECT split,COUNT(*) FROM windows GROUP BY split"))
        if counts != descriptor["split_counts"]:
            raise ValueError("SQLite cache split counts changed")
    return {
        key: descriptor[key]
        for key in ("identity", "dataset_sha256", "metadata", "report", "content_sha256")
    } | {
        "splits": {
            split: SQLiteWindows(database, split, descriptor["split_counts"][split])
            for split in SPLITS
        },
        "storage": {
            "kind": "sqlite",
            "path": str(database),
            "database_sha256": descriptor["database_sha256"],
        },
    }


def prepare_sqlite_dataset(
    manifest_path,
    dataset_root,
    cache_dir,
    rows,
    identity,
    *,
    context,
    min_per_genre,
    max_per_genre,
    normalize_tempo=False,
    progress_bins=0,
):
    """Build a transactionally published cache, bounded by one encoded MIDI file.

    Window payloads are never collected for the complete dataset in memory.
    Existing caches are rehashed before lazy sequences are returned.
    """
    from musicgen.training import preprocessing

    identity = {
        **identity,
        "storage": "sqlite",
        "cache_format": CACHE_FORMAT,
        "cache_code_sha256": digest(__file__),
    }
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    directory = cache_dir / (_identity(identity) + ".sqlite")
    if directory.exists():
        return _load(directory, identity)
    staging = Path(tempfile.mkdtemp(prefix=".sqlite-build-", dir=cache_dir))
    connection = None
    temporary_report = None
    try:
        connection = sqlite3.connect(staging / "windows.sqlite3")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-8192")
        connection.execute(
            "CREATE TABLE windows(split TEXT NOT NULL,idx INTEGER NOT NULL,"
            "candidate_id TEXT NOT NULL,composition_group TEXT NOT NULL,genre_id INTEGER NOT NULL,"
            "payload BLOB NOT NULL,payload_sha256 TEXT NOT NULL,"
            "PRIMARY KEY(split,idx)) WITHOUT ROWID"
        )
        connection.execute("CREATE INDEX genre_windows ON windows(split,genre_id,idx)")
        connection.execute("CREATE INDEX file_windows ON windows(split,candidate_id,genre_id,idx)")
        connection.execute("BEGIN")
        retained, retained_splits = Counter(), Counter()
        split_counts = dict.fromkeys(SPLITS, 0)
        window_hashes = {split: hashlib.sha256() for split in SPLITS}
        counters = {
            name: {genre: Counter() for genre in GENRES}
            for name in ("program_counts", "drum_pitch_counts", "meter_counts")
        }
        reports, rejected = [], []
        for index, row in enumerate(rows, 1):
            split = row["split"]
            first_index = split_counts[split]
            previous_hash = window_hashes[split].copy()
            connection.execute("SAVEPOINT source_file")
            try:
                options = (
                    {"normalize_tempo": normalize_tempo, "progress_bins": progress_bins}
                    if normalize_tempo or progress_bins
                    else {}
                )
                encoded = preprocessing.encode_midi(
                    source_path(dataset_root, row["midi_path"]), **options
                )
                for sequence in encoded["sequences"]:
                    framing = (
                        {
                            "progress_ids": sequence.get("progress_ids"),
                            "supervise_eos": sequence.get("is_piece_end", True),
                        }
                        if normalize_tempo or progress_bins
                        else {}
                    )
                    for window in preprocessing.make_windows(
                        sequence["ids"],
                        context,
                        **framing,
                    ):
                        record = {
                            **window,
                            "candidate_id": row["candidate_id"],
                            "composition_group": row["composition_group"],
                            "genre_id": GENRES.index(row["genre"]),
                        }
                        payload = _payload(record)
                        checksum = hashlib.sha256(payload).hexdigest()
                        connection.execute(
                            "INSERT INTO windows VALUES(?,?,?,?,?,?,?)",
                            (
                                split,
                                split_counts[split],
                                row["candidate_id"],
                                row["composition_group"],
                                record["genre_id"],
                                zlib.compress(payload, level=1),
                                checksum,
                            ),
                        )
                        split_counts[split] += 1
                        window_hashes[split].update((checksum + "\n").encode())
                count = split_counts[split] - first_index
                if not count:
                    raise ValueError("No usable encoded sequences")
            except (OSError, ValueError) as exc:
                connection.execute("ROLLBACK TO source_file")
                split_counts[split] = first_index
                window_hashes[split] = previous_hash
                rejected.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "genre": row["genre"],
                        "split": split,
                        "error": str(exc),
                    }
                )
                continue
            finally:
                connection.execute("RELEASE source_file")
            retained[row["genre"]] += 1
            retained_splits[row["genre"], split] += 1
            if split == "train":
                for name, values in encoded["counts"].items():
                    counters[name][row["genre"]].update(values)
            reports.append(
                {
                    "candidate_id": row["candidate_id"],
                    "genre": row["genre"],
                    "split": split,
                    "windows": count,
                    **encoded["report"],
                }
            )
            if index % 100 == 0:
                print(f"Prepared {index}/{len(rows)} MIDI files into SQLite", flush=True)
        report = {
            "selected_files": len(rows),
            "retained_files": sum(retained.values()),
            "retained_by_genre": dict(retained),
            "retained_by_genre_split": {
                genre: {split: retained_splits[genre, split] for split in SPLITS}
                for genre in GENRES
            },
            "windows_by_split": split_counts,
            "rejected_files": rejected,
            "file_reports": reports,
        }
        temporary_report = cache_dir / (staging.name + ".report.json")
        temporary_report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        os.replace(temporary_report, cache_dir / "preparation_report.json")
        for genre in GENRES:
            if not min_per_genre <= retained[genre] <= max_per_genre:
                raise ValueError(
                    f"{genre}: retained {retained[genre]} files, expected "
                    f"{min_per_genre}–{max_per_genre}; inspect preparation_report.json"
                )
            if any(not retained_splits[genre, split] for split in SPLITS):
                raise ValueError(f"{genre}: a split has no usable MIDI files")
        dataset_sha = _identity(
            {
                "identity": identity,
                "window_hashes": {
                    split: checksum.hexdigest() for split, checksum in window_hashes.items()
                },
                "counts": counters,
            }
        )
        metadata = {
            "dataset_kind": identity.get("dataset_kind", "EXP9_APPROVED_MIDI"),
            "dataset_sha256": dataset_sha,
            "vocab_sha256": identity["vocab_sha256"],
            "pipeline_version": identity["pipeline_version"],
            "program_counts_train": {
                genre: dict(counters["program_counts"][genre]) for genre in GENRES
            },
            "drum_pitch_counts_train": {
                genre: dict(counters["drum_pitch_counts"][genre]) for genre in GENRES
            },
            "meter_counts_train": {
                genre: dict(counters["meter_counts"][genre]) for genre in GENRES
            },
            "source_audit": audit_manifest(
                manifest_path,
                dataset_root,
                min_per_genre=min_per_genre,
                max_per_genre=max_per_genre,
                verify_hashes=True,
            ),
        }
        connection.commit()
        connection.close()
        connection = None
        descriptor = {
            "format": CACHE_FORMAT,
            "identity": identity,
            "dataset_sha256": dataset_sha,
            "metadata": metadata,
            "report": report,
            "split_counts": split_counts,
            "database_sha256": digest(staging / "windows.sqlite3"),
        }
        descriptor["content_sha256"] = _identity(descriptor)
        (staging / "metadata.json").write_text(
            json.dumps(descriptor, allow_nan=False, separators=(",", ":")) + "\n"
        )
        staging.rename(directory)
    finally:
        if connection is not None:
            connection.close()
        if temporary_report is not None:
            temporary_report.unlink(missing_ok=True)
        if staging.exists():
            shutil.rmtree(staging)
    return _load(directory, identity)
