"""Build and safely unpack a private, checksummed Colab source-data bundle.

This module uses only the standard library until ``build_bundle`` validates the
local manifest. Archive metadata is data; it never supplies executable commands.
Checksums detect corruption, not the trustworthiness of an unknown publisher.
"""

import csv
import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

FORMAT = "cadence-colab-v1"
GENRES = ("classical", "romantic", "rock_pop")
MAX_TOTAL_BYTES = 4_000_000_000
MAX_MEMBER_BYTES = 100_000_000
MAX_MEMBERS = 50_000


def _safe_name(name):
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or str(path) != name
        or "\\" in name
        or ":" in name
        or any(part in {".", ".."} or part.startswith(".") for part in path.parts)
        or any(ord(char) < 32 for char in name)
    ):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return path


def _project_member(name):
    if name in {"project/pyproject.toml", "project/README.md"}:
        return True
    path = PurePosixPath(name)
    if not name.startswith("project/src/musicgen/"):
        return False
    return (
        path.suffix == ".py"
        or name == "project/src/musicgen/generation/runtime/provenance.json"
        or ("/frontend/templates/" in name and path.suffix == ".html")
        or ("/frontend/static/css/" in name and path.suffix == ".css")
        or ("/frontend/static/js/" in name and path.suffix == ".js")
    )


def _allowed_member(name):
    return (
        name in {"bundle.json", "dataset/selected.csv"}
        or _project_member(name)
        or (name.startswith("dataset/") and PurePosixPath(name).suffix.lower() in {".mid", ".midi"})
    )


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key in bundle metadata")
        result[key] = value
    return result


def _inspect(archive, max_total_bytes, max_member_bytes, max_members):
    infos = archive.infolist()
    if len(infos) > max_members or not infos:
        raise ValueError("Archive member count exceeds the permitted limit")
    names = set()
    total = 0
    for info in infos:
        _safe_name(info.filename)
        mode = stat.S_IFMT(info.external_attr >> 16)
        if (
            info.orig_filename != info.filename
            or info.filename in names
            or not _allowed_member(info.filename)
            or info.is_dir()
            or mode not in {0, stat.S_IFREG}
            or info.flag_bits & 1
        ):
            raise ValueError(f"Unsupported, duplicate or unsafe archive member: {info.filename}")
        if info.file_size > max_member_bytes:
            raise ValueError("Archive member exceeds the permitted size limit")
        total += info.file_size
        names.add(info.filename)
    if total > max_total_bytes:
        raise ValueError("Archive total size exceeds the permitted limit")
    if "bundle.json" not in names or archive.getinfo("bundle.json").file_size > 16_000_000:
        raise ValueError("Missing or oversized bundle metadata")
    try:
        record = json.loads(archive.read("bundle.json"), object_pairs_hook=_json_object)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid bundle metadata") from exc
    if not isinstance(record, dict) or record.get("format") != FORMAT:
        raise ValueError("Unsupported bundle format")
    expected = {
        "project_root": "project",
        "dataset_root": "dataset",
        "manifest_path": "dataset/selected.csv",
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Unexpected project or dataset layout")
    files = record.get("files")
    if not isinstance(files, dict) or set(files) != names - {"bundle.json"}:
        raise ValueError("Bundle file inventory does not match the archive")
    for name, item in files.items():
        if (
            not isinstance(item, dict)
            or type(item.get("size")) is not int
            or item["size"] != archive.getinfo(name).file_size
            or not isinstance(item.get("sha256"), str)
            or len(item["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in item["sha256"])
        ):
            raise ValueError(f"Invalid size or SHA-256 for archive member: {name}")
    required = {
        "project/pyproject.toml",
        "project/README.md",
        "project/src/musicgen/__init__.py",
        "dataset/selected.csv",
    }
    if not required <= files.keys():
        raise ValueError("Bundle is missing required project files")
    if files["dataset/selected.csv"]["sha256"] != record.get("manifest_sha256"):
        raise ValueError("Selected-manifest checksum differs from bundle metadata")
    _check_manifest(archive, record)
    return record


def _check_manifest(archive, record):
    """Check selection integrity before importing any bundled application code."""
    if archive.getinfo("dataset/selected.csv").file_size > 50_000_000:
        raise ValueError("Selected manifest exceeds the permitted size limit")
    try:
        text = archive.read("dataset/selected.csv").decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Selected manifest is not UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text, newline=""))
    required = {
        "candidate_id",
        "sha256",
        "midi_path",
        "genre",
        "genre_id",
        "composition_group",
        "split",
        "training_status",
        "approved_for_training",
    }
    columns = reader.fieldnames or []
    if len(set(columns)) != len(columns) or not required <= set(columns):
        raise ValueError("Invalid selected-manifest columns")
    counts, split_counts = Counter(), Counter()
    identifiers, hashes, members, groups = set(), set(), set(), {}
    for row in reader:
        if (
            None in row
            or any(value is None for value in row.values())
            or any(not row.get(field, "").strip() for field in required)
        ):
            raise ValueError("Invalid or incomplete selected-manifest row")
        row = {key: value.strip() for key, value in row.items()}
        genre, split = row["genre"], row["split"]
        if (
            genre not in GENRES
            or split not in {"train", "val", "test"}
            or row["genre_id"] != str(GENRES.index(genre))
            or row["training_status"] != "APPROVED"
            or row["approved_for_training"].lower() != "true"
        ):
            raise ValueError("Manifest approval, genre or split is invalid")
        member = "dataset/" + str(PurePosixPath(row["midi_path"]))
        _safe_name(member)
        checksum = row["sha256"].lower()
        if (
            member not in record["files"]
            or member == "dataset/selected.csv"
            or record["files"][member]["sha256"] != checksum
            or member in members
            or checksum in hashes
            or row["candidate_id"] in identifiers
        ):
            raise ValueError("Manifest MIDI inventory or source checksum is inconsistent")
        group = row["composition_group"]
        if group in groups and groups[group] != (genre, split):
            raise ValueError("Manifest composition group crosses splits or genre labels")
        groups[group] = genre, split
        identifiers.add(row["candidate_id"])
        hashes.add(checksum)
        members.add(member)
        counts[genre] += 1
        split_counts[genre, split] += 1
    minimum, maximum = record.get("min_per_genre"), record.get("max_per_genre")
    if (
        type(minimum) is not int
        or type(maximum) is not int
        or not 1 <= minimum <= maximum <= MAX_MEMBERS
    ):
        raise ValueError("Invalid selection count bounds")
    if any(not minimum <= counts[genre] <= maximum for genre in GENRES):
        raise ValueError("Selected genre counts do not match the bundle contract")
    if any(
        not split_counts[genre, split] for genre in GENRES for split in ("train", "val", "test")
    ):
        raise ValueError("Selected files must cover every genre and source split")
    dataset_files = {name for name in record["files"] if name.startswith("dataset/")}
    if dataset_files != members | {"dataset/selected.csv"}:
        raise ValueError("Bundle contains unselected dataset files")
    audit = record.get("audit", {})
    if (
        not isinstance(audit, dict)
        or audit.get("rows") != len(members)
        or audit.get("by_genre") != dict(counts)
        or audit.get("manifest_sha256") != record["manifest_sha256"]
    ):
        raise ValueError("Bundle audit summary differs from the selected manifest")


def _copy_verified(archive, record, destination=None):
    for name, item in record["files"].items():
        checksum, size = hashlib.sha256(), 0
        target = None
        if destination is not None:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            target = path.open("xb")
        try:
            with archive.open(name) as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    size += len(block)
                    if size > item["size"]:
                        raise ValueError(f"Archive member exceeds its declared size: {name}")
                    checksum.update(block)
                    if target is not None:
                        target.write(block)
        finally:
            if target is not None:
                target.close()
        if size != item["size"] or checksum.hexdigest() != item["sha256"]:
            raise ValueError(f"Archive checksum mismatch: {name}")


def validate_bundle(
    archive,
    *,
    max_total_bytes=MAX_TOTAL_BYTES,
    max_member_bytes=MAX_MEMBER_BYTES,
    max_members=MAX_MEMBERS,
):
    """Validate layout, selection, limits and every payload checksum without extracting."""
    with zipfile.ZipFile(archive) as source:
        record = _inspect(source, max_total_bytes, max_member_bytes, max_members)
        _copy_verified(source, record)
    return record


def extract_bundle(
    archive,
    destination,
    *,
    max_total_bytes=MAX_TOTAL_BYTES,
    max_member_bytes=MAX_MEMBER_BYTES,
    max_members=MAX_MEMBERS,
):
    """Verify and extract into a new directory; failed extraction leaves no partial data."""
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Extraction destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".cadence-extract-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as source:
            record = _inspect(source, max_total_bytes, max_member_bytes, max_members)
            _copy_verified(source, record, staging)
            (staging / "bundle.json").write_text(json.dumps(record, indent=2) + "\n")
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Extraction destination already exists: {destination}")
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        "project_root": destination / "project",
        "dataset_root": destination / "dataset",
        "manifest_path": destination / "dataset" / "selected.csv",
        "record": record,
    }


def build_bundle(
    manifest,
    midi_root,
    output,
    *,
    project_root=None,
    min_per_genre=200,
    max_per_genre=250,
):
    """Package only reviewed source files and the exact verified MIDI selection."""
    from musicgen.training.manifest import audit_manifest, digest, load_approved, source_path

    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Bundle output already exists: {output}")
    project = Path(project_root).resolve() if project_root else Path(__file__).resolve().parents[3]
    rows = load_approved(
        manifest, midi_root=midi_root, min_per_genre=min_per_genre, max_per_genre=max_per_genre
    )
    sources = {
        "project/pyproject.toml": (project / "pyproject.toml", None),
        "project/README.md": (project / "README.md", None),
        "dataset/selected.csv": (Path(manifest), digest(manifest)),
    }
    for path in sorted((project / "src" / "musicgen").rglob("*")):
        if path.is_file():
            name = "project/" + path.relative_to(project).as_posix()
            if _project_member(name):
                sources[name] = path, None
    for row in rows:
        sources["dataset/" + row["midi_path"]] = (
            source_path(midi_root, row["midi_path"]),
            row["sha256"],
        )
    record = {
        "format": FORMAT,
        "project_root": "project",
        "dataset_root": "dataset",
        "manifest_path": "dataset/selected.csv",
        "min_per_genre": min_per_genre,
        "max_per_genre": max_per_genre,
        "manifest_sha256": digest(manifest),
        "audit": audit_manifest(
            manifest, midi_root, min_per_genre=min_per_genre, max_per_genre=max_per_genre
        ),
        "files": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".cadence-bundle-", dir=output.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, (path, expected_sha) in sorted(sources.items()):
                _safe_name(name)
                if name.startswith("project/") and not path.resolve().is_relative_to(project):
                    raise ValueError(f"Project source symlink escapes the project: {path}")
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = (stat.S_IFREG | 0o644) << 16
                checksum, size = hashlib.sha256(), 0
                with path.open("rb") as source, archive.open(info, "w") as target:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        size += len(block)
                        if size > MAX_MEMBER_BYTES:
                            raise ValueError(f"Source file exceeds the bundle size limit: {name}")
                        checksum.update(block)
                        target.write(block)
                if expected_sha is not None and checksum.hexdigest() != expected_sha:
                    raise ValueError(f"Source changed during bundle creation: {name}")
                record["files"][name] = {"size": size, "sha256": checksum.hexdigest()}
            info = zipfile.ZipInfo("bundle.json", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, json.dumps(record, sort_keys=True, indent=2) + "\n")
        validate_bundle(temporary)
        os.link(temporary, output)  # Atomic publication without overwriting an existing archive.
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "archive": str(output.resolve()),
        "archive_sha256": digest(output),
        "archive_bytes": output.stat().st_size,
        "selected_files": len(rows),
        "by_genre": record["audit"]["by_genre"],
        "manifest_sha256": record["manifest_sha256"],
        "project_files": sum(name.startswith("project/") for name in record["files"]),
    }
