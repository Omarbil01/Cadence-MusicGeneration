"""CPU-first training, checkpoint selection, and application-compatible export.

The public notebook is the entry point. This module keeps its implementation
testable and shares the application's model, MIDI reader, and vocabulary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from musicgen.generation.generator import MidiGenerator
from musicgen.generation.runtime import structure
from musicgen.generation.runtime.model import MusicTransformer
from musicgen.generation.runtime.tokenizer import GENRES, VOCAB_SHA
from musicgen.shared.config import Settings
from musicgen.training import preprocessing
from musicgen.training.augmentation import augment_window
from musicgen.training.cache import SQLiteWindows, prepare_sqlite_dataset
from musicgen.training.manifest import audit_manifest, digest, load_approved, source_path

PIPELINE_VERSION = "cadence-cpu-training-v2"


def _json_write(path: Path, value: dict, *, compact=False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    options = {"separators": (",", ":")} if compact else {"indent": 2}
    temporary.write_text(json.dumps(value, allow_nan=False, **options) + "\n")
    os.replace(temporary, path)


def _identity(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare_dataset(
    manifest_path,
    dataset_root,
    cache_dir,
    *,
    context=256,
    min_per_genre=200,
    max_per_genre=250,
    storage="json",
    normalize_tempo=False,
    progress_bins=0,
) -> dict:
    """Verify sources and prepare cached windows without changing supplied splits.

    Invalid MIDI files are reported and excluded, with a hard minimum retained
    file count for each genre. Instrument statistics use retained training files
    only. File hashes are rechecked even when a prepared cache already exists.
    """
    if isinstance(context, bool) or not isinstance(context, int) or context < 16:
        raise ValueError("Prepared context must be an integer of at least 16")
    if storage not in {"json", "sqlite"}:
        raise ValueError("Prepared storage must be 'json' or 'sqlite'")
    if not isinstance(normalize_tempo, bool):
        raise ValueError("normalize_tempo must be a boolean")
    if (
        isinstance(progress_bins, bool)
        or not isinstance(progress_bins, int)
        or (progress_bins != 0 and not 2 <= progress_bins <= 64)
    ):
        raise ValueError("progress_bins must be zero or between 2 and 64")
    rows = load_approved(
        manifest_path,
        midi_root=dataset_root,
        min_per_genre=min_per_genre,
        max_per_genre=max_per_genre,
    )
    identity = {
        "pipeline_version": PIPELINE_VERSION,
        "preprocessing_sha256": digest(preprocessing.__file__),
        "manifest_sha256": digest(manifest_path),
        "vocab_sha256": VOCAB_SHA,
        "context": context,
        "min_per_genre": min_per_genre,
        "max_per_genre": max_per_genre,
    }
    # Preserve the legacy JSON identity when the optional representation is disabled.
    if normalize_tempo or progress_bins:
        identity.update(normalize_tempo=normalize_tempo, progress_bins=progress_bins)
        identity["structure_sha256"] = digest(structure.__file__)
    if all(
        row.get("rules_version")
        and row.get("authorization_basis") == "user_authorized_expanded_training_selection"
        for row in rows
    ):
        identity["dataset_kind"] = "PDMX_CURATED_MIDI"
    if storage == "sqlite":
        return prepare_sqlite_dataset(
            manifest_path,
            dataset_root,
            cache_dir,
            rows,
            identity,
            context=context,
            min_per_genre=min_per_genre,
            max_per_genre=max_per_genre,
            normalize_tempo=normalize_tempo,
            progress_bins=progress_bins,
        )
    dataset_sha = _identity(identity)
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / (dataset_sha + ".json")
    if cache_path.is_file():
        cached = json.loads(cache_path.read_text())
        if cached.get("identity") != identity:
            raise ValueError("Prepared cache identity does not match this run")
        checksum = cached.pop("content_sha256", None)
        if checksum != _identity(cached):
            raise ValueError("Prepared cache content changed; remove the cache and prepare again")
        cached["content_sha256"] = checksum
        return cached
    splits = {split: [] for split in ("train", "val", "test")}
    retained, retained_splits = Counter(), Counter()
    counters = {
        name: {genre: Counter() for genre in GENRES}
        for name in ("program_counts", "drum_pitch_counts", "meter_counts")
    }
    reports, rejected = [], []
    for index, row in enumerate(rows, 1):
        try:
            options = (
                {"normalize_tempo": normalize_tempo, "progress_bins": progress_bins}
                if normalize_tempo or progress_bins
                else {}
            )
            encoded = preprocessing.encode_midi(
                source_path(dataset_root, row["midi_path"]), **options
            )
            windows = [
                window
                for sequence in encoded["sequences"]
                for window in preprocessing.make_windows(
                    sequence["ids"],
                    context,
                    **(
                        {
                            "progress_ids": sequence.get("progress_ids"),
                            "supervise_eos": sequence.get("is_piece_end", True),
                        }
                        if normalize_tempo or progress_bins
                        else {}
                    ),
                )
            ]
            if not windows:
                raise ValueError("No usable encoded sequences")
        except (OSError, ValueError) as exc:
            rejected.append(
                {
                    "candidate_id": row["candidate_id"],
                    "genre": row["genre"],
                    "split": row["split"],
                    "error": str(exc),
                }
            )
            continue
        for window in windows:
            splits[row["split"]].append(
                {
                    **window,
                    "candidate_id": row["candidate_id"],
                    "composition_group": row["composition_group"],
                    "genre_id": GENRES.index(row["genre"]),
                }
            )
        retained[row["genre"]] += 1
        retained_splits[row["genre"], row["split"]] += 1
        if row["split"] == "train":
            for name, values in encoded["counts"].items():
                counters[name][row["genre"]].update(values)
        reports.append(
            {
                "candidate_id": row["candidate_id"],
                "genre": row["genre"],
                "split": row["split"],
                "windows": len(windows),
                **encoded["report"],
            }
        )
        if index % 100 == 0:
            print(f"Prepared {index}/{len(rows)} MIDI files", flush=True)
    report = {
        "selected_files": len(rows),
        "retained_files": sum(retained.values()),
        "retained_by_genre": dict(retained),
        "retained_by_genre_split": {
            genre: {split: retained_splits[genre, split] for split in splits} for genre in GENRES
        },
        "windows_by_split": {split: len(windows) for split, windows in splits.items()},
        "rejected_files": rejected,
        "file_reports": reports,
    }
    _json_write(cache_dir / "preparation_report.json", report)
    for genre in GENRES:
        if not min_per_genre <= retained[genre] <= max_per_genre:
            raise ValueError(
                f"{genre}: retained {retained[genre]} files, expected "
                f"{min_per_genre}–{max_per_genre}; inspect preparation_report.json"
            )
        if any(not retained_splits[genre, split] for split in splits):
            raise ValueError(f"{genre}: a split has no usable MIDI files")
    dataset_sha = _identity({"identity": identity, "splits": splits, "counts": counters})
    metadata = {
        "dataset_kind": identity.get("dataset_kind", "EXP9_APPROVED_MIDI"),
        "dataset_sha256": dataset_sha,
        "vocab_sha256": VOCAB_SHA,
        "pipeline_version": PIPELINE_VERSION,
        "program_counts_train": {g: dict(counters["program_counts"][g]) for g in GENRES},
        "drum_pitch_counts_train": {g: dict(counters["drum_pitch_counts"][g]) for g in GENRES},
        "meter_counts_train": {g: dict(counters["meter_counts"][g]) for g in GENRES},
        "source_audit": audit_manifest(
            manifest_path,
            dataset_root,
            min_per_genre=min_per_genre,
            max_per_genre=max_per_genre,
            verify_hashes=True,
        ),
    }
    result = {
        "identity": identity,
        "dataset_sha256": dataset_sha,
        "metadata": metadata,
        "report": report,
        "splits": splits,
    }
    result["content_sha256"] = _identity(result)
    _json_write(cache_path, result, compact=True)
    return result


@dataclass(frozen=True)
class TrainingConfig:
    context: int = 256
    d_model: int = 128
    layers: int = 3
    heads: int = 4
    dropout: float = 0.15
    batch_size: int = 8
    grad_accum: int = 4
    epochs: int = 500
    lr: float = 3e-4
    weight_decay: float = 0.01
    patience: int = 50
    min_delta: float = 0.0001
    seed: int = 42
    torch_threads: int = 4
    max_windows_per_file: int = 16
    device: str = "cpu"
    smoke: bool = False
    progress_bins: int = 0
    relative_attention: bool = False
    transpose_semitones: int = 0
    precision: str = "float32"
    val_windows_per_file: int = 0

    def __post_init__(self):
        counts = (
            self.context,
            self.d_model,
            self.layers,
            self.heads,
            self.batch_size,
            self.grad_accum,
            self.epochs,
            self.patience,
            self.torch_threads,
            self.max_windows_per_file,
        )
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts):
            raise ValueError("Model and training counts must be positive integers")
        if self.context < 16 or self.d_model % self.heads or self.d_model % 2:
            raise ValueError(
                "Context must be >=16; model width must be even and divisible by heads"
            )
        if not 0 <= self.dropout < 1 or self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid dropout, learning rate or weight decay")
        if self.min_delta < 0 or self.device not in {"cpu", "cuda"}:
            raise ValueError("Invalid early-stopping delta or device")
        if (
            isinstance(self.progress_bins, bool)
            or not isinstance(self.progress_bins, int)
            or (self.progress_bins != 0 and not 2 <= self.progress_bins <= 64)
        ):
            raise ValueError("Progress bins must be zero or between 2 and 64")
        if not isinstance(self.relative_attention, bool):
            raise ValueError("relative_attention must be boolean")
        if (
            isinstance(self.transpose_semitones, bool)
            or not isinstance(self.transpose_semitones, int)
            or not 0 <= self.transpose_semitones <= 5
        ):
            raise ValueError("Transposition radius must be an integer between 0 and 5")
        if self.precision not in {"float32", "amp_float16", "amp_bfloat16"}:
            raise ValueError("Precision must be float32, amp_float16 or amp_bfloat16")
        if (
            isinstance(self.val_windows_per_file, bool)
            or not isinstance(self.val_windows_per_file, int)
            or self.val_windows_per_file < 0
            or self.val_windows_per_file == 1
        ):
            raise ValueError("Validation window cap must be zero or an integer of at least two")

    def model_config(self):
        result = {
            key: getattr(self, key) for key in ("context", "d_model", "layers", "heads", "dropout")
        }
        if self.progress_bins:
            result["progress_bins"] = self.progress_bins
        if self.relative_attention:
            result["relative_attention"] = True
        return result

    def record(self):
        """Leave legacy signatures unchanged when new options are disabled."""
        values = asdict(self)
        if not self.progress_bins:
            values.pop("progress_bins")
        if not self.relative_attention:
            values.pop("relative_attention")
        if not self.transpose_semitones:
            values.pop("transpose_semitones")
        if self.precision == "float32":
            values.pop("precision")
        if not self.val_windows_per_file:
            values.pop("val_windows_per_file")
        return values


class _Windows(Dataset):
    def __init__(self, windows, *, progress_bins=0, transpose_semitones=0, seed=42, epoch=None):
        self.windows = windows
        self.progress_bins = progress_bins
        self.transpose_semitones = transpose_semitones
        self.seed, self.epoch = seed, epoch

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        row = self.windows[index]
        if self.epoch is not None and self.transpose_semitones:
            row = augment_window(
                row, radius=self.transpose_semitones, seed=self.seed, epoch=self.epoch, index=index
            )
        values = (
            torch.tensor(row["input_ids"]),
            torch.tensor(row["labels"]),
            row["genre_id"],
            row["composition_group"],
        )
        if self.progress_bins:
            progress = row.get("progress_ids")
            if (
                progress is None
                or len(progress) != len(row["input_ids"])
                or any(not 0 <= value < self.progress_bins for value in progress)
            ):
                raise ValueError("Prepared window has missing or invalid progress conditioning")
            values += (torch.tensor(progress),)
        return values


def _loader(windows, config, indices=None, *, epoch=None):
    dataset = _Windows(
        windows,
        progress_bins=config.progress_bins,
        transpose_semitones=config.transpose_semitones,
        seed=config.seed,
        epoch=epoch,
    )
    if indices is not None:
        dataset = Subset(dataset, indices)
    # No worker subprocesses or pinned memory are needed for this CPU-first pipeline.
    return DataLoader(dataset, batch_size=config.batch_size, num_workers=0)


def _device_batch(batch, config):
    x, y, genres, groups = batch[:4]
    features = batch[4].to(config.device) if len(batch) == 5 else None
    return x.to(config.device), y.to(config.device), genres.to(config.device), groups, features


def _forward(model, x, genres, features):
    return model(x, genres) if features is None else model(x, genres, features)


def _epoch_indices(windows, config, epoch):
    """Rotate file windows and balance genres without mixing source splits."""
    rng = random.Random(config.seed + epoch)
    genres = defaultdict(list)
    if isinstance(windows, SQLiteWindows):
        file_indices = windows.file_indices()
    else:
        files = defaultdict(list)
        for index, row in enumerate(windows):
            files[row["candidate_id"]].append(index)
        file_indices = ((windows[indices[0]]["genre_id"], indices) for indices in files.values())
    for genre_id, indices in file_indices:
        limit = min(config.max_windows_per_file, len(indices))
        if getattr(config, "progress_bins", 0):
            selected = rng.sample(indices[:-1], limit - 1) + [indices[-1]]
        else:
            selected = rng.sample(indices, limit)
        genres[genre_id].extend(selected)
    if set(genres) != set(range(len(GENRES))):
        raise ValueError("Training requires windows from all three genres")
    size = max(map(len, genres.values()))
    result = []
    for indices in genres.values():
        result.extend(indices + rng.choices(indices, k=size - len(indices)))
    rng.shuffle(result)
    return result


def _new_model(config):
    torch.set_num_threads(config.torch_threads)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; use device='cpu'")
    if config.precision != "float32" and config.device != "cuda":
        raise ValueError("CUDA mixed precision requires device='cuda'")
    if config.precision == "amp_bfloat16" and not torch.cuda.is_bf16_supported(
        including_emulation=False
    ):
        raise ValueError("This CUDA GPU does not support bfloat16; choose amp_float16")
    torch.manual_seed(config.seed)
    return MusicTransformer(**config.model_config()).to(config.device)


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(torch.device(device))


def _autocast(config):
    if config.precision == "float32":
        return nullcontext()
    dtype = torch.float16 if config.precision == "amp_float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _new_scaler(config):
    return torch.amp.GradScaler("cuda") if config.precision == "amp_float16" else None


def _backward(loss, config, scaler):
    if scaler is None:
        loss.backward()
        return 1
    # Bound the scaled FP16 backward signal before normalization. Correct for
    # padding and partial groups once, using the actual accumulated token count.
    divisor = config.batch_size * config.context * config.grad_accum
    scaler.scale(loss / divisor).backward()
    return divisor


def _finish_update(model, optimizer, scaler, tokens, *, backward_divisor=1):
    """Normalize the accumulated token sum, then clip and perform one AMP-safe update."""
    if scaler is not None:
        scaler.unscale_(optimizer)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.div_(max(1, tokens) / backward_divisor)
    # Overflow is recoverable in FP16: GradScaler records it before clipping and
    # skips the update. Other precision modes retain the strict finite guard.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=scaler is None)
    if scaler is None:
        optimizer.step()
        stepped = True
    else:
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        stepped = scaler.get_scale() >= previous_scale
    optimizer.zero_grad(set_to_none=True)
    return stepped


def _validation_indices(windows, config):
    """Select fixed first/last and evenly spaced windows from every validation file."""
    cap = config.val_windows_per_file
    if not cap:
        return None
    if isinstance(windows, SQLiteWindows):
        files = windows.file_indices()
    else:
        grouped = defaultdict(list)
        for index, row in enumerate(windows):
            grouped[row["candidate_id"]].append(index)
        files = ((None, indices) for indices in grouped.values())
    result = []
    for _, indices in files:
        if len(indices) <= cap:
            result.extend(indices)
        else:
            result.extend(
                indices[round(index * (len(indices) - 1) / (cap - 1))] for index in range(cap)
            )
    return result


def _validation_protocol(windows, config, indices):
    return {
        "strategy": "fixed_evenly_spaced_per_file" if indices is not None else "all_windows",
        "max_windows_per_file": config.val_windows_per_file,
        "available_windows": len(windows),
        "selected_windows": len(indices) if indices is not None else len(windows),
        "indices_sha256": _identity(indices) if indices is not None else None,
        "aggregation": "mean_composition_loss_per_genre_then_equal_genre_mean",
    }


def _evaluate(model, windows, config, *, indices=None):
    """Mean file-group loss per genre, then an equal mean across the genres."""
    sums, counts = defaultdict(float), Counter()
    model.eval()
    with torch.inference_mode():
        loader = _loader(windows, config) if indices is None else _loader(windows, config, indices)
        for batch in loader:
            genre_ids = batch[2].tolist()
            row_counts = batch[1].ne(-100).sum(dim=1).tolist()
            x, y, genres, groups, features = _device_batch(batch, config)
            with _autocast(config):
                losses = F.cross_entropy(
                    _forward(model, x, genres, features).flatten(0, 1),
                    y.flatten(),
                    ignore_index=-100,
                    reduction="none",
                ).reshape(y.shape)
            # Transfer batch summaries once instead of synchronizing each GPU row.
            row_sums = losses.sum(dim=1).tolist()
            for genre, group, loss_sum, count in zip(genre_ids, groups, row_sums, row_counts):
                key = (genre, group)
                sums[key] += loss_sum
                counts[key] += count
    per_genre = {}
    for genre_id, genre in enumerate(GENRES):
        values = [sums[key] / counts[key] for key in counts if key[0] == genre_id and counts[key]]
        if not values:
            raise ValueError(f"No evaluation targets for {genre}")
        per_genre[genre] = sum(values) / len(values)
    return {"loss": sum(per_genre.values()) / len(GENRES), "per_genre": per_genre}


def benchmark(prepared, config, *, steps=3):
    """Measure loader-inclusive training/validation batches without checkpoint writes."""
    if steps < 1:
        raise ValueError("Benchmark steps must be positive")
    model = _new_model(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scaler = _new_scaler(config)
    indices = _epoch_indices(prepared["splits"]["train"], config, 0)
    loader_options = {"epoch": 0} if config.transpose_semitones else {}
    loader = _loader(prepared["splits"]["train"], config, indices, **loader_options)
    validation_indices = _validation_indices(prepared["splits"]["val"], config)
    validation_loader = _loader(prepared["splits"]["val"], config, validation_indices)

    def backward_batch(batch):
        tokens = int(batch[1].ne(-100).sum())
        x, y, genres, _, features = _device_batch(batch, config)
        with _autocast(config):
            loss = F.cross_entropy(
                _forward(model, x, genres, features).flatten(0, 1),
                y.flatten(),
                ignore_index=-100,
                reduction="sum",
            )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite benchmark loss")
        return tokens, _backward(loss, config, scaler)

    # Warm up kernels and Adam state before measuring. No real run is modified.
    model.train()
    optimizer.zero_grad(set_to_none=True)
    tokens, divisor = backward_batch(next(iter(loader)))
    _finish_update(model, optimizer, scaler, tokens, backward_divisor=divisor)
    _synchronize(config.device)
    timings = []
    iterator = iter(loader)
    tokens, position, updates, skipped = 0, 0, 0, 0
    for index in range(1, steps + 1):
        _synchronize(config.device)
        started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            position = 0
            batch = next(iterator)
        position += 1
        count, divisor = backward_batch(batch)
        tokens += count
        if position % config.grad_accum == 0 or position == len(loader) or index == steps:
            stepped = _finish_update(model, optimizer, scaler, tokens, backward_divisor=divisor)
            updates += int(stepped)
            skipped += int(not stepped)
            tokens = 0
        _synchronize(config.device)
        timings.append(time.perf_counter() - started)
    if not updates:
        raise FloatingPointError("Benchmark made no optimizer updates; inspect AMP overflow")
    seconds = sum(timings) / max(1, len(timings))
    validation_timings = []
    model.eval()
    iterator = iter(validation_loader)
    with torch.inference_mode():
        for index in range(steps + 1):
            _synchronize(config.device)
            started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(validation_loader)
                batch = next(iterator)
            x, y, genres, _, features = _device_batch(batch, config)
            with _autocast(config):
                losses = F.cross_entropy(
                    _forward(model, x, genres, features).flatten(0, 1),
                    y.flatten(),
                    ignore_index=-100,
                    reduction="none",
                ).reshape(y.shape)
            if not bool(torch.isfinite(losses).all()):
                raise FloatingPointError("Nonfinite benchmark validation loss")
            losses.sum(dim=1).tolist()
            _synchronize(config.device)
            if index:
                validation_timings.append(time.perf_counter() - started)
    validation_seconds = sum(validation_timings) / len(validation_timings)
    training_estimate = seconds * len(loader)
    validation_estimate = validation_seconds * len(validation_loader)
    protocol = _validation_protocol(prepared["splits"]["val"], config, validation_indices)
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "measured_batches": len(timings),
        "seconds_per_batch": seconds,
        "batches_per_epoch": len(loader),
        "training_windows": len(indices),
        "optimizer_updates_measured": updates,
        "skipped_optimizer_updates": skipped,
        "precision": config.precision,
        "validation_batches_per_epoch": len(validation_loader),
        "seconds_per_validation_batch": validation_seconds,
        "full_validation_windows": protocol["available_windows"],
        "validation_windows_evaluated": protocol["selected_windows"],
        "validation_protocol": protocol,
        "estimated_training_seconds_per_epoch": training_estimate,
        "estimated_validation_seconds_per_epoch": validation_estimate,
        "estimated_epoch_seconds": training_estimate + validation_estimate,
        "note": (
            "Includes source-cache loading, transfers, training updates and validation; "
            "excludes checkpoint/history writes and variable Drive I/O. "
            "A short sample is approximate; actual epoch timings are logged."
        ),
    }


def _save(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(prepared, run_dir, config, *, resume_from=None):
    """Train from scratch or resume at an epoch boundary using validation selection."""
    if prepared["identity"]["context"] != config.context:
        raise ValueError("Prepared data and model context differ")
    if prepared["identity"].get("progress_bins", 0) != config.progress_bins:
        raise ValueError("Prepared data and model progress conditioning differ")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume_from is None and any(run_dir.iterdir()):
        raise FileExistsError("Use a new run directory or resume from its last.pt checkpoint")
    model = _new_model(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scaler = _new_scaler(config)
    validation_indices = _validation_indices(prepared["splits"]["val"], config)
    validation_protocol = _validation_protocol(
        prepared["splits"]["val"], config, validation_indices
    )
    history, best_loss, stale, start_epoch, optimizer_steps = [], math.inf, 0, 0, 0
    skipped_updates = 0
    signature = {
        "vocab_sha256": VOCAB_SHA,
        "dataset_sha256": prepared["dataset_sha256"],
        "model_config": model.config,
        "training_config": config.record(),
    }
    if config.val_windows_per_file:
        signature["validation_protocol"] = validation_protocol
    if resume_from is not None:
        # Loading on CPU keeps RNG tensors and Adam step counters off the GPU.
        # load_state_dict then places model weights and optimizer moments correctly.
        previous = torch.load(resume_from, map_location="cpu", weights_only=True)
        if previous["signature"] != signature:
            raise ValueError("Resume requires the same dataset, model and training configuration")
        if Path(resume_from).resolve().parent != run_dir.resolve():
            raise ValueError("Resume in the original run directory to preserve its best checkpoint")
        model.load_state_dict(previous["model"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        if scaler is not None:
            if "scaler" not in previous:
                raise ValueError("FP16 resume checkpoint is missing its gradient-scaler state")
            scaler.load_state_dict(previous["scaler"])
        history, best_loss = previous["history"], previous["best_loss"]
        stale, start_epoch = previous["stale_epochs"], previous["epoch"]
        optimizer_steps = previous["optimizer_steps"]
        skipped_updates = previous.get("skipped_updates", 0)
        torch.set_rng_state(previous["torch_rng_state"].cpu())
        if config.device == "cuda" and "cuda_rng_state" in previous:
            torch.cuda.set_rng_state(previous["cuda_rng_state"].cpu())
        if start_epoch and not (run_dir / "best.pt").is_file():
            raise FileNotFoundError("The run's validation-selected best.pt is missing")
    _json_write(
        run_dir / "run.json",
        {
            "signature": signature,
            "environment": {
                "python": platform.python_version(),
                "pytorch": str(torch.__version__),
                "platform": platform.platform(),
                "torch_threads": config.torch_threads,
                "cuda_runtime": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name() if config.device == "cuda" else None,
            },
            "dataset_report": {
                key: value for key, value in prepared["report"].items() if key != "file_reports"
            },
            "validation_protocol": validation_protocol,
            "precision": config.precision,
        },
    )
    batches = math.ceil(
        len(_epoch_indices(prepared["splits"]["train"], config, 0)) / config.batch_size
    )
    total_steps = max(1, math.ceil(batches / config.grad_accum) * config.epochs)
    warmup = max(1, round(total_steps * 0.05))
    for epoch in range(start_epoch, config.epochs):
        if stale >= config.patience:
            break
        _synchronize(config.device)
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loader_options = {"epoch": epoch} if config.transpose_semitones else {}
        loader = _loader(
            prepared["splits"]["train"],
            config,
            _epoch_indices(prepared["splits"]["train"], config, epoch),
            **loader_options,
        )
        total_loss, total_tokens, accumulation_tokens = 0.0, 0, 0
        for batch, values in enumerate(loader, 1):
            tokens = int(values[1].ne(-100).sum())
            x, y, genres, _, features = _device_batch(values, config)
            with _autocast(config):
                loss = F.cross_entropy(
                    _forward(model, x, genres, features).flatten(0, 1),
                    y.flatten(),
                    ignore_index=-100,
                    reduction="sum",
                )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            backward_divisor = _backward(loss, config, scaler)
            total_loss += (
                loss.detach().double() if config.device == "cuda" else float(loss.detach())
            )
            total_tokens += tokens
            accumulation_tokens += tokens
            if batch % config.grad_accum == 0 or batch == len(loader):
                next_step = optimizer_steps + 1
                if next_step <= warmup:
                    factor = next_step / warmup
                else:
                    progress = (next_step - warmup) / max(1, total_steps - warmup)
                    factor = 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
                for group in optimizer.param_groups:
                    group["lr"] = config.lr * factor
                stepped = _finish_update(
                    model,
                    optimizer,
                    scaler,
                    accumulation_tokens,
                    backward_divisor=backward_divisor,
                )
                optimizer_steps += int(stepped)
                skipped_updates += int(not stepped)
                accumulation_tokens = 0
        _synchronize(config.device)
        train_seconds = time.perf_counter() - started
        validation_started = time.perf_counter()
        if validation_indices is None:
            validation = _evaluate(model, prepared["splits"]["val"], config)
        else:
            validation = _evaluate(
                model, prepared["splits"]["val"], config, indices=validation_indices
            )
        _synchronize(config.device)
        validation_seconds = time.perf_counter() - validation_started
        val_loss = validation["loss"]
        if not math.isfinite(val_loss):
            raise FloatingPointError("Nonfinite validation loss")
        improved = val_loss < best_loss - config.min_delta
        if improved:
            best_loss, stale = val_loss, 0
        else:
            stale += 1
        record = {
            "epoch": epoch + 1,
            "train_loss": float(total_loss) / max(1, total_tokens),
            "val_loss": val_loss,
            "val_per_genre": validation["per_genre"],
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "epoch_seconds": time.perf_counter() - started,
            "lr": optimizer.param_groups[0]["lr"],
            "optimizer_steps": optimizer_steps,
        }
        if scaler is not None:
            record["skipped_updates"] = skipped_updates
        history.append(record)
        checkpoint = {
            "model": model.state_dict(),
            "epoch": epoch + 1,
            "signature": signature,
            "dataset_metadata": prepared["metadata"],
            "optimizer": optimizer.state_dict(),
            "optimizer_steps": optimizer_steps,
            "history": history,
            "best_loss": best_loss,
            "stale_epochs": stale,
            "torch_rng_state": torch.get_rng_state(),
        }
        if config.device == "cuda":
            checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state()
        if scaler is not None:
            checkpoint["scaler"] = scaler.state_dict()
            checkpoint["skipped_updates"] = skipped_updates
        if improved:
            _save(
                run_dir / "best.pt",
                {
                    key: checkpoint[key]
                    for key in ("model", "epoch", "signature", "dataset_metadata")
                },
            )
        # Commit resume state only after the corresponding selected model is durable.
        _save(run_dir / "last.pt", checkpoint)
        _json_write(run_dir / "history.json", {"history": history, "signature": signature})
        print(
            f"Epoch {epoch + 1}: train={record['train_loss']:.4f} "
            f"val={val_loss:.4f} ({record['epoch_seconds']:.1f}s)",
            flush=True,
        )
    return {
        "best_checkpoint": str(run_dir / "best.pt"),
        "last_checkpoint": str(run_dir / "last.pt"),
        "history": history,
        "epochs_completed": len(history),
        "stopped_early": stale >= config.patience,
    }


def smoke_train(prepared, run_dir, *, device="cpu", precision="float32"):
    """Run one tiny epoch; this checkpoint is explicitly rejected by the app."""
    small = {**prepared, "splits": {}}
    for split, windows in prepared["splits"].items():
        if split == "test":
            small["splits"][split] = windows
            continue
        if isinstance(windows, SQLiteWindows):
            small["splits"][split] = windows.take_per_genre(2)
        else:
            selected = {genre: [] for genre in range(len(GENRES))}
            for row in windows:
                if len(selected[row["genre_id"]]) < 2:
                    selected[row["genre_id"]].append(row)
                if all(len(values) == 2 for values in selected.values()):
                    break
            small["splits"][split] = [row for values in selected.values() for row in values]
    config = TrainingConfig(
        context=prepared["identity"]["context"],
        d_model=32,
        layers=1,
        heads=4,
        batch_size=2,
        grad_accum=1,
        epochs=1,
        max_windows_per_file=2,
        torch_threads=2,
        progress_bins=prepared["identity"].get("progress_bins", 0),
        relative_attention=bool(prepared["identity"].get("progress_bins", 0)),
        device=device,
        precision=precision,
        smoke=True,
    )
    return train(small, run_dir, config)


def evaluate_test(best_checkpoint, prepared, *, device="cpu"):
    """Evaluate the held-out split after selection; repeat calls reuse saved metrics."""
    path = Path(best_checkpoint)
    record_path = path.parent / "test_metrics.json"
    identity = {"checkpoint_sha256": digest(path), "dataset_sha256": prepared["dataset_sha256"]}
    if record_path.is_file():
        previous = json.loads(record_path.read_text())
        if previous["identity"] != identity:
            raise ValueError("Test metrics already exist for another selected checkpoint")
        return previous
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    signature = checkpoint["signature"]
    if signature["dataset_sha256"] != prepared["dataset_sha256"]:
        raise ValueError("Checkpoint and held-out data differ")
    if signature["training_config"].get("smoke"):
        raise ValueError("Smoke runs are not held-out evaluations")
    config = replace(
        TrainingConfig(**signature["training_config"]),
        device=device,
        precision="float32"
        if device == "cpu"
        else signature["training_config"].get("precision", "float32"),
    )
    model = _new_model(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    result = {
        "identity": identity,
        "device": device,
        "precision": config.precision,
        **_evaluate(model, prepared["splits"]["test"], config),
    }
    _json_write(record_path, result)
    return result


def generate_samples(checkpoint, output_dir, *, seconds=5, seeds=(42,), device="cpu"):
    """Generate fixed-seed listening samples through the actual application runtime."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine = MidiGenerator(
        Settings(_env_file=None, checkpoint_path=Path(checkpoint), device=device)
    )
    engine.load()
    generation_root = Path(__file__).resolve().parents[1] / "generation"
    profile = _identity(
        {
            "source_sha256": {
                name: digest(generation_root / name)
                for name in (
                    "generator.py",
                    "musical.py",
                    "runtime/model.py",
                    "runtime/grammar.py",
                    "runtime/context.py",
                    "runtime/midi.py",
                    "runtime/structure.py",
                )
            },
            "ending_style": engine.settings.ending_style,
            "repetition_penalty": engine.settings.repetition_penalty,
            "palette": engine.settings.palette,
            "torch_threads": engine.settings.torch_threads,
        }
    )
    results = []
    for genre in GENRES:
        for seed in seeds:
            name = (
                f"{genre}_{seconds}s_seed{seed}_{engine.info['checkpoint_sha256'][:8]}"
                f"_{engine.info['device']}_p{profile[:8]}"
            )
            output, sidecar = output_dir / (name + ".mid"), output_dir / (name + ".json")
            if output.exists() or sidecar.exists():
                if output.exists() and sidecar.exists():
                    results.append(json.loads(sidecar.read_text()))
                    continue
                raise FileExistsError("An incomplete prior sample exists; choose a new directory")
            metadata = engine.generate(
                genre=genre, duration_seconds=seconds, output_path=output, seed=seed
            )
            result = {"midi": str(output), "sampling_profile_sha256": profile, **metadata}
            _json_write(sidecar, result)
            results.append(result)
    return results


def export_checkpoint(source, destination):
    """Validate and export inference fields only; preserve any existing deployed model."""
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(
            "Destination already exists; preserve/version the current model first"
        )
    engine = MidiGenerator(Settings(_env_file=None, checkpoint_path=source, device="cpu"))
    engine.load()
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _save(
        destination,
        {key: checkpoint[key] for key in ("model", "epoch", "signature", "dataset_metadata")},
    )
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train the app-compatible MIDI Transformer on CPU."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--midi-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("storage/training/cache"))
    parser.add_argument("--epochs", type=int, help="Maximum epochs for a new run (default: 500)")
    parser.add_argument(
        "--patience", type=int, help="Epochs without validation improvement (default: 50)"
    )
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke and args.resume:
        parser.error("Smoke checks use a new output directory; they cannot resume full runs")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        config = TrainingConfig(**saved["signature"]["training_config"])
        if config.device != "cpu":
            parser.error("This command resumes CPU runs only")
        for key in ("epochs", "patience", "torch_threads"):
            if getattr(args, key) is not None and getattr(args, key) != getattr(config, key):
                parser.error("Resume must keep the saved training configuration")
    else:
        overrides = {
            key: getattr(args, key)
            for key in ("epochs", "patience", "torch_threads")
            if getattr(args, key) is not None
        }
        config = replace(TrainingConfig(), **overrides)
    prepared = prepare_dataset(args.manifest, args.midi_root, args.cache, context=config.context)
    print("Usable files:", prepared["report"]["retained_by_genre"], flush=True)
    result = (
        smoke_train(prepared, args.output)
        if args.smoke
        else train(prepared, args.output, config, resume_from=args.resume)
    )
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
