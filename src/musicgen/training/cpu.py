"""CPU-only command runner for the balanced 250-per-genre MIDI experiment."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULTS = {
    "context": 256,
    "d_model": 128,
    "layers": 3,
    "heads": 4,
    "batch_size": 8,
    "grad_accum": 4,
    "epochs": 500,
    "patience": 50,
    "torch_threads": min(4, os.cpu_count() or 1),
    "max_windows_per_file": 8,
    "val_windows_per_file": 8,
    "transpose_semitones": 2,
}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "storage/training/pdmx_balanced250/dataset/selected.csv",
    )
    parser.add_argument(
        "--midi-root", type=Path, default=REPO_ROOT.parent / "music_generation_pytorch_v1"
    )
    parser.add_argument(
        "--output", type=Path, help="New run directory; resume defaults to checkpoint directory"
    )
    parser.add_argument(
        "--cache", type=Path, default=REPO_ROOT / "storage/training/pdmx_balanced250/cache_cpu"
    )
    for name, default in DEFAULTS.items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int,
            help=f"New-run value (default: {default}); resume must retain saved settings",
        )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--prepare-only", action="store_true", help="Verify and cache the dataset only"
    )
    actions.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Estimate CPU training and validation time only",
    )
    actions.add_argument(
        "--smoke",
        action="store_true",
        help="Train a tiny one-epoch pipeline test in a new output directory",
    )
    parser.add_argument("--benchmark-steps", type=int, default=3)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume this CPU experiment's last.pt with its saved configuration",
    )
    parser.add_argument(
        "--max-epoch-minutes",
        type=float,
        help="Stop before training if estimated compute per epoch exceeds this optional budget",
    )
    return parser


def _validate_selection(manifest):
    from musicgen.generation.runtime.tokenizer import GENRES
    from musicgen.training.balanced import SELECTION_POLICY
    from musicgen.training.manifest import load_approved

    rows = load_approved(manifest, min_per_genre=250, max_per_genre=250, verify_hashes=False)
    if any(row.get("selection_policy") != SELECTION_POLICY for row in rows):
        raise ValueError("Use the new balanced250 selected.csv, with 250 usable files per genre")
    if len({row["composition_group"] for row in rows}) != 750:
        raise ValueError("The balanced selection must have 750 distinct composition families")
    counts = Counter((row["genre"], row["split"]) for row in rows)
    if any(
        counts[genre, split] != expected
        for genre in GENRES
        for split, expected in (("train", 175), ("val", 50), ("test", 25))
    ):
        raise ValueError("Expected 175 train, 50 validation and 25 test files per genre")


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if args.benchmark_steps < 1:
        parser.error("--benchmark-steps must be positive")
    if args.max_epoch_minutes is not None and (
        not math.isfinite(args.max_epoch_minutes) or args.max_epoch_minutes <= 0
    ):
        parser.error("--max-epoch-minutes must be finite and positive")
    if args.smoke and args.resume:
        parser.error("Smoke tests use a new output directory and cannot resume a full run")

    # Keep CUDA invisible before importing torch; all helper calls also explicitly
    # select CPU/float32, including when the installed wheel happens to support CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    from musicgen.training.manifest import digest
    from musicgen.training.training import (
        TrainingConfig,
        benchmark,
        prepare_dataset,
        smoke_train,
        train,
    )

    checkpoint = None
    try:
        if args.resume:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
            needed = {"signature", "optimizer", "torch_rng_state", "optimizer_steps", "history"}
            if not needed <= checkpoint.keys():
                raise ValueError(
                    "Resume needs last.pt with optimizer and random state, not best.pt"
                )
            config = TrainingConfig(**checkpoint["signature"]["training_config"])
            if config.device != "cpu" or config.precision != "float32" or config.smoke:
                raise ValueError("Resume requires a full CPU/float32 run from this experiment")
            if config.progress_bins != 16 or not config.relative_attention:
                raise ValueError("This runner resumes the structured balanced CPU experiment")
            for name in DEFAULTS:
                value = getattr(args, name)
                if value is not None and value != getattr(config, name):
                    raise ValueError(f"Resume must keep the saved --{name.replace('_', '-')} value")
            output = args.output or args.resume.resolve().parent
        else:
            values = {
                name: getattr(args, name) if getattr(args, name) is not None else value
                for name, value in DEFAULTS.items()
            }
            config = TrainingConfig(
                **values,
                device="cpu",
                precision="float32",
                progress_bins=16,
                relative_attention=True,
                smoke=False,
            )
            output = args.output or REPO_ROOT / "storage/training/balanced250_cpu_v1"
        output = output.resolve()
        if args.resume and args.resume.resolve().parent != output:
            raise ValueError("Resume must use the checkpoint's original output directory")
        if not args.resume and not (args.prepare_only or args.benchmark_only):
            if output.exists() and (not output.is_dir() or any(output.iterdir())):
                raise FileExistsError(
                    "Output is not empty; choose a new --output or resume its last.pt"
                )
        _validate_selection(args.manifest)
        if checkpoint is not None:
            source_sha = (
                checkpoint.get("dataset_metadata", {})
                .get("source_audit", {})
                .get("manifest_sha256")
            )
            if source_sha != digest(args.manifest):
                raise ValueError("Resume manifest differs from the run's recorded source selection")
        prepared = prepare_dataset(
            args.manifest,
            args.midi_root,
            args.cache,
            context=config.context,
            min_per_genre=250,
            max_per_genre=250,
            storage="sqlite",
            normalize_tempo=True,
            progress_bins=config.progress_bins,
        )
        if (
            checkpoint is not None
            and checkpoint["signature"]["dataset_sha256"] != prepared["dataset_sha256"]
        ):
            raise ValueError(
                "Resume requires the original prepared dataset and preprocessing version"
            )
        summary = {
            "device": "cpu",
            "precision": "float32",
            "configuration": asdict(config),
            "manifest": str(args.manifest.resolve()),
            "midi_root": str(args.midi_root.resolve()),
            "output": str(output),
            "dataset_sha256": prepared["dataset_sha256"],
            "files_by_genre": prepared["report"]["retained_by_genre"],
            "files_by_genre_split": prepared["report"]["retained_by_genre_split"],
            "windows_by_split": prepared["report"]["windows_by_split"],
        }
        print(json.dumps(summary, indent=2), flush=True)
        # train() requires a fresh run directory. Keep all preflight artifacts in
        # its parent so preparation and benchmarking do not make that run nonempty.
        output.parent.mkdir(parents=True, exist_ok=True)
        preview = output.parent / f"{output.name}.preflight.json"
        if args.prepare_only:
            preview.write_text(json.dumps(summary, indent=2) + "\n")
            print(f"Dataset preparation complete. Preflight report: {preview}", flush=True)
            return 0
        if args.smoke:
            result = smoke_train(prepared, output, device="cpu", precision="float32")
        else:
            measured = benchmark(prepared, config, steps=args.benchmark_steps)
            summary["benchmark"] = measured
            preview.write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"benchmark": measured}, indent=2), flush=True)
            print(
                "Estimated training + validation: "
                f"{measured['estimated_epoch_seconds'] / 60:.2f} minutes/epoch; "
                "checkpoint writes add time.",
                flush=True,
            )
            print(f"Preflight report: {preview}", flush=True)
            if args.benchmark_only:
                return 0
            if (
                args.max_epoch_minutes is not None
                and measured["estimated_epoch_seconds"] > args.max_epoch_minutes * 60
            ):
                raise ValueError(
                    "Estimated epoch exceeds --max-epoch-minutes; adjust the profile or budget"
                )
            result = train(prepared, output, config, resume_from=args.resume)
        print(
            json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2),
            flush=True,
        )
        return 0
    except KeyboardInterrupt:
        print(
            "Training stopped. Resume from last.pt after the last completed epoch; "
            "an unfinished epoch is not saved.",
            flush=True,
        )
        return 130
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
