"""Standalone inference CLI using the same engine as the worker."""

import argparse
import json
import secrets
from pathlib import Path

from musicgen.shared.config import Settings

from .generator import MidiGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate a MIDI with the Experiment 9 checkpoint."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--genre", choices=["classical", "romantic", "rock_pop"], required=True)
    parser.add_argument("--seconds", type=int, default=45)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--palette", choices=["original", "band"], default="original")
    args = parser.parse_args()
    if not 5 <= args.seconds <= 240:
        parser.error("--seconds must be between 5 and 240")
    if args.output.suffix.lower() != ".mid":
        parser.error("--output must have a .mid extension")
    metadata_path = args.output.with_suffix(".json")
    if args.output.exists() or metadata_path.exists():
        parser.error("Choose a new output filename")
    settings = Settings(checkpoint_path=args.checkpoint, device=args.device, palette=args.palette)
    engine = MidiGenerator(settings)
    engine.load()
    metadata = engine.generate(
        genre=args.genre,
        duration_seconds=args.seconds,
        output_path=args.output,
        seed=args.seed if args.seed is not None else secrets.randbelow(2**31),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"midi": str(args.output), "metadata": str(metadata_path)}))


if __name__ == "__main__":
    main()
