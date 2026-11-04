"""Build the expanded local PDMX selection; never starts model training."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musicgen.training.pdmx import build_manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdmx-csv", type=Path, required=True)
    parser.add_argument("--midi-root", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--classical-scope", choices=("baroque_classical", "classical"), default="baroque_classical"
    )
    parser.add_argument(
        "--beethoven-policy", choices=("review", "classical", "romantic"), default="review"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = build_manifest(
        args.pdmx_csv,
        args.midi_root,
        args.output,
        baseline_manifest=args.baseline_manifest,
        classical_scope=args.classical_scope,
        beethoven_policy=args.beethoven_policy,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"composer_rules", "quarantined_families"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
