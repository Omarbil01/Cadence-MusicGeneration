"""Select a usable, balanced 250-per-genre subset of the curated PDMX pool."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musicgen.training.balanced import build_balanced_manifest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curated-manifest", type=Path, required=True)
    parser.add_argument("--midi-root", type=Path, required=True)
    parser.add_argument("--preparation-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = build_balanced_manifest(
        args.curated_manifest,
        args.midi_root,
        args.output,
        preparation_report=args.preparation_report,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"file_reports", "rejected_candidates", "selected_strata_by_genre"}
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
