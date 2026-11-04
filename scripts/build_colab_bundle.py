"""Build the private source-data ZIP used by the Google Colab notebook."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musicgen.training.colab import build_bundle  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--midi-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("storage/training/cadence_colab_bundle.zip")
    )
    parser.add_argument("--min-per-genre", type=int, default=200)
    parser.add_argument("--max-per-genre", type=int, default=250)
    args = parser.parse_args()
    print(
        json.dumps(
            build_bundle(
                args.manifest,
                args.midi_root,
                args.output,
                min_per_genre=args.min_per_genre,
                max_per_genre=args.max_per_genre,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
