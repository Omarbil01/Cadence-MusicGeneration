"""Train the balanced MIDI Transformer on a Linux CPU; see --help for options."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from musicgen.training.cpu import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
