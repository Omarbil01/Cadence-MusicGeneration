"""Job-specific output locations and atomic JSON writes."""

import json
import os
import re
from pathlib import Path

JOB_ID = re.compile(r"^[0-9a-f]{32}$")


class OutputStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def directory(self, job_id: str) -> Path:
        if not JOB_ID.fullmatch(job_id):
            raise ValueError("Invalid job ID")
        candidate = self.root / job_id
        if candidate.is_symlink():
            raise ValueError("Output job directory cannot be a symlink")
        path = candidate.resolve()
        if path.parent != self.root:
            raise ValueError("Output directory escapes the output root")
        return path

    def path(self, job_id: str, name: str = "output.mid") -> Path:
        if name not in {"output.mid", "metadata.json", "output.partial.mid"}:
            raise ValueError("Invalid output filename")
        directory = self.directory(job_id)
        path = (directory / name).resolve()
        if path.parent != directory:
            raise ValueError("Output file escapes its job directory")
        return path

    def prepare(self, job_id: str) -> Path:
        directory = self.directory(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def write_metadata(self, job_id: str, metadata: dict):
        path = self.path(job_id, "metadata.json")
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
