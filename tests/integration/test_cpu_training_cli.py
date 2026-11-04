"""Exercise the standalone CPU runner with a complete, tiny balanced manifest."""

# ruff: noqa: E402

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from musicgen.generation.runtime.midi import Note, write_midi
from musicgen.generation.runtime.model import MusicTransformer
from musicgen.generation.runtime.tokenizer import GENRES
from musicgen.training.balanced import SELECTION_POLICY
from musicgen.training.manifest import digest

pytestmark = pytest.mark.checkpoint
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "train_balanced250_cpu.py"
PROFILE = [
    "--context",
    "16",
    "--d-model",
    "16",
    "--layers",
    "1",
    "--heads",
    "4",
    "--batch-size",
    "128",
    "--grad-accum",
    "1",
    "--max-windows-per-file",
    "1",
    "--val-windows-per-file",
    "2",
    "--epochs",
    "1",
    "--patience",
    "1",
    "--torch-threads",
    "1",
]


@pytest.fixture(scope="module")
def balanced_selection(tmp_path_factory):
    root = tmp_path_factory.mktemp("cpu-cli-balanced")
    rows = []
    for genre_id, genre in enumerate(GENRES):
        for index in range(250):
            split = "train" if index < 175 else "val" if index < 225 else "test"
            path = root / "mid" / f"{genre}_{index}.mid"
            path.parent.mkdir(exist_ok=True)
            program, pitch = genre_id * 24 + index // 36, 48 + index % 36
            write_midi(
                path, [Note(0, 1, pitch, program), Note(1, 2, pitch + 4, program)], end_beat=2
            )
            rows.append(
                {
                    "candidate_id": path.stem,
                    "sha256": digest(path),
                    "midi_path": path.relative_to(root).as_posix(),
                    "genre": genre,
                    "genre_id": str(genre_id),
                    "composition_group": path.stem,
                    "split": split,
                    "training_status": "APPROVED",
                    "approved_for_training": "true",
                    "selection_policy": SELECTION_POLICY,
                }
            )
    manifest = root / "selected.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert len({row["sha256"] for row in rows}) == 750
    return {"root": root, "manifest": manifest, "cache": root / "cache"}


def _run(selection, *options):
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(selection["manifest"]),
            "--midi-root",
            str(selection["root"]),
            "--cache",
            str(selection["cache"]),
            "--benchmark-steps",
            "1",
            *options,
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )


@pytest.mark.parametrize("action", ["--prepare-only", "--benchmark-only"])
def test_cpu_preflight_does_not_create_a_training_run(balanced_selection, tmp_path, action):
    output = tmp_path / "future_run"
    completed = _run(balanced_selection, *PROFILE, "--output", str(output), action)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not output.exists()
    preview = json.loads((tmp_path / "future_run.preflight.json").read_text())
    assert preview["device"] == "cpu" and preview["precision"] == "float32"
    assert preview["files_by_genre"] == dict.fromkeys(GENRES, 250)
    assert all(
        values == {"train": 175, "val": 50, "test": 25}
        for values in preview["files_by_genre_split"].values()
    )
    assert ("benchmark" in preview) == (action == "--benchmark-only")
    if action == "--benchmark-only":
        assert preview["benchmark"]["estimated_epoch_seconds"] > 0


def test_cpu_cli_trains_real_weights_and_resumes_only_matching_cpu_state(
    balanced_selection, tmp_path
):
    output = tmp_path / "run"
    completed = _run(balanced_selection, *PROFILE, "--output", str(output))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    last_path = output / "last.pt"
    state = torch.load(last_path, map_location="cpu", weights_only=True)
    assert state["epoch"] == 1 and state["optimizer_steps"] > 0
    saved_config = state["signature"]["training_config"]
    assert saved_config["device"] == "cpu" and saved_config.get("precision", "float32") == "float32"
    assert saved_config["progress_bins"] == 16 and saved_config["relative_attention"] is True
    assert "cuda_rng_state" not in state and "scaler" not in state
    assert all(
        tensor.device.type == "cpu" and torch.isfinite(tensor).all()
        for tensor in state["model"].values()
    )
    torch.manual_seed(saved_config["seed"])
    initial = MusicTransformer(**state["signature"]["model_config"]).state_dict()
    assert not torch.equal(initial["progress.weight"], state["model"]["progress.weight"])
    assert not torch.equal(initial["relative_bias.weight"], state["model"]["relative_bias.weight"])
    provenance = json.loads((output / "run.json").read_text())
    assert provenance["environment"]["gpu_name"] is None
    assert provenance["precision"] == "float32"
    assert state["dataset_metadata"]["source_audit"]["manifest_sha256"] == digest(
        balanced_selection["manifest"]
    )
    assert state["dataset_metadata"]["source_audit"]["unique_composition_groups"] == 750
    assert (output / "best.pt").is_file() and (output / "history.json").is_file()
    before = digest(last_path)
    resumed = _run(balanced_selection, "--resume", str(last_path))
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert digest(last_path) == before
    preview = json.loads((tmp_path / "run.preflight.json").read_text())
    assert preview["output"] == str(output)
    assert preview["configuration"]["context"] == 16
    assert preview["configuration"]["epochs"] == 1

    invalid_best = _run(balanced_selection, "--resume", str(output / "best.pt"))
    assert invalid_best.returncode == 2 and "Resume needs last.pt" in invalid_best.stderr
    changed_epochs = _run(balanced_selection, "--resume", str(last_path), "--epochs", "2")
    assert changed_epochs.returncode == 2 and "saved --epochs" in changed_epochs.stderr
    state["signature"]["training_config"]["device"] = "cuda"
    gpu_checkpoint = tmp_path / "gpu_last.pt"
    torch.save(state, gpu_checkpoint)
    wrong_device = _run(balanced_selection, "--resume", str(gpu_checkpoint))
    assert wrong_device.returncode == 2 and "CPU/float32" in wrong_device.stderr
    assert digest(last_path) == before
