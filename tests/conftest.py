"""Test doubles are confined to tests; the app never substitutes a fake model."""

import time

import pytest

from musicgen.generation.runtime.midi import Note, write_midi
from musicgen.shared.config import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        checkpoint_path=tmp_path / "missing.pt",
        output_dir=tmp_path / "outputs",
        max_duration_seconds=240,
        worker_heartbeat_seconds=1,
        admin_db_path=tmp_path / "admin.sqlite3",
    )


class FixtureGenerator:
    def __init__(self, *, fail=False):
        self.loads = 0
        self.calls = 0
        self.fail = fail

    def load(self):
        self.loads += 1

    def generate(self, *, genre, duration_seconds, output_path, seed, cancelled=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("/private/model/path must never appear in a public error")
        beats = duration_seconds * 2
        write_midi(output_path, [Note(0, beats, 60, 0, 80)], end_beat=beats, bpm=120)
        return {
            "actual_duration_seconds": duration_seconds,
            "genre": genre,
            "seed": seed,
            "fixture_only": True,
        }


@pytest.fixture
def fixture_generator():
    return FixtureGenerator()


def wait_for_job(client, job_id, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get("/api/generations/" + job_id)
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(0.02)
    raise AssertionError("Job did not reach a terminal state")


@pytest.fixture
def checkpoint(tmp_path):
    from musicgen.generation.runtime.tokenizer import GENRES, VOCAB_SHA

    torch = pytest.importorskip("torch")
    from musicgen.generation.runtime.model import MusicTransformer

    torch.set_num_threads(1)
    torch.manual_seed(1)
    model = MusicTransformer(context=32, d_model=32, layers=1, heads=4, dropout=0)
    data_sha = "a" * 64
    record = {
        "model": model.state_dict(),
        "epoch": 1,
        "signature": {
            "vocab_sha256": VOCAB_SHA,
            "dataset_sha256": data_sha,
            "model_config": model.config,
            "training_config": {"smoke": False},
        },
        "dataset_metadata": {
            "dataset_kind": "EXP9_APPROVED_MIDI",
            "dataset_sha256": data_sha,
            "program_counts_train": {genre: {"0": 10} for genre in GENRES},
            "drum_pitch_counts_train": {genre: {} for genre in GENRES},
            "meter_counts_train": {genre: {"4/4": 10} for genre in GENRES},
        },
    }
    path = tmp_path / "untrained_test_fixture.pt"
    torch.save(record, path)
    return path
