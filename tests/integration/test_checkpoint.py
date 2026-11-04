"""Actual PyTorch inference with a tiny untrained checkpoint fixture, never music evaluation."""

import pytest
from conftest import wait_for_job
from fastapi.testclient import TestClient

from musicgen.api.main import create_app
from musicgen.generation.generator import MidiGenerator, ModelUnavailable
from musicgen.generation.runtime.midi import read_midi
from musicgen.generation.runtime.tokenizer import GENRES, VOCAB_SHA

pytestmark = pytest.mark.checkpoint


@pytest.mark.parametrize("genre", GENRES)
def test_real_pytorch_request_to_midi(settings, checkpoint, genre):
    settings.checkpoint_path = checkpoint
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health/ready").json()["ready"]
        response = client.post("/api/generations", json={"genre": genre, "duration_seconds": 5})
        job_id = response.json()["id"]
        job = wait_for_job(client, job_id)
        assert job["status"] == "completed", job
        assert client.get(job["download_url"]).content.startswith(b"MThd")
        metadata = client.get(job["metadata_url"]).json()
        assert metadata["vocab_sha256"] == VOCAB_SHA
        assert metadata["context_mode"] == "framed"
        assert metadata["notes"] > 0
        notes, info = read_midi(settings.output_dir / job_id / "output.mid")
        assert not info["unmatched_off"] and not info["dangling_on"]
        assert all(21 <= n.pitch <= 108 for n in notes)
        assert metadata["actual_duration_seconds"] == pytest.approx(5, abs=0.1)


def test_generation_seed_repeats_and_cancellation_stops(settings, checkpoint, tmp_path):
    settings.checkpoint_path = checkpoint
    engine = MidiGenerator(settings)
    engine.load()
    for name in ("one.mid", "two.mid"):
        engine.generate(
            genre="rock_pop",
            duration_seconds=5,
            output_path=tmp_path / name,
            seed=42,
        )
    assert (tmp_path / "one.mid").read_bytes() == (tmp_path / "two.mid").read_bytes()
    with pytest.raises(RuntimeError, match="interrupted"):
        engine.generate(
            genre="rock_pop",
            duration_seconds=5,
            output_path=tmp_path / "cancel.mid",
            seed=1,
            cancelled=lambda: True,
        )
    assert not (tmp_path / "cancel.mid").exists()


def test_rejects_incompatible_checkpoint(settings, checkpoint):
    import torch

    record = torch.load(checkpoint, weights_only=True)
    record["signature"]["vocab_sha256"] = "changed"
    torch.save(record, checkpoint)
    settings.checkpoint_path = checkpoint
    with pytest.raises(ModelUnavailable) as error:
        MidiGenerator(settings).load()
    assert error.value.code == "checkpoint_incompatible"
