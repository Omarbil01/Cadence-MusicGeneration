"""Verify structured conditioning from real MIDI bytes through inference export."""

# Skip before importing model/training modules in the CI job without PyTorch.
# ruff: noqa: E402

import csv
import struct

import pytest

torch = pytest.importorskip("torch")

from musicgen.generation.generator import MidiGenerator, ModelUnavailable
from musicgen.generation.runtime.midi import Note, read_midi, write_midi
from musicgen.generation.runtime.model import MusicTransformer
from musicgen.generation.runtime.tokenizer import EOS, GENRES, TO_ID
from musicgen.shared.config import Settings
from musicgen.training import preprocessing, training
from musicgen.training.cache import SQLiteWindows
from musicgen.training.manifest import digest

pytestmark = pytest.mark.checkpoint


def _vlq(value):
    result = [value & 127]
    value >>= 7
    while value:
        result.append(128 | (value & 127))
        value >>= 7
    return bytes(reversed(result))


def _replace_conductor(path):
    data = path.read_bytes()
    original_length = struct.unpack(">I", data[18:22])[0]
    events = [
        (0, b"\xff\x51\x03\x07\xa1\x20"),
        (0, b"\xff\x58\x04\x04\x02\x18\x08"),
        (960, b"\xff\x51\x03\x0a\x2c\x2b"),
        (1920, b"\xff\x58\x04\x03\x02\x18\x08"),
        (2880, b"\xff\x51\x03\x06\x1a\x80"),
    ]
    previous, payload = 0, bytearray()
    for tick, event in events:
        payload += _vlq(tick - previous) + event
        previous = tick
    payload += b"\x00\xff\x2f\x00"
    conductor = b"MTrk" + struct.pack(">I", len(payload)) + payload
    path.write_bytes(data[:14] + conductor + data[22 + original_length :])


@pytest.fixture
def structured(tmp_path):
    rows = []
    for genre_id, genre in enumerate(GENRES):
        for split_id, split in enumerate(("train", "val", "test")):
            path = tmp_path / "mid" / f"{genre}_{split}.mid"
            path.parent.mkdir(exist_ok=True)
            notes = [
                Note(
                    index / 2, (index + 1) / 2, 60 + genre_id + index % 5, genre_id + split_id * 16
                )
                for index in range(24)
            ]
            write_midi(path, notes, bpm=120, end_beat=12)
            _replace_conductor(path)
            rows.append(
                {
                    "candidate_id": path.stem,
                    "sha256": digest(path),
                    "midi_path": path.relative_to(tmp_path).as_posix(),
                    "genre": genre,
                    "genre_id": str(genre_id),
                    "composition_group": path.stem,
                    "split": split,
                    "training_status": "APPROVED",
                    "approved_for_training": "true",
                    "rules_version": "structured-test-selection-v1",
                    "authorization_basis": "user_authorized_expanded_training_selection",
                }
            )
    manifest = tmp_path / "selected.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    prepared = training.prepare_dataset(
        manifest,
        tmp_path,
        tmp_path / "cache",
        context=32,
        storage="sqlite",
        normalize_tempo=True,
        progress_bins=16,
        min_per_genre=3,
        max_per_genre=3,
    )
    return prepared, tmp_path / rows[0]["midi_path"]


@pytest.fixture
def structured_config():
    return training.TrainingConfig(
        context=32,
        d_model=16,
        layers=1,
        heads=4,
        dropout=0.15,
        progress_bins=16,
        relative_attention=True,
        epochs=1,
        batch_size=2,
        grad_accum=2,
        max_windows_per_file=2,
        torch_threads=1,
        seed=19,
        lr=0.003,
        patience=5,
    )


def test_normalized_sqlite_windows_supervise_only_true_piece_end(structured):
    prepared, source = structured
    assert prepared["metadata"]["dataset_kind"] == "PDMX_CURATED_MIDI"
    original = preprocessing.encode_midi(source)
    normalized = preprocessing.encode_midi(source, normalize_tempo=True, progress_bins=16)
    assert len(original["sequences"]) == 4
    assert len(normalized["sequences"]) == 2
    assert [sequence["is_piece_end"] for sequence in normalized["sequences"]] == [False, True]
    assert len({sequence["bpm"] for sequence in normalized["sequences"]}) == 1
    for split, windows in prepared["splits"].items():
        assert isinstance(windows, SQLiteWindows)
        for genre in GENRES:
            rows = [row for row in windows if row["candidate_id"] == f"{genre}_{split}"]
            assert sum(row["labels"].count(EOS) for row in rows) == 1
            assert all(len(row["progress_ids"]) == 32 for row in rows)
            assert max(value for row in rows for value in row["progress_ids"]) == 15


def test_relative_attention_and_progress_preserve_causality_and_affect_logits(
    structured, structured_config
):
    prepared, _ = structured
    row = prepared["splits"]["train"][0]
    torch.manual_seed(structured_config.seed)
    model = MusicTransformer(**{**structured_config.model_config(), "dropout": 0}).eval()
    x = torch.tensor([row["input_ids"]])
    features = torch.tensor([row["progress_ids"]])
    genres = torch.tensor([row["genre_id"]])
    cut = 12
    with torch.inference_mode():
        baseline = model(x, genres, features)
        future_x, future_features = x.clone(), features.clone()
        future_x[:, cut:] = TO_ID["PITCH_75"]
        future_features[:, cut:] = 15
        changed_future = model(future_x, genres, future_features)
        torch.testing.assert_close(baseline[:, :cut], changed_future[:, :cut], rtol=0, atol=1e-6)
        different_progress = features.clone()
        different_progress[:, 0] = (int(features[0, 0]) + 1) % 16
        assert not torch.allclose(baseline[:, 0], model(x, genres, different_progress)[:, 0])
        model.relative_bias.weight[1, 0] = 3
        learned_eval = model(x, genres, features)
        assert torch.isfinite(learned_eval).all()
        assert not torch.allclose(baseline[:, 1:cut], learned_eval[:, 1:cut])
        # Dropout is disabled: eval must preserve the learned float bias semantics.
        model.train()
        learned_training = model(x, genres, features)
        torch.testing.assert_close(learned_eval, learned_training, rtol=1e-5, atol=1e-6)


def test_structured_cpu_pipeline_exports_and_generates_without_reading_holdout(
    structured, structured_config, tmp_path, monkeypatch
):
    prepared, _ = structured
    original_decode = SQLiteWindows._decode

    def no_holdout_decode(self, row):
        if self.split == "test":
            raise AssertionError("Training, benchmark and smoke must not read held-out payloads")
        return original_decode(self, row)

    monkeypatch.setattr(SQLiteWindows, "_decode", no_holdout_decode)
    measured = training.benchmark(prepared, structured_config, steps=1)
    assert measured["measured_batches"] == 1 and measured["seconds_per_batch"] > 0
    smoke = training.smoke_train(prepared, tmp_path / "smoke")
    smoke_checkpoint = torch.load(smoke["best_checkpoint"], weights_only=True)
    assert smoke_checkpoint["signature"]["model_config"]["progress_bins"] == 16
    assert smoke_checkpoint["signature"]["model_config"]["relative_attention"] is True
    with pytest.raises(ModelUnavailable, match="Smoke-test"):
        training.export_checkpoint(smoke["best_checkpoint"], tmp_path / "smoke_export.pt")
    torch.manual_seed(structured_config.seed)
    initial = MusicTransformer(**structured_config.model_config()).state_dict()
    result = training.train(prepared, tmp_path / "run", structured_config)
    last = torch.load(result["last_checkpoint"], weights_only=True)
    for parameter in ("relative_bias.weight", "progress.weight"):
        assert not torch.equal(last["model"][parameter], initial[parameter])
    exported = training.export_checkpoint(result["best_checkpoint"], tmp_path / "export.pt")
    checkpoint = torch.load(exported, weights_only=True)
    assert set(checkpoint) == {"model", "epoch", "signature", "dataset_metadata"}
    engine = MidiGenerator(
        Settings(
            _env_file=None,
            checkpoint_path=exported,
            device="cpu",
            torch_threads=1,
            max_tokens=5000,
            generation_timeout_seconds=30,
        )
    )
    info = engine.load()
    assert info["progress_bins"] == 16 and engine.model.relative_attention is True
    for genre in GENRES:
        output = tmp_path / f"{genre}.mid"
        metadata = engine.generate(genre=genre, duration_seconds=5, output_path=output, seed=42)
        notes, integrity = read_midi(output)
        assert notes and not integrity["unmatched_off"] and not integrity["dangling_on"]
        assert metadata["actual_duration_seconds"] == pytest.approx(5, abs=0.15)
        assert metadata["progress_bins"] == 16


def test_structured_sqlite_resume_restores_relative_progress_and_optimizer_state(
    structured, structured_config, tmp_path, monkeypatch
):
    from dataclasses import replace

    prepared, _ = structured
    config = replace(structured_config, epochs=2)
    complete = training.train(prepared, tmp_path / "complete", config)
    original_save = training._save

    def stop_at_epoch_two(path, record):
        if path.name == "last.pt" and record["epoch"] == 2:
            raise KeyboardInterrupt("Epoch boundary interruption")
        return original_save(path, record)

    monkeypatch.setattr(training, "_save", stop_at_epoch_two)
    run = tmp_path / "resume"
    with pytest.raises(KeyboardInterrupt, match="Epoch boundary"):
        training.train(prepared, run, config)
    assert torch.load(run / "last.pt", weights_only=True)["epoch"] == 1
    monkeypatch.setattr(training, "_save", original_save)
    resumed = training.train(prepared, run, config, resume_from=run / "last.pt")
    expected = torch.load(complete["last_checkpoint"], weights_only=True)
    actual = torch.load(resumed["last_checkpoint"], weights_only=True)
    assert expected["optimizer_steps"] == actual["optimizer_steps"]
    assert all(
        torch.equal(expected["model"][key], actual["model"][key]) for key in expected["model"]
    )
    assert expected["history"][-1]["val_loss"] == actual["history"][-1]["val_loss"]
