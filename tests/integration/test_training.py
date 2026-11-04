"""Exercise real MIDI preparation, CPU optimization, resume, and inference export."""

# Skip before importing the training modules in the CI job without PyTorch.
# ruff: noqa: E402

import copy
import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from musicgen.generation.generator import MidiGenerator, ModelUnavailable
from musicgen.generation.runtime.midi import Note, write_midi
from musicgen.generation.runtime.model import MusicTransformer
from musicgen.generation.runtime.tokenizer import GENRES, VOCAB_SHA
from musicgen.shared.config import Settings
from musicgen.training import training
from musicgen.training.manifest import digest

pytestmark = pytest.mark.checkpoint


def _write_manifest(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def midi_selection(tmp_path):
    rows = []
    for genre_id, genre in enumerate(GENRES):
        for split_id, split in enumerate(("train", "val", "test")):
            path = tmp_path / "mid" / f"{genre}_{split}.mid"
            path.parent.mkdir(exist_ok=True)
            program = genre_id + 16 * split_id
            notes = [Note(i / 2, (i + 1) / 2, 60 + genre_id + i % 4, program) for i in range(12)]
            write_midi(path, notes, meter=((4, 4), (3, 4), (6, 8))[split_id], end_beat=6)
            rows.append(
                {
                    "candidate_id": path.stem,
                    "sha256": digest(path),
                    "midi_path": str(path.relative_to(tmp_path)),
                    "genre": genre,
                    "genre_id": str(genre_id),
                    "composition_group": path.stem,
                    "split": split,
                    "training_status": "APPROVED",
                    "approved_for_training": "true",
                    "genre_status": "PROVISIONAL",
                    "rights_status": "UNVERIFIED",
                }
            )
    manifest = tmp_path / "approved.csv"
    _write_manifest(manifest, rows)
    return {"root": tmp_path, "manifest": manifest, "rows": rows, "cache": tmp_path / "cache"}


def _prepare(selection, *, context=32):
    return training.prepare_dataset(
        selection["manifest"],
        selection["root"],
        selection["cache"],
        context=context,
        min_per_genre=3,
        max_per_genre=3,
    )


@pytest.fixture
def prepared(midi_selection):
    return _prepare(midi_selection)


@pytest.fixture
def tiny_config():
    return training.TrainingConfig(
        context=32,
        d_model=16,
        layers=1,
        heads=4,
        dropout=0.2,
        batch_size=2,
        grad_accum=2,
        epochs=2,
        lr=0.005,
        patience=5,
        torch_threads=1,
        max_windows_per_file=2,
        seed=17,
    )


def test_prepare_preserves_split_isolation_and_training_only_statistics(
    midi_selection, prepared, monkeypatch
):
    assert prepared["report"]["selected_files"] == prepared["report"]["retained_files"] == 9
    assert prepared["report"]["rejected_files"] == []
    assert prepared["metadata"]["dataset_sha256"] == prepared["dataset_sha256"]
    for genre_id, genre in enumerate(GENRES):
        assert prepared["metadata"]["program_counts_train"][genre] == {str(genre_id): 12}
        assert prepared["metadata"]["meter_counts_train"][genre] == {"4/4": 1}
        assert prepared["metadata"]["drum_pitch_counts_train"][genre] == {}
    split_groups = []
    for split, windows in prepared["splits"].items():
        assert windows
        expected = {row["candidate_id"] for row in midi_selection["rows"] if row["split"] == split}
        assert {window["candidate_id"] for window in windows} == expected
        split_groups.append({window["composition_group"] for window in windows})
        assert all(len(window["input_ids"]) == len(window["labels"]) == 32 for window in windows)
    assert all(
        not left & right for i, left in enumerate(split_groups) for right in split_groups[i + 1 :]
    )

    def unexpected_encoding(*args, **kwargs):
        raise AssertionError("Unchanged prepared data should reuse its verified cache")

    monkeypatch.setattr(training.preprocessing, "encode_midi", unexpected_encoding)
    assert _prepare(midi_selection) == prepared
    source = midi_selection["root"] / midi_selection["rows"][0]["midi_path"]
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Source hash mismatch"):
        _prepare(midi_selection)


def test_prepare_rejects_tampered_cached_windows(midi_selection, prepared):
    cache_files = [
        path
        for path in midi_selection["cache"].glob("*.json")
        if path.name != "preparation_report.json"
    ]
    assert len(cache_files) == 1
    cache_path = cache_files[0]
    cached = json.loads(cache_path.read_text())
    cached["splits"]["train"][0]["input_ids"][0] += 1
    cache_path.write_text(json.dumps(cached))
    with pytest.raises(ValueError, match="[Cc]ache"):
        _prepare(midi_selection)


def test_prepare_reports_rejected_file_before_retained_count_failure(midi_selection):
    rejected = midi_selection["rows"][0]
    source = midi_selection["root"] / rejected["midi_path"]
    write_midi(source, [], end_beat=4)
    rejected["sha256"] = digest(source)
    _write_manifest(midi_selection["manifest"], midi_selection["rows"])
    with pytest.raises(ValueError, match="retained 2 files"):
        _prepare(midi_selection)
    report = json.loads((midi_selection["cache"] / "preparation_report.json").read_text())
    assert report["retained_files"] == 8
    assert report["rejected_files"][0]["candidate_id"] == rejected["candidate_id"]
    assert "no matched" in report["rejected_files"][0]["error"].lower()
    assert not list(midi_selection["cache"].glob("[0-9a-f]" * 64 + ".json"))


def test_cpu_training_selects_on_validation_and_exports_strict_app_checkpoint(
    prepared, tiny_config, tmp_path, monkeypatch
):
    torch.manual_seed(tiny_config.seed)
    initial = MusicTransformer(**tiny_config.model_config()).state_dict()
    original_evaluate = training._evaluate
    evaluated_splits = []

    def evaluate_validation(model, windows, config):
        assert windows is prepared["splits"]["val"]
        evaluated_splits.append("val")
        return original_evaluate(model, windows, config)

    monkeypatch.setattr(training, "_evaluate", evaluate_validation)
    result = training.train(prepared, tmp_path / "run", tiny_config)
    assert result["epochs_completed"] == 2
    assert evaluated_splits == ["val", "val"]
    last = torch.load(result["last_checkpoint"], weights_only=True)
    assert last["optimizer_steps"] > 0
    assert any(not torch.equal(initial[key], last["model"][key]) for key in initial)
    assert all(torch.isfinite(torch.tensor(epoch["val_loss"])) for epoch in result["history"])

    exported = training.export_checkpoint(
        result["best_checkpoint"], tmp_path / "export" / "best.pt"
    )
    checkpoint = torch.load(exported, weights_only=True)
    assert set(checkpoint) == {"model", "epoch", "signature", "dataset_metadata"}
    assert checkpoint["signature"]["vocab_sha256"] == VOCAB_SHA
    assert checkpoint["signature"]["dataset_sha256"] == prepared["dataset_sha256"]
    engine = MidiGenerator(Settings(_env_file=None, checkpoint_path=exported, device="cpu"))
    assert engine.load()["model_context"] == tiny_config.context
    assert set(engine.model.state_dict()) == set(checkpoint["model"])
    with pytest.raises(FileExistsError, match="Destination already exists"):
        training.export_checkpoint(result["best_checkpoint"], exported)

    monkeypatch.setattr(training, "_evaluate", original_evaluate)
    metrics = training.evaluate_test(result["best_checkpoint"], prepared)
    assert set(metrics["per_genre"]) == set(GENRES)
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    metric_path = Path(result["best_checkpoint"]).parent / "test_metrics.json"
    metric_bytes = metric_path.read_bytes()

    def unexpected_test_repeat(*args, **kwargs):
        raise AssertionError("A saved held-out evaluation must be reused")

    monkeypatch.setattr(training, "_evaluate", unexpected_test_repeat)
    assert training.evaluate_test(result["best_checkpoint"], prepared) == metrics
    assert metric_path.read_bytes() == metric_bytes
    with pytest.raises(ValueError, match="another selected checkpoint"):
        training.evaluate_test(result["best_checkpoint"], {**prepared, "dataset_sha256": "f" * 64})


def test_smoke_checkpoint_cannot_be_exported_or_evaluated(prepared, tmp_path):
    result = training.smoke_train(prepared, tmp_path / "smoke")
    with pytest.raises(ModelUnavailable, match="Smoke-test"):
        training.export_checkpoint(result["best_checkpoint"], tmp_path / "export.pt")
    with pytest.raises(ValueError, match="Smoke runs"):
        training.evaluate_test(result["best_checkpoint"], prepared)
    assert not (tmp_path / "export.pt").exists()


def test_epoch_boundary_resume_matches_uninterrupted_cpu_training(
    prepared, tiny_config, tmp_path, monkeypatch
):
    uninterrupted = training.train(prepared, tmp_path / "uninterrupted", tiny_config)
    original_save = training._save

    def interrupt_before_second_last_checkpoint(path, payload):
        if path.name == "last.pt" and payload["epoch"] == 2:
            raise KeyboardInterrupt("Simulated stop before epoch two is committed")
        return original_save(path, payload)

    monkeypatch.setattr(training, "_save", interrupt_before_second_last_checkpoint)
    run_dir = tmp_path / "resumed"
    with pytest.raises(KeyboardInterrupt, match="Simulated stop"):
        training.train(prepared, run_dir, tiny_config)
    interrupted = torch.load(run_dir / "last.pt", weights_only=True)
    assert interrupted["epoch"] == 1
    monkeypatch.setattr(training, "_save", original_save)
    resumed = training.train(prepared, run_dir, tiny_config, resume_from=run_dir / "last.pt")
    expected = torch.load(uninterrupted["last_checkpoint"], weights_only=True)
    actual = torch.load(resumed["last_checkpoint"], weights_only=True)
    assert expected["epoch"] == actual["epoch"] == 2
    assert expected["optimizer_steps"] == actual["optimizer_steps"]
    assert all(
        torch.equal(expected["model"][key], actual["model"][key]) for key in expected["model"]
    )
    assert expected["history"][-1]["train_loss"] == actual["history"][-1]["train_loss"]
    assert expected["history"][-1]["val_loss"] == actual["history"][-1]["val_loss"]


def test_training_guards_context_and_resume_dataset_identity(prepared, tiny_config, tmp_path):
    with pytest.raises(ValueError, match="context differ"):
        training.train(prepared, tmp_path / "wrong_context", replace(tiny_config, context=64))
    run_dir = tmp_path / "run"
    result = training.train(prepared, run_dir, tiny_config)
    changed = copy.deepcopy(prepared)
    changed["dataset_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="same dataset, model and training configuration"):
        training.train(changed, run_dir, tiny_config, resume_from=result["last_checkpoint"])


def test_gradient_accumulation_matches_large_batches_including_partial_final_group(
    prepared, tiny_config, tmp_path, monkeypatch
):
    small_batches = replace(tiny_config, dropout=0, epochs=1, batch_size=2, grad_accum=2)
    large_batches = replace(small_batches, batch_size=4, grad_accum=1)
    # Six selected windows give one full four-sample update and one two-sample tail.
    assert len(training._epoch_indices(prepared["splits"]["train"], small_batches, 0)) == 6
    updates = []
    original_step = torch.optim.AdamW.step

    def record_gradients(optimizer, *args, **kwargs):
        updates.append(
            [
                parameter.grad.detach().clone()
                for group in optimizer.param_groups
                for parameter in group["params"]
                if parameter.grad is not None
            ]
        )
        return original_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", record_gradients)
    small = training.train(prepared, tmp_path / "small", small_batches)
    large = training.train(prepared, tmp_path / "large", large_batches)
    small_state = torch.load(small["last_checkpoint"], weights_only=True)
    large_state = torch.load(large["last_checkpoint"], weights_only=True)
    assert small_state["optimizer_steps"] == large_state["optimizer_steps"] == 2
    # Compare normalized gradients directly: Adam can amplify roundoff in the
    # mathematically-zero attention-key bias gradient, even with equivalent batches.
    assert len(updates) == 4
    assert all(
        torch.allclose(left, right, atol=1e-6, rtol=1e-4)
        for micro, full in zip(updates[:2], updates[2:])
        for left, right in zip(micro, full)
    )
    assert small["history"][0]["train_loss"] == pytest.approx(
        large["history"][0]["train_loss"], abs=1e-6
    )


def test_failure_saving_selected_model_does_not_advance_resume_checkpoint(
    prepared, tiny_config, tmp_path, monkeypatch
):
    config = replace(tiny_config, min_delta=0)
    original_save = training._save
    original_evaluate = training._evaluate

    def improving_validation(model, windows, cfg):
        values = original_evaluate(model, windows, cfg)
        # Force selection on both epochs to exercise the write-order failure path.
        values["loss"] = 10.0 - len(list((tmp_path / "run").glob("last.pt")))
        return values

    def fail_second_best(path, payload):
        if path.name == "best.pt" and payload["epoch"] == 2:
            raise OSError("Simulated disk write failure")
        original_save(path, payload)

    monkeypatch.setattr(training, "_evaluate", improving_validation)
    monkeypatch.setattr(training, "_save", fail_second_best)
    with pytest.raises(OSError, match="Simulated disk"):
        training.train(prepared, tmp_path / "run", config)
    assert torch.load(tmp_path / "run/last.pt", weights_only=True)["epoch"] == 1
    assert torch.load(tmp_path / "run/best.pt", weights_only=True)["epoch"] == 1


def test_cpu_benchmark_times_even_a_single_available_batch(prepared, tiny_config):
    config = replace(tiny_config, batch_size=64)
    result = training.benchmark(prepared, config, steps=2)
    assert result["batches_per_epoch"] == 1
    assert result["measured_batches"] == 2
    assert result["seconds_per_batch"] > 0


def test_early_stopping_waits_fifty_epochs_and_resets_after_improvement(
    prepared, tiny_config, tmp_path, monkeypatch
):
    config = replace(tiny_config, epochs=500, patience=50, max_windows_per_file=1)
    evaluations = 0

    def validation_plateau_then_improvement(model, windows, cfg):
        nonlocal evaluations
        evaluations += 1
        loss = 2.0 if evaluations < 25 else 1.0
        return {"loss": loss, "per_genre": dict.fromkeys(GENRES, loss)}

    monkeypatch.setattr(training, "_evaluate", validation_plateau_then_improvement)
    result = training.train(prepared, tmp_path / "patience", config)
    assert result["stopped_early"]
    assert result["epochs_completed"] == evaluations == 75
    assert torch.load(result["best_checkpoint"], weights_only=True)["epoch"] == 25
    assert torch.load(result["last_checkpoint"], weights_only=True)["stale_epochs"] == 50


def test_training_cli_applies_explicit_epoch_and_patience_settings(prepared, tmp_path, monkeypatch):
    received = {}

    def capture_train(data, output, config, *, resume_from):
        received.update(epochs=config.epochs, patience=config.patience, output=output)
        return {"best_checkpoint": str(output / "best.pt")}

    monkeypatch.setattr(training, "prepare_dataset", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(training, "train", capture_train)
    output = tmp_path / "cli"
    training.main(
        [
            "--manifest",
            "manifest.csv",
            "--midi-root",
            "mid",
            "--output",
            str(output),
            "--epochs",
            "500",
            "--patience",
            "50",
        ]
    )
    assert received == {"epochs": 500, "patience": 50, "output": output}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires an NVIDIA CUDA GPU")
def test_cuda_training_resume_metrics_and_cpu_export(prepared, tiny_config, tmp_path):
    config = replace(tiny_config, device="cuda", dropout=0)
    measured = training.benchmark(prepared, config, steps=2)
    assert measured["measured_batches"] == 2 and measured["seconds_per_batch"] > 0
    smoke = training.smoke_train(prepared, tmp_path / "gpu_smoke", device="cuda")
    smoke_checkpoint = torch.load(smoke["last_checkpoint"], map_location="cpu", weights_only=True)
    assert smoke_checkpoint["signature"]["training_config"]["device"] == "cuda"
    assert smoke_checkpoint["cuda_rng_state"].device.type == "cpu"
    run = training.train(prepared, tmp_path / "gpu", config)
    resumed = training.train(prepared, tmp_path / "gpu", config, resume_from=run["last_checkpoint"])
    assert resumed["history"] == run["history"]
    metrics = training.evaluate_test(run["best_checkpoint"], prepared, device="cuda")
    assert metrics["device"] == "cuda" and torch.isfinite(torch.tensor(metrics["loss"]))
    exported = training.export_checkpoint(run["best_checkpoint"], tmp_path / "cpu_export.pt")
    assert MidiGenerator(Settings(_env_file=None, checkpoint_path=exported, device="cpu")).load()
