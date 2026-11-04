"""Check fixed validation coverage, AMP normalization, and resumable update state."""

# ruff: noqa: E402

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

import test_structured_training as structured_cases
import test_training as baseline_cases

from musicgen.generation.generator import MidiGenerator
from musicgen.shared.config import Settings
from musicgen.training import training
from musicgen.training.cache import SQLiteWindows

pytestmark = pytest.mark.checkpoint


@pytest.fixture
def midi_selection(tmp_path):
    return baseline_cases.midi_selection.__wrapped__(tmp_path)


@pytest.fixture
def prepared(midi_selection):
    return baseline_cases.prepared.__wrapped__(midi_selection)


@pytest.fixture
def tiny_config():
    return baseline_cases.tiny_config.__wrapped__()


@pytest.fixture
def structured(tmp_path):
    return structured_cases.structured.__wrapped__(tmp_path)


class CPUScaler:
    """Exercise the scaler lifecycle on CPU; this does not emulate FP16 arithmetic."""

    def __init__(self, *, skip_first=False):
        self.value, self.calls = 128.0, 0
        self.skip_first, self.skipped = skip_first, False

    def scale(self, loss):
        return loss * self.value

    def unscale_(self, optimizer):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    parameter.grad.div_(self.value)

    def get_scale(self):
        return self.value

    def step(self, optimizer):
        self.calls += 1
        self.skipped = self.skip_first and self.calls == 1
        if not self.skipped:
            optimizer.step()

    def update(self):
        if self.skipped:
            self.value /= 2

    def state_dict(self):
        return {"scale": self.value, "calls": self.calls, "skip_first": self.skip_first}

    def load_state_dict(self, state):
        self.value, self.calls = state["scale"], state["calls"]
        self.skip_first = state["skip_first"]


def test_fixed_validation_subset_covers_every_file_and_both_endpoints_without_payload_reads(
    structured, tiny_config, monkeypatch
):
    prepared, _ = structured
    windows = prepared["splits"]["val"]
    config = replace(tiny_config, val_windows_per_file=2)
    with monkeypatch.context() as local:

        def unexpected_decode(*args):
            raise AssertionError("Validation index selection must read only SQLite metadata")

        local.setattr(SQLiteWindows, "_decode", unexpected_decode)
        indices = training._validation_indices(windows, config)
        assert indices == training._validation_indices(windows, config)
        expected = {index for _, group in windows.file_indices() for index in (group[0], group[-1])}
        assert set(indices) == expected
    assert len({windows[index]["candidate_id"] for index in indices}) == 3
    assert len(indices) < len(windows)
    protocol = training._validation_protocol(windows, config, indices)
    assert protocol["selected_windows"] == len(indices)
    assert protocol["available_windows"] == len(windows)
    assert len(protocol["indices_sha256"]) == 64


def test_scaled_sum_loss_preserves_token_weighting_for_padded_partial_groups(tiny_config):
    torch.manual_seed(4)
    scaled = torch.nn.Linear(3, 2)
    reference = torch.nn.Linear(3, 2)
    reference.load_state_dict(scaled.state_dict())
    first = torch.optim.SGD(scaled.parameters(), lr=0.1)
    second = torch.optim.SGD(reference.parameters(), lr=0.1)
    inputs = torch.randn(5, 3)
    targets = torch.tensor([0, 1, -100, 1, 0])
    config = replace(tiny_config, batch_size=3, grad_accum=2)
    scaler, tokens = CPUScaler(), 0
    for begin, end in ((0, 3), (3, 5)):
        loss = torch.nn.functional.cross_entropy(
            scaled(inputs[begin:end]), targets[begin:end], ignore_index=-100, reduction="sum"
        )
        divisor = training._backward(loss, config, scaler)
        tokens += int(targets[begin:end].ne(-100).sum())
    assert training._finish_update(scaled, first, scaler, tokens, backward_divisor=divisor)
    loss = (
        torch.nn.functional.cross_entropy(
            reference(inputs), targets, ignore_index=-100, reduction="sum"
        )
        / tokens
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
    second.step()
    for actual, expected in zip(scaled.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


def test_skipped_updates_do_not_advance_schedule_and_scaler_state_resumes(
    prepared, tiny_config, tmp_path, monkeypatch
):
    monkeypatch.setattr(training, "_new_scaler", lambda config: CPUScaler(skip_first=True))
    config = replace(tiny_config, epochs=2, val_windows_per_file=2, transpose_semitones=2)
    complete = training.train(prepared, tmp_path / "complete", config)
    original_save = training._save

    def stop_second_epoch(path, state):
        if path.name == "last.pt" and state["epoch"] == 2:
            raise KeyboardInterrupt("Simulated interruption")
        original_save(path, state)

    monkeypatch.setattr(training, "_save", stop_second_epoch)
    directory = tmp_path / "resume"
    with pytest.raises(KeyboardInterrupt, match="Simulated"):
        training.train(prepared, directory, config)
    first_epoch = torch.load(directory / "last.pt", weights_only=True)
    assert first_epoch["optimizer_steps"] == 1
    assert first_epoch["skipped_updates"] == 1
    assert first_epoch["history"][0]["lr"] == config.lr
    assert first_epoch["scaler"]["calls"] == 2
    monkeypatch.setattr(training, "_save", original_save)
    resumed = training.train(prepared, directory, config, resume_from=directory / "last.pt")
    expected = torch.load(complete["last_checkpoint"], weights_only=True)
    actual = torch.load(resumed["last_checkpoint"], weights_only=True)
    assert actual["scaler"] == expected["scaler"]
    assert actual["optimizer_steps"] == expected["optimizer_steps"] == 3
    assert actual["skipped_updates"] == expected["skipped_updates"] == 1
    assert all(torch.equal(actual["model"][key], expected["model"][key]) for key in actual["model"])
    assert (
        actual["signature"]["validation_protocol"] == expected["signature"]["validation_protocol"]
    )
    assert actual["history"][-1]["val_loss"] == expected["history"][-1]["val_loss"]


def test_validation_cap_never_caps_or_augments_heldout_evaluation(
    prepared, tiny_config, tmp_path, monkeypatch
):
    config = replace(tiny_config, epochs=1, val_windows_per_file=2, transpose_semitones=2)
    original_evaluate = training._evaluate
    seen = []

    def inspect(model, windows, config, *, indices=None):
        if windows is prepared["splits"]["test"]:
            assert indices is None
            assert config.precision == "float32"
            seen.append("full_test")
        else:
            assert windows is prepared["splits"]["val"] and indices is not None
            seen.append("capped_validation")
        return original_evaluate(model, windows, config, indices=indices)

    monkeypatch.setattr(training, "_evaluate", inspect)
    run = training.train(prepared, tmp_path / "run", config)
    assert seen == ["capped_validation"]
    metrics = training.evaluate_test(run["best_checkpoint"], prepared)
    assert seen == ["capped_validation", "full_test"]
    assert metrics["precision"] == "float32"


def test_benchmark_reports_loader_inclusive_training_and_fixed_validation_costs(
    prepared, tiny_config
):
    config = replace(tiny_config, val_windows_per_file=2, transpose_semitones=2)
    result = training.benchmark(prepared, config, steps=3)
    assert result["optimizer_updates_measured"] > 0
    assert result["estimated_training_seconds_per_epoch"] > 0
    assert result["estimated_validation_seconds_per_epoch"] > 0
    assert result["estimated_epoch_seconds"] == pytest.approx(
        result["estimated_training_seconds_per_epoch"]
        + result["estimated_validation_seconds_per_epoch"]
    )
    assert result["validation_windows_evaluated"] < result["full_validation_windows"]
    assert result["validation_protocol"]["max_windows_per_file"] == 2


def test_legacy_config_records_omit_new_defaults_and_amp_requires_cuda(tiny_config):
    assert "precision" not in tiny_config.record()
    assert "val_windows_per_file" not in tiny_config.record()
    with pytest.raises(ValueError, match="requires device='cuda'"):
        training._new_model(replace(tiny_config, precision="amp_float16"))
    with pytest.raises(ValueError, match="Validation window cap"):
        replace(tiny_config, val_windows_per_file=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires an NVIDIA CUDA GPU")
@pytest.mark.parametrize("precision", ["amp_float16", "amp_bfloat16"])
def test_cuda_amp_scaler_resume_and_full_cpu_evaluation(prepared, tiny_config, tmp_path, precision):
    if precision == "amp_bfloat16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        pytest.skip("CUDA GPU lacks native bfloat16")
    config = replace(
        tiny_config,
        device="cuda",
        precision=precision,
        epochs=1,
        val_windows_per_file=2,
        transpose_semitones=2,
    )
    measured = training.benchmark(prepared, config, steps=2)
    assert measured["optimizer_updates_measured"] > 0
    smoke = training.smoke_train(prepared, tmp_path / "smoke", device="cuda", precision=precision)
    assert (
        torch.load(smoke["last_checkpoint"], map_location="cpu", weights_only=True)[
            "optimizer_steps"
        ]
        > 0
    )
    run = training.train(prepared, tmp_path / "run", config)
    last = torch.load(run["last_checkpoint"], map_location="cpu", weights_only=True)
    assert last["optimizer_steps"] > 0
    assert ("scaler" in last) == (precision == "amp_float16")
    repeated = training.train(
        prepared, tmp_path / "run", config, resume_from=run["last_checkpoint"]
    )
    assert repeated["history"] == run["history"]
    metrics = training.evaluate_test(run["best_checkpoint"], prepared, device="cpu")
    assert metrics["precision"] == "float32"
    exported = training.export_checkpoint(run["best_checkpoint"], tmp_path / "portable.pt")
    assert MidiGenerator(Settings(_env_file=None, checkpoint_path=exported, device="cpu")).load()
