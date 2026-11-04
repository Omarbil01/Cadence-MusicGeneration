"""New conditioning changes predictions while causal attention excludes future events."""

import pytest

torch = pytest.importorskip("torch")


def test_structured_model_is_causal_and_learns_progress_and_relative_distances():
    from musicgen.generation.runtime.model import MusicTransformer
    from musicgen.generation.runtime.tokenizer import BOS, TO_ID

    torch.set_num_threads(1)
    torch.manual_seed(23)
    model = MusicTransformer(
        context=32,
        d_model=16,
        layers=2,
        heads=4,
        dropout=0,
        progress_bins=16,
        relative_attention=True,
    )
    x = torch.full((2, 16), TO_ID["PITCH_60"], dtype=torch.long)
    x[:, 0] = BOS
    genres = torch.tensor([0, 1])
    progress = torch.zeros_like(x)
    output = model(x, genres, progress)
    output.square().mean().backward()
    assert model.progress.weight.grad.abs().sum() > 0
    assert model.relative_bias.weight.grad.abs().sum() > 0
    assert torch.isfinite(output).all()
    with torch.inference_mode():
        model.eval()
        before = model(x, genres, progress)
        changed = x.clone()
        changed[:, 8:] = TO_ID["PITCH_72"]
        future_progress = progress.clone()
        future_progress[:, 8:] = 15
        after = model(changed, genres, future_progress)
        torch.testing.assert_close(before[:, :8], after[:, :8], atol=1e-6, rtol=1e-6)
        assert not torch.allclose(before, model(x, genres, progress + 1))
    with pytest.raises(ValueError, match="Progress-conditioned"):
        model(x, genres)


def test_default_model_retains_legacy_checkpoint_config_and_state_names():
    from musicgen.generation.runtime.model import MusicTransformer

    model = MusicTransformer(context=32, d_model=16, layers=1, heads=4, dropout=0)
    assert set(model.config) == {"context", "d_model", "layers", "heads", "dropout"}
    assert not any("relative_bias" in key or "progress" in key for key in model.state_dict())
