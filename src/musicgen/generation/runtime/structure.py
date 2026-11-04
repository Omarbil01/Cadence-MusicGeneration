"""Shared duration conditioning for preprocessing and autoregressive generation."""

from .context import framed_context


def progress_bin(at: float, total: float, bins: int) -> int:
    """Map elapsed musical time to a bounded whole-piece progress bucket."""
    if bins < 2 or total <= 0:
        raise ValueError("Progress needs at least two bins and positive duration")
    return min(bins - 1, max(0, int(max(0.0, at) / total * bins)))


def prompt_features(ids, features, context):
    """Frame tokens and their progress together, resetting the synthetic header."""
    if len(ids) != len(features):
        raise ValueError("Token and progress lengths differ")
    prompt, positions = framed_context(ids, context, return_positions=True)
    first = next(features[position] for position in positions if position is not None)
    return prompt, [first if position is None else features[position] for position in positions]
