"""Deterministic training-only transposition of complete MIDI note events."""

import random

from musicgen.generation.runtime.tokenizer import TO_ID, TOKENS


def _pitch_positions(row):
    program = None
    inputs, labels = [], []
    for index, (token_id, label_id) in enumerate(zip(row["input_ids"], row["labels"])):
        token = TOKENS[token_id]
        if token.startswith("PROGRAM_"):
            program = int(token.split("_")[1])
        if program is not None and program != 128:
            if token.startswith("PITCH_"):
                inputs.append((index, int(token.split("_")[1])))
            if label_id >= 0 and TOKENS[label_id].startswith("PITCH_"):
                labels.append((index, int(TOKENS[label_id].split("_")[1])))
    return inputs, labels


def transpose_window(row, semitones):
    """Shift melodic pitches in inputs and next-token targets; preserve drums."""
    if isinstance(semitones, bool) or not isinstance(semitones, int):
        raise ValueError("Transposition must be an integer number of semitones")
    inputs, labels = _pitch_positions(row)
    if any(not 0 <= pitch + semitones <= 127 for _, pitch in inputs + labels):
        raise ValueError("Transposition would move a melodic pitch outside MIDI range")
    result = {**row, "input_ids": list(row["input_ids"]), "labels": list(row["labels"])}
    for key, positions in (("input_ids", inputs), ("labels", labels)):
        for index, pitch in positions:
            result[key][index] = TO_ID[f"PITCH_{pitch + semitones}"]
    return result


def augment_window(row, *, radius, seed, epoch, index):
    """Choose a stable legal shift without changing source caches or global RNG."""
    if not radius:
        return row
    if not 1 <= radius <= 5:
        raise ValueError("Transposition radius must be between 1 and 5")
    inputs, labels = _pitch_positions(row)
    pitches = [pitch for _, pitch in inputs + labels]
    shifts = [
        shift
        for shift in range(-radius, radius + 1)
        if all(0 <= pitch + shift <= 127 for pitch in pitches)
    ]
    rng = random.Random(seed + epoch * 1_000_003 + index * 97_409)
    return transpose_window(row, rng.choice(shifts))
