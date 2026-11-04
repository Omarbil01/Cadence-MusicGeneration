"""Melodic transposition preserves rhythm, drums, target alignment and source data."""

import copy
import random

import pytest

from musicgen.generation.runtime.tokenizer import BAR, BOS, EOS, PAD, TO_ID, TOKENS
from musicgen.training.augmentation import augment_window, transpose_window
from musicgen.training.preprocessing import make_windows


def _window(pitch=60):
    ids = [BOS, TO_ID["BPM_120"], TO_ID["METER_4_4"], BAR]
    for program, note_pitch in [(0, pitch), (128, 36), (32, 48)]:
        ids += [
            TO_ID[token]
            for token in (
                "POS_0",
                f"PROGRAM_{program}",
                f"PITCH_{note_pitch}",
                "DUR_4",
                "VEL_10",
            )
        ]
    return {
        **make_windows(ids + [EOS], 32)[0],
        "genre_id": 0,
        "composition_group": "work",
        "candidate_id": "one",
    }


def test_transposition_changes_melody_and_labels_without_changing_drums_or_masks():
    row = _window()
    original = copy.deepcopy(row)
    moved = transpose_window(row, 2)
    assert row == original
    assert len(moved["input_ids"]) == len(moved["labels"]) == 32
    assert TO_ID["PITCH_62"] in moved["input_ids"]
    assert TO_ID["PITCH_50"] in moved["input_ids"]
    assert TO_ID["PITCH_36"] in moved["input_ids"]
    for x, y in zip(moved["input_ids"][1:], moved["labels"][:-1]):
        if y not in {-100, EOS}:
            assert x == y
    assert [i for i, x in enumerate(row["input_ids"]) if x == PAD] == [
        i for i, x in enumerate(moved["input_ids"]) if x == PAD
    ]
    assert [i for i, y in enumerate(row["labels"]) if y == -100] == [
        i for i, y in enumerate(moved["labels"]) if y == -100
    ]
    for old, new in zip(row["input_ids"], moved["input_ids"]):
        if not TOKENS[old].startswith("PITCH_"):
            assert old == new


def test_augmentation_is_reproducible_epoch_local_and_keeps_midi_pitch_bounds():
    row = _window(0)
    state = random.getstate()
    shifts = set()
    for epoch in range(20):
        first = augment_window(row, radius=2, seed=42, epoch=epoch, index=0)
        second = augment_window(row, radius=2, seed=42, epoch=epoch, index=0)
        assert first == second
        pitch_id = first["input_ids"][6]
        shifts.add(int(TOKENS[pitch_id].split("_")[1]))
    assert random.getstate() == state
    assert shifts <= {0, 1, 2}
    assert len(shifts) > 1
    with pytest.raises(ValueError, match="outside MIDI range"):
        transpose_window(row, -1)


def test_validation_dataset_never_applies_training_augmentation():
    pytest.importorskip("torch")
    from musicgen.training.training import _Windows

    row = _window()
    validation = _Windows([row], transpose_semitones=2, epoch=None)
    assert validation[0][0].tolist() == row["input_ids"]
    training = _Windows([row], transpose_semitones=2, epoch=3, seed=42)
    expected = augment_window(row, radius=2, seed=42, epoch=3, index=0)
    assert training[0][0].tolist() == expected["input_ids"]
    assert training[0][1].tolist() == expected["labels"]
    assert row == _window()
