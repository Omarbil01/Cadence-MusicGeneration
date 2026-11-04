"""Encoding preserves supported MIDI semantics and real causal targets."""

import json
import struct

import pytest

from musicgen.generation.runtime.context import framed_context
from musicgen.generation.runtime.grammar import NoteGrammar
from musicgen.generation.runtime.midi import Note, read_midi, write_midi
from musicgen.generation.runtime.tokenizer import BAR, BOS, EOS, PAD, TO_ID, TOKENS
from musicgen.training.preprocessing import encode_midi, make_windows


def _midi(tmp_path, notes, **kwargs):
    path = tmp_path / "example.mid"
    write_midi(path, notes, **kwargs)
    return path


def _token_ids(notes=12):
    result = [BOS, TO_ID["BPM_120"], TO_ID["METER_4_4"], BAR]
    for index in range(notes):
        if index and index % 4 == 0:
            result.append(BAR)
        result.extend(
            TO_ID[token]
            for token in (
                f"POS_{index % 4 * 4}",
                "PROGRAM_0",
                f"PITCH_{60 + index % 8}",
                "DUR_4",
                "VEL_10",
            )
        )
    return result + [EOS]


def _vlq(value):
    result = [value & 127]
    value >>= 7
    while value:
        result.append(128 | (value & 127))
        value >>= 7
    return bytes(reversed(result))


def _replace_conductor(path, events):
    data = path.read_bytes()
    old_size = struct.unpack(">I", data[18:22])[0]
    payload = bytearray()
    previous = 0
    for tick, event in events:
        payload += _vlq(tick - previous) + event
        previous = tick
    payload += b"\x00\xff\x2f\x00"
    track = b"MTrk" + struct.pack(">I", len(payload)) + payload
    path.write_bytes(data[:14] + track + data[22 + old_size :])


def test_encoding_round_trip_preserves_quantized_notes_in_runtime(tmp_path):
    notes = [
        Note(0, 0.5, 60, 0, 80),
        Note(0.75, 1.5, 64, 0, 96),
        Note(1, 1.25, 36, 128, 64),
        Note(2, 3, 67, 0, 72),
    ]
    path = _midi(tmp_path, notes, bpm=120, meter=(4, 4))
    encoded = encode_midi(path)
    sequence = encoded["sequences"][0]
    grammar = NoteGrammar(
        target_beats=3,
        bpm=120,
        meter=(4, 4),
        programs=(0, 128),
        drum_pitches=(36,),
    )
    for token in sequence["ids"][4:]:
        grammar.accept(token)
    assert grammar.done
    assert [(n.start, n.end, n.pitch, n.program, n.velocity) for n in grammar.notes] == [
        (n.start, n.end, n.pitch, n.program, n.velocity) for n in notes
    ]
    roundtrip = tmp_path / "roundtrip.mid"
    write_midi(roundtrip, grammar.notes, end_beat=3)
    reread, info = read_midi(roundtrip)
    assert [(n.start, n.end, n.pitch, n.program) for n in reread] == [
        (n.start, n.end, n.pitch, n.program) for n in notes
    ]
    assert not info["unmatched_off"] and not info["dangling_on"]
    assert encoded["counts"] == {
        "program_counts": {"0": 3, "128": 1},
        "drum_pitch_counts": {"36": 1},
        "meter_counts": {"4/4": 1},
    }
    json.dumps(encoded)


@pytest.mark.parametrize("context", [16, 32, 64, 256])
def test_windows_supervise_every_real_target_once_without_artificial_eos(context):
    ids = _token_ids(31)
    windows = make_windows(ids, context)
    supervised = []
    next_target = 4
    for window in windows:
        inputs, labels = window["input_ids"], window["labels"]
        assert len(inputs) == len(labels) == context
        assert inputs[:4] == ids[:4]
        assert window["target_start"] == next_target
        new_targets = [label for label in labels if label != -100]
        assert new_targets == ids[next_target : window["source_end"]]
        supervised.extend(new_targets)
        next_target = window["source_end"]
        assert labels[:3] == [-100] * 3
        assert all(label == -100 for token, label in zip(inputs, labels) if token == PAD)
        source_start, source_end = window["source_start"], window["source_end"]
        prefix = ids[:3] + ([] if ids[source_start] == BAR else [BAR])
        framed = prefix + ids[source_start:source_end]
        assert [token for token in inputs if token != PAD] == framed[:-1]
        if len(framed) >= 16:
            assert framed_context(ids[:source_end], len(framed)) == framed
    assert supervised == ids[4:]
    assert supervised.count(EOS) == 1
    assert EOS not in [label for window in windows[:-1] for label in window["labels"]]


def test_padding_and_overlap_never_create_training_targets():
    ids = _token_ids(1)
    window = make_windows(ids, 32)[0]
    assert [label for label in window["labels"] if label != -100] == ids[4:]
    assert window["input_ids"].count(PAD) == 32 - (len(ids) - 1)
    assert window["labels"].count(-100) == 32 - len(ids[4:])
    longer_ids = _token_ids(10)
    longer = make_windows(longer_ids, 16)
    for window in longer[1:]:
        first_new_target = next(i for i, label in enumerate(window["labels"]) if label != -100)
        # BAR overlap needs one token; a preceding note needs all five fields.
        expected = 3 if longer_ids[window["source_start"]] == BAR else 8
        assert first_new_target == expected


@pytest.mark.parametrize(
    "transform",
    [
        lambda ids: ids[:-1],
        lambda ids: ids[:7] + ids[8:],
        lambda ids: ids[:5] + [EOS] + ids[5:],
        lambda ids: ids[:5] + [len(TOKENS)] + ids[6:],
        lambda ids: ids[:5] + [True] + ids[6:],
        lambda ids: ids[:4] + [TO_ID["POS_16"]] + ids[5:],
    ],
)
def test_malformed_or_partial_note_sequences_are_rejected(transform):
    with pytest.raises(ValueError):
        make_windows(transform(_token_ids(2)), 32)


def test_tempo_and_meter_changes_clip_tails_without_adding_attacks(tmp_path):
    path = _midi(
        tmp_path,
        [Note(0, 1, 60, 0), Note(3, 5, 62, 0), Note(4, 5, 64, 0), Note(7, 8, 65, 0)],
    )
    _replace_conductor(
        path,
        [
            (0, b"\xff\x51\x03" + (500000).to_bytes(3, "big")),
            (0, b"\xff\x58\x04\x04\x02\x18\x08"),
            (1920, b"\xff\x51\x03" + (666667).to_bytes(3, "big")),
            (1920, b"\xff\x58\x04\x03\x02\x18\x08"),
        ],
    )
    encoded = encode_midi(path)
    assert [seq["meter"] for seq in encoded["sequences"]] == [[4, 4], [3, 4]]
    assert [seq["bpm"] for seq in encoded["sequences"]] == [120, 90]
    assert [seq["start_beat"] for seq in encoded["sequences"]] == [0, 4]
    assert encoded["report"]["notes_clipped_at_metadata_change"] == 1
    assert encoded["report"]["retained_notes"] == 4
    assert [TOKENS[token] for token in encoded["sequences"][0]["ids"]][-3:] == [
        "DUR_4",
        "VEL_10",
        "EOS",
    ]
    assert encoded["counts"]["meter_counts"] == {"4/4": 1, "3/4": 1}


def test_tempo_only_change_keeps_existing_meter_bar_origin(tmp_path):
    path = _midi(tmp_path, [Note(0, 1, 60, 0), Note(2.5, 3, 64, 0)])
    _replace_conductor(
        path,
        [
            (0, b"\xff\x51\x03" + (500000).to_bytes(3, "big")),
            (960, b"\xff\x51\x03" + (400000).to_bytes(3, "big")),
        ],
    )
    sequences = encode_midi(path)["sequences"]
    assert len(sequences) == 2
    assert sequences[1]["start_beat"] == 0
    assert TOKENS[sequences[1]["ids"][4]] == "POS_10"
    assert sequences[1]["bpm"] == 150


def test_long_silence_splits_and_unsupported_meters_are_reported(tmp_path):
    path = _midi(tmp_path, [Note(0, 1, 60, 0), Note(10000, 10001, 64, 0)])
    encoded = encode_midi(path, max_silent_bars=2)
    assert len(encoded["sequences"]) == 2
    assert all(len(seq["ids"]) == 10 for seq in encoded["sequences"])
    assert encoded["report"]["long_silence_splits"] == 1
    # A long held note is clipped to the vocabulary's duration range. The
    # resulting encoded silence must also split rather than create many BARs.
    path = _midi(tmp_path, [Note(0, 10000, 60, 0), Note(10000, 10001, 64, 0)])
    encoded = encode_midi(path, max_silent_bars=2)
    assert len(encoded["sequences"]) == 2
    assert all(len(seq["ids"]) == 10 for seq in encoded["sequences"])
    assert encoded["report"]["long_durations_clipped"] == 1
    path = _midi(tmp_path, [Note(0, 1, 60, 0)], meter=(13, 4))
    encoded = encode_midi(path)
    assert encoded["sequences"] == []
    assert encoded["report"]["notes_in_unsupported_meter_segments"] == 1
    assert encoded["report"]["unsupported_segments"][0]["meter"] == [13, 4]
    assert encoded["counts"]["program_counts"] == {}


def test_conversion_reports_clipped_lengths_and_duplicate_attacks(tmp_path):
    path = _midi(
        tmp_path,
        [Note(0, 40, 60, 0), Note(0.05, 0.5, 60, 0), Note(1, 2, 60, 0)],
    )
    encoded = encode_midi(path)
    assert encoded["report"]["long_durations_clipped"] == 1
    assert encoded["report"]["same_position_notes_collapsed"] == 1
    assert encoded["report"]["same_pitch_overlaps_truncated"] == 1
    assert encoded["report"]["retained_notes"] == 2


def test_normalized_rubato_keeps_phrases_and_marks_only_true_piece_ending(tmp_path):
    path = _midi(
        tmp_path,
        [Note(0, 3, 60, 0), Note(2, 4, 64, 0), Note(4, 6, 67, 0), Note(6, 8, 72, 0)],
    )
    _replace_conductor(
        path,
        [
            (0, b"\xff\x51\x03" + (500000).to_bytes(3, "big")),
            (480, b"\xff\x51\x03" + (600000).to_bytes(3, "big")),
            (960, b"\xff\x51\x03" + (400000).to_bytes(3, "big")),
            (1920, b"\xff\x58\x04\x03\x02\x18\x08"),
        ],
    )
    old = encode_midi(path)
    encoded = encode_midi(path, normalize_tempo=True, progress_bins=16)
    assert len(encoded["sequences"]) == 2
    assert len(old["sequences"]) > len(encoded["sequences"])
    assert [sequence["is_piece_end"] for sequence in encoded["sequences"]] == [False, True]
    assert encoded["report"]["retained_notes"] == 4
    # The note through the tempo change is no longer clipped at beat one.
    assert "DUR_12" in [TOKENS[token] for token in encoded["sequences"][0]["ids"]]
    assert len({sequence["bpm"] for sequence in encoded["sequences"]}) == 1
    targets = []
    for sequence in encoded["sequences"]:
        features = sequence["progress_ids"]
        assert len(features) == len(sequence["ids"])
        assert all(0 <= value < 16 for value in features)
        windows = make_windows(
            sequence["ids"],
            32,
            progress_ids=features,
            supervise_eos=sequence["is_piece_end"],
        )
        for window in windows:
            assert len(window["progress_ids"]) == len(window["input_ids"]) == 32
            targets.extend(label for label in window["labels"] if label != -100)
    assert targets.count(EOS) == 1
    assert encoded["sequences"][-1]["progress_ids"][-1] == 15


def test_progress_stays_aligned_when_context_is_reframed():
    from musicgen.generation.runtime.structure import progress_bin, prompt_features

    ids = _token_ids(40)
    features = [progress_bin(index, len(ids) - 1, 16) for index in range(len(ids))]
    for window in make_windows(ids, 32, progress_ids=features):
        start, stop = window["source_start"], window["source_end"]
        header = ids[:3] + ([] if ids[start] == BAR else [BAR])
        framed = header + ids[start:stop]
        expected = [features[start]] * len(header) + features[start:stop]
        assert window["progress_ids"][: len(framed) - 1] == expected[:-1]
        if len(framed) >= 16 and stop > len(framed):
            prompt, values = prompt_features(ids[:stop], features[:stop], len(framed))
            assert prompt == framed
            _, positions = framed_context(ids[:stop], len(framed), return_positions=True)
            # A real preceding BAR can have the same token representation as a
            # synthetic header BAR. Source features must still follow the actual
            # framing choice, including all note-event fields.
            assert all(
                value == features[position]
                for value, position in zip(values, positions)
                if position is not None
            )
    with pytest.raises(ValueError, match="lengths"):
        prompt_features(ids, features[:-1], 32)
    with pytest.raises(ValueError, match="Progress"):
        make_windows(ids, 32, progress_ids=[0])
