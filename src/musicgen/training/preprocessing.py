"""Deterministic MIDI encoding and complete-event causal training windows.

Meter and tempo changes start separate sequences. Notes crossing a change are
clipped, rather than reattacked in the next sequence. Tempos use the nearest
supported BPM (lower value on a tie). Each sequence keeps its source meter's
bar origin, trims empty edge bars, and splits internal silence longer than
``max_silent_bars``. Timing uses the runtime's four steps per quarter note;
durations beyond 128 steps are clipped and every conversion is reported.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from musicgen.generation.runtime.midi import Note, read_midi
from musicgen.generation.runtime.structure import progress_bin
from musicgen.generation.runtime.tokenizer import (
    BAR,
    BOS,
    EOS,
    GRID,
    METERS,
    PAD,
    TEMPOS,
    TO_ID,
    TOKENS,
)

IGNORE_INDEX = -100


def _active_metadata(info: dict, beat: float) -> tuple[tuple[int, int], float, float]:
    meter, origin, microseconds = (4, 4), 0.0, 500000
    for at, numerator, denominator in info["meters"]:
        if at > beat:
            break
        meter, origin = (numerator, denominator), at
    for at, tempo in info["tempos"]:
        if at > beat:
            break
        microseconds = tempo
    return meter, origin, 60000000 / microseconds


def _quantize(notes: list[Note], origin: float, end: float, report: Counter) -> list[Note]:
    limit = round((end - origin) * GRID)
    voices = defaultdict(list)
    for note in notes:
        start = max(0, round((note.start - origin) * GRID))
        if start >= limit:
            report["quantized_onsets_outside_segment"] += 1
            continue
        stop = min(limit, max(start + 1, round((min(note.end, end) - origin) * GRID)))
        if note.end > end:
            report["notes_clipped_at_metadata_change"] += 1
        if stop - start > 128:
            stop = start + 128
            report["long_durations_clipped"] += 1
        velocity = max(1, min(16, round(note.velocity / 8))) * 8
        converted = replace(note, start=start / GRID, end=stop / GRID, velocity=min(127, velocity))
        voices[note.program, note.pitch].append(converted)
    result = []
    for voice in voices.values():
        kept = []
        for note in sorted(voice, key=lambda n: (n.start, n.end, n.velocity)):
            if kept and note.start == kept[-1].start:
                kept[-1] = note
                report["same_position_notes_collapsed"] += 1
                continue
            if kept and note.start < kept[-1].end:
                kept[-1] = replace(kept[-1], end=note.start)
                report["same_pitch_overlaps_truncated"] += 1
            kept.append(note)
        result.extend(kept)
    return sorted(result, key=lambda n: (n.start, n.program, n.pitch, n.end))


def _sequence(notes: list[Note], bpm: int, meter: tuple[int, int]) -> list[int]:
    ids = [BOS, TO_ID[f"BPM_{bpm}"], TO_ID[f"METER_{meter[0]}_{meter[1]}"], BAR]
    bar_steps = round(meter[0] * 4 / meter[1] * GRID)
    current_bar = 0
    for note in notes:
        start = round(note.start * GRID)
        bar, position = divmod(start, bar_steps)
        ids.extend([BAR] * (bar - current_bar))
        current_bar = bar
        ids.extend(
            [
                TO_ID[f"POS_{position}"],
                TO_ID[f"PROGRAM_{note.program}"],
                TO_ID[f"PITCH_{note.pitch}"],
                TO_ID[f"DUR_{round((note.end - note.start) * GRID)}"],
                TO_ID[f"VEL_{max(1, min(16, round(note.velocity / 8)))}"],
            ]
        )
    return ids + [EOS]


def encode_midi(
    path: str | Path,
    *,
    sustain: bool = True,
    max_unmatched_ratio: float = 0.03,
    max_silent_bars: int = 32,
    normalize_tempo: bool = False,
    progress_bins: int = 0,
) -> dict:
    """Return JSON-serializable sequences, file counts and a conversion report.

    Counts describe retained encoded notes once per source file, before window
    overlap. The caller must aggregate them from the training split only.
    Unsupported-meter segments are skipped explicitly in ``report``. A file
    with no retained sequences must be rejected by dataset preparation.
    """
    if not 0 <= max_unmatched_ratio <= 1:
        raise ValueError("max_unmatched_ratio must be between zero and one")
    if isinstance(max_silent_bars, bool) or not isinstance(max_silent_bars, int):
        raise ValueError("max_silent_bars must be a positive integer")
    if not 1 <= max_silent_bars <= 4096:
        raise ValueError("max_silent_bars must be between 1 and 4096")
    if not isinstance(normalize_tempo, bool):
        raise ValueError("normalize_tempo must be boolean")
    if isinstance(progress_bins, bool) or not isinstance(progress_bins, int):
        raise ValueError("progress_bins must be zero or an integer between 2 and 64")
    if progress_bins != 0 and not 2 <= progress_bins <= 64:
        raise ValueError("progress_bins must be zero or an integer between 2 and 64")
    notes, info = read_midi(path, sustain=sustain)
    if not notes:
        raise ValueError("MIDI has no matched positive-duration notes")
    if info["unmatched_event_ratio"] > max_unmatched_ratio:
        raise ValueError("MIDI exceeds the unmatched-note-event threshold")
    end = max(note.end for note in notes)
    piece_start = min(note.start for note in notes)
    # MIDI beats already express score timing. Normalizing BPM retains that
    # timing while avoiding thousands of tiny sequences from rubato changes.
    tempo_weights = Counter()
    tempo_changes = sorted({0.0, end} | {at for at, _ in info["tempos"] if 0 <= at < end})
    for start, stop in zip(tempo_changes, tempo_changes[1:]):
        _, _, value = _active_metadata(info, start)
        tempo_weights[value] += stop - start
    dominant_bpm = max(tempo_weights, key=tempo_weights.get)
    changes = sorted(
        {0.0, end}
        | {at for at, *_ in info["meters"] if 0 <= at < end}
        | (set() if normalize_tempo else {at for at, _ in info["tempos"] if 0 <= at < end})
    )
    report = Counter(input_notes=len(notes))
    sequences = []
    programs, drums, meters = Counter(), Counter(), Counter()
    unsupported = []
    for start, stop in zip(changes, changes[1:]):
        selected = [note for note in notes if start <= note.start < stop]
        if not selected:
            continue
        meter, meter_origin, source_bpm = _active_metadata(info, start)
        if normalize_tempo:
            source_bpm = dominant_bpm
        bar_beats = meter[0] * 4 / meter[1]
        if meter not in METERS or not 1 <= round(bar_beats * GRID) <= 96:
            report["unsupported_meter_segments"] += 1
            report["notes_in_unsupported_meter_segments"] += len(selected)
            unsupported.append({"start_beat": start, "end_beat": stop, "meter": list(meter)})
            continue
        bpm = min(TEMPOS, key=lambda value: abs(value - source_bpm))
        if not math.isclose(bpm, source_bpm, rel_tol=1e-6):
            report["tempo_values_quantized"] += 1
        chunks = []
        active_end = min(selected[0].end, selected[0].start + 128 / GRID)
        current = []
        for note in selected:
            if current and note.start - active_end > max_silent_bars * bar_beats:
                chunks.append(current)
                current = []
                report["long_silence_splits"] += 1
            current.append(note)
            representable_end = min(note.end, note.start + 128 / GRID)
            active_end = (
                max(representable_end, active_end) if len(current) > 1 else representable_end
            )
        chunks.append(current)
        for chunk in chunks:
            origin = (
                meter_origin + math.floor((chunk[0].start - meter_origin) / bar_beats) * bar_beats
            )
            chunk_end = min(stop, max(note.end for note in chunk))
            converted = _quantize(chunk, origin, chunk_end, report)
            if not converted:
                report["empty_quantized_segments"] += 1
                continue
            ids = _sequence(converted, bpm, meter)
            sequences.append(
                {
                    "ids": ids,
                    "start_beat": origin,
                    "end_beat": chunk_end,
                    "bpm": bpm,
                    "meter": list(meter),
                    "notes": len(converted),
                }
            )
            if normalize_tempo or progress_bins:
                sequences[-1]["is_piece_end"] = math.isclose(chunk_end, end, abs_tol=1e-6)
            if progress_bins:
                values, bar, position = [], -1, 0
                bar_steps = round(bar_beats * GRID)
                for token in ids:
                    if token == BAR:
                        bar += 1
                        position = 0
                    elif TOKENS[token].startswith("POS_"):
                        position = int(TOKENS[token].split("_")[1])
                    at = origin + (max(0, bar) * bar_steps + position) / GRID
                    if token == EOS:
                        at = chunk_end
                    values.append(progress_bin(at - piece_start, end - piece_start, progress_bins))
                sequences[-1]["progress_ids"] = values
            meters[f"{meter[0]}/{meter[1]}"] += 1
            for note in converted:
                programs[str(note.program)] += 1
                if note.program == 128:
                    drums[str(note.pitch)] += 1
    report["retained_notes"] = sum(sequence["notes"] for sequence in sequences)
    report["sequences"] = len(sequences)
    return {
        "sequences": sequences,
        "counts": {
            "program_counts": dict(programs),
            "drum_pitch_counts": dict(drums),
            "meter_counts": dict(meters),
        },
        "report": {
            **dict(report),
            "unsupported_segments": unsupported,
            "unmatched_event_ratio": info["unmatched_event_ratio"],
            "tempo_policy": (
                "normalize_to_dominant_bpm_preserving_score_beats"
                if normalize_tempo
                else "split_at_changes_then_nearest_supported_bpm"
            ),
            "progress_bins": progress_bins,
            "meter_policy": "split_at_changes_preserving_source_bar_origin",
            "silence_policy": "trim_empty_edge_bars_and_split_long_internal_silence",
        },
    }


def _groups(ids: Sequence[int]) -> list[tuple[int, int]]:
    if any(isinstance(token, bool) or not isinstance(token, int) for token in ids):
        raise ValueError("Token IDs must be integers")
    if any(token < 0 or token >= len(TOKENS) for token in ids):
        raise ValueError("Unknown token ID")
    if (
        len(ids) < 10
        or ids[0] != BOS
        or not TOKENS[ids[1]].startswith("BPM_")
        or not TOKENS[ids[2]].startswith("METER_")
        or ids[3] != BAR
        or ids[-1] != EOS
    ):
        raise ValueError("Expected BOS/BPM/METER/BAR, complete notes and final EOS")
    _, numerator, denominator = TOKENS[ids[2]].split("_")
    bar_steps = round(int(numerator) * 4 / int(denominator) * GRID)
    groups, index, previous_position = [], 3, -1
    note_count = 0
    while index < len(ids) - 1:
        if ids[index] == BAR:
            groups.append((index, index + 1))
            previous_position = -1
            index += 1
            continue
        fields = ("POS_", "PROGRAM_", "PITCH_", "DUR_", "VEL_")
        if index + 5 > len(ids) - 1 or any(
            not TOKENS[ids[index + offset]].startswith(field) for offset, field in enumerate(fields)
        ):
            raise ValueError("Malformed or partial note event")
        position = int(TOKENS[ids[index]].split("_")[1])
        if not previous_position <= position < bar_steps:
            raise ValueError("Note position is outside the bar or moves backward")
        previous_position = position
        groups.append((index, index + 5))
        note_count += 1
        index += 5
    if not note_count:
        raise ValueError("Sequence has no complete notes")
    return groups + [(len(ids) - 1, len(ids))]


def make_windows(
    ids: Sequence[int], context: int, *, progress_ids=None, supervise_eos=True
) -> list[dict]:
    """Frame complete events and supervise each real post-header token once.

    ``input_ids`` and next-token ``labels`` have exactly ``context`` entries.
    Labels for the conditioning header, reused overlap and padding are -100.
    Continuations reuse one complete source event and prepend BOS/BPM/METER
    plus BAR if that event is a note, matching runtime ``framed_context``.
    EOS is supervised only at the actual segment end, never at a window cut.
    Source bounds identify the original overlap and new target interval.
    """
    if isinstance(context, bool) or not isinstance(context, int) or context < 16:
        raise ValueError("context must be an integer of at least 16")
    ids = list(ids)
    if progress_ids is not None:
        if len(progress_ids) != len(ids) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in progress_ids
        ):
            raise ValueError("Progress must contain one nonnegative integer per source token")
    groups = _groups(ids)
    windows, next_group, target_start = [], 0, 4
    while next_group < len(groups):
        first_group = max(0, next_group - 1)
        source_start = groups[first_group][0]
        header = ids[:3] + ([] if ids[source_start] == BAR else [BAR])
        end_group = next_group
        while end_group < len(groups):
            source_end = groups[end_group][1]
            if len(header) + source_end - source_start > context + 1:
                break
            end_group += 1
        if end_group <= next_group:
            raise ValueError("Context cannot fit framing and a complete new event")
        source_end = groups[end_group - 1][1]
        framed = header + ids[source_start:source_end]
        positions = [None] * len(header) + list(range(source_start, source_end))
        inputs = framed[:-1]
        labels = [
            token
            if position is not None and position >= target_start and (token != EOS or supervise_eos)
            else IGNORE_INDEX
            for token, position in zip(framed[1:], positions[1:])
        ]
        padding = context - len(inputs)
        windows.append(
            {
                "input_ids": inputs + [PAD] * padding,
                "labels": labels + [IGNORE_INDEX] * padding,
                "source_start": source_start,
                "source_end": source_end,
                "target_start": target_start,
            }
        )
        if progress_ids is not None:
            feature_values = [progress_ids[source_start]] * len(header) + list(
                progress_ids[source_start:source_end]
            )
            windows[-1]["progress_ids"] = feature_values[:-1] + [0] * padding
        next_group, target_start = end_group, source_end
    return windows
