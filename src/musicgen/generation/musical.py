"""Transparent decoding and release heuristics, independent of model weights.

These policies can reduce exact event loops and leave an audible release. They
do not infer a key, compose a harmonic cadence, or establish historical style.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace

from .runtime.midi import Note
from .runtime.tokenizer import GRID, TOKENS


@dataclass(frozen=True)
class EndingPlan:
    style: str
    target_beats: float
    generation_beats: float
    taper_start: float
    last_onset: float
    release_start: float


def plan_ending(target_beats: float, bpm: float, meter, style="release") -> EndingPlan:
    """Reserve held notes and silence *inside* the requested MIDI duration."""
    if style not in {"release", "none"}:
        raise ValueError("Unknown ending style")
    if not math.isfinite(target_beats) or not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("Ending duration and tempo must be finite and positive")
    total = round(target_beats * GRID)
    if total < 8:
        raise ValueError("Ending planning needs at least two quarter-note beats")
    target = total / GRID
    if style == "none":
        return EndingPlan(style, target, target, target, target, target)
    bar_steps = round(meter[0] * 4 / meter[1] * GRID)
    if not 1 <= bar_steps <= 96:
        raise ValueError("Unsupported ending meter")
    duration_seconds = target * 60 / bpm
    tail_seconds = min(0.75, duration_seconds * 0.08)
    tail = min(total - 3, max(1, round(tail_seconds * bpm / 60 * GRID)))
    release = total - tail
    taper = min(2 * bar_steps, max(4, round(total * 0.18)), release)
    hold = min(release - 1, max(2, round(taper * 0.2)))
    last_onset = release - hold
    # The grammar's last possible onset is one grid step before its target.
    # Stop generation early instead of forcing attacks into the rendered release.
    generation = last_onset + 1
    return EndingPlan(
        style,
        target,
        generation / GRID,
        (release - taper) / GRID,
        last_onset / GRID,
        release / GRID,
    )


def apply_ending(notes: list[Note], plan: EndingPlan) -> tuple[list[Note], dict]:
    """Thin onset groups, soften attacks, and release the actual final sonority.

    Chord groups stay together. Only the last existing melodic onset is sustained;
    pitches are never added or labelled a tonic cadence. Percussion is not held.
    """
    if not notes:
        raise ValueError("Cannot finish an empty note sequence")
    if plan.style == "none":
        return list(notes), {"style": "none", "harmonic_cadence": False}
    groups = defaultdict(list)
    for note in notes:
        if note.start <= plan.last_onset + 1e-8:
            groups[round(note.start * GRID)].append(note)
    kept = []
    last_step = None
    removed_groups = 0
    softened = 0
    width = max(1 / GRID, plan.release_start - plan.taper_start)
    for step, group in sorted(groups.items()):
        onset = step / GRID
        progress = min(1.0, max(0.0, (onset - plan.taper_start) / width))
        # A growing minimum distance between attacks preserves whole sonorities.
        minimum_gap = 1 + round(3 * progress)
        if onset >= plan.taper_start and last_step is not None and step - last_step < minimum_gap:
            removed_groups += 1
            continue
        last_step = step
        smooth = progress * progress * (3 - 2 * progress)
        for note in group:
            velocity = max(1, min(127, round(note.velocity * (1 - 0.45 * smooth))))
            end = min(note.end, plan.release_start)
            if end <= note.start:
                continue
            softened += velocity < note.velocity
            kept.append(replace(note, end=end, velocity=velocity))
    if not kept:
        raise ValueError("No usable notes precede the planned release")
    melodic = [note for note in kept if note.program != 128]
    final_onset = max((note.start for note in melodic), default=None)
    held = 0
    # Do not resurrect an old melodic note when the closing section is drums-only.
    if final_onset is not None and final_onset >= plan.taper_start:
        for index, note in enumerate(kept):
            if note.program != 128 and note.start == final_onset and note.end < plan.release_start:
                kept[index] = replace(note, end=plan.release_start)
                held += 1
    kept.sort(key=lambda note: (note.start, note.program, note.pitch, note.end))
    last_end = max(note.end for note in kept)
    return kept, {
        "style": "release",
        "policy": "onset_density_and_velocity_taper_then_held_release",
        "harmonic_cadence": False,
        "taper_start_beat": plan.taper_start,
        "last_allowed_onset_beat": plan.last_onset,
        "release_start_beat": plan.release_start,
        "tail_silence_beats": round(plan.target_beats - last_end, 6),
        "notes_removed": len(notes) - len(kept),
        "onset_groups_removed": removed_groups,
        "notes_softened": softened,
        "final_melodic_notes_held": held,
        "last_onset_beat": max(note.start for note in kept),
    }


def event_repetition_penalties(
    notes: list[Note],
    *,
    phase: str,
    pending: dict,
    candidate_ids,
    strength=0.3,
    ngram=4,
    free_occurrences=2,
    history_limit=96,
) -> list[float]:
    """Softly penalize repeated melodic note-event continuations.

    A match needs the preceding events' pitches, durations and onset intervals.
    The first two prior occurrences are free so motifs and accompaniment can
    repeat. Only pitch/duration candidates are changed; formatting tokens and
    percussion are exempt. No candidate is forbidden by this policy.
    """
    candidates = list(candidate_ids)
    result = [0.0] * len(candidates)
    if strength <= 0 or phase not in {"pitch", "duration"} or pending.get("program") == 128:
        return result
    if ngram < 2 or history_limit < ngram or free_occurrences < 0:
        raise ValueError("Invalid event repetition history configuration")
    history = []
    for note in reversed(notes):
        if note.program == pending.get("program"):
            history.append(note)
            if len(history) == history_limit:
                break
    history.reverse()
    if len(history) < ngram:
        return result
    events = []
    previous = None
    for note in history:
        start = round(note.start * GRID)
        events.append(
            (
                note.pitch,
                round((note.end - note.start) * GRID),
                None if previous is None else start - previous,
            )
        )
        previous = start
    prefix = events[-(ngram - 1) :]
    next_delta = pending["start"] - previous
    matches = [
        event
        for index, event in enumerate(events)
        if index >= ngram - 1
        and events[index - ngram + 1 : index] == prefix
        and event[2] == next_delta
    ]
    match_counts = Counter(
        event[0] if phase == "pitch" else (event[0], event[1]) for event in matches
    )
    token_prefix = "PITCH_" if phase == "pitch" else "DUR_"
    for index, token_id in enumerate(candidates):
        token = TOKENS[token_id]
        if not token.startswith(token_prefix):
            continue
        value = int(token[len(token_prefix) :])
        key = value if phase == "pitch" else (pending.get("pitch"), value)
        count = match_counts[key]
        excess = max(0, count - free_occurrences)
        result[index] = min(0.9, strength * math.log1p(excess))
    return result


def structure_metrics(notes: list[Note], *, target_beats: float, meter, bpm: float) -> dict:
    """Describe exact repetition and the ending envelope, without a quality score."""
    if not notes or target_beats <= 0 or bpm <= 0:
        raise ValueError("Structure metrics need notes, positive duration and tempo")
    bar_steps = round(meter[0] * 4 / meter[1] * GRID)
    if bar_steps < 1:
        raise ValueError("Invalid metric meter")
    full_bars = round(target_beats * GRID) // bar_steps
    bars = defaultdict(list)
    events = []
    previous = None
    for note in sorted(notes, key=lambda item: (item.start, item.program, item.pitch, item.end)):
        step = round(note.start * GRID)
        duration = round((note.end - note.start) * GRID)
        bar, position = divmod(step, bar_steps)
        if bar < full_bars:
            bars[bar].append((position, note.program, note.pitch, duration))
        events.append(
            (note.program, note.pitch, duration, None if previous is None else step - previous)
        )
        previous = step
    bar_counts = Counter(tuple(sorted(pattern)) for pattern in bars.values())
    nonempty_bars = sum(bar_counts.values())
    event_counts = Counter(tuple(events[index : index + 4]) for index in range(len(events) - 3))
    event_windows = sum(event_counts.values())
    ending_width = min(2 * bar_steps / GRID, target_beats * 0.2)
    ending_start = target_beats - ending_width

    def section(start, end):
        attacks = [note for note in notes if start <= note.start < end]
        width = max(1 / GRID, end - start)
        return {
            "start_beat": round(start, 6),
            "end_beat": round(end, 6),
            "note_attacks": len(attacks),
            "note_attacks_per_beat": round(len(attacks) / width, 6),
            "onsets_per_beat": round(len({note.start for note in attacks}) / width, 6),
            "mean_velocity": (
                round(sum(note.velocity for note in attacks) / len(attacks), 6) if attacks else None
            ),
        }

    return {
        "interpretation": "descriptive_only_not_a_musical_quality_score",
        "bar_pattern_fields": ["onset_step", "program", "pitch", "duration_steps"],
        "nonempty_complete_bars": nonempty_bars,
        "repeated_bar_fraction": round(
            (nonempty_bars - len(bar_counts)) / max(1, nonempty_bars), 6
        ),
        "event_ngram": 4,
        "repeated_event_ngram_fraction": round(
            sum(max(0, count - 1) for count in event_counts.values()) / max(1, event_windows), 6
        ),
        "earlier_section": section(0, ending_start),
        "ending_section": section(ending_start, target_beats),
        "silence_tail_seconds": round(
            max(0.0, target_beats - max(note.end for note in notes)) * 60 / bpm, 6
        ),
        "last_attack_before_end_seconds": round(
            max(0.0, target_beats - max(note.start for note in notes)) * 60 / bpm, 6
        ),
    }
