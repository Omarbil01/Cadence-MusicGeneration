#!/usr/bin/env python3
"""Describe generated MIDI files without assigning a musical-quality score.

From the repository root, run with the project's Python environment:
    .venv/bin/python scripts/evaluation/analyze_midi.py FILE.mid ... --output report.json

No additional dependencies are required. Measurements describe the encoded notes;
they do not establish genre fidelity, harmonic coherence, or listener preference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from musicgen.generation.musical import structure_metrics  # noqa: E402
from musicgen.generation.runtime.midi import beat_to_seconds, read_midi  # noqa: E402

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
GM_FAMILIES = (
    "piano",
    "chromatic percussion",
    "organ",
    "guitar",
    "bass",
    "strings",
    "ensemble",
    "brass",
    "reed",
    "pipe",
    "synth lead",
    "synth pad",
    "synth effects",
    "ethnic",
    "percussive",
    "sound effects",
)


def rounded(value):
    return round(value, 6)


def summary(values):
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": rounded(min(values)),
        "median": rounded(statistics.median(values)),
        "mean": rounded(statistics.mean(values)),
        "max": rounded(max(values)),
    }


def entropy(counter):
    total = sum(counter.values())
    return (
        rounded(-sum((n / total) * math.log2(n / total) for n in counter.values()))
        if total
        else 0.0
    )


def timing_activity(notes, start, end):
    """Time-weighted activity/polyphony, with note-offs before simultaneous onsets."""
    events = []
    for a, b, _ in notes:
        a, b = max(a, start), min(b, end)
        if b > a:
            events.extend(((a, 1), (b, -1)))
    events.sort()
    cursor, voices, active, area, peak = start, 0, 0.0, 0.0, 0
    silent_start, gaps = start, []
    for at, change in events:
        width = at - cursor
        area += voices * width
        if voices:
            active += width
        before = voices
        voices += change
        peak = max(peak, voices)
        if before == 0 and voices > 0:
            if at > silent_start:
                gaps.append((silent_start, at))
        elif before > 0 and voices == 0:
            silent_start = at
        cursor = at
    if cursor < end:
        area += voices * (end - cursor)
        if voices:
            active += end - cursor
    if voices == 0 and end > silent_start:
        gaps.append((silent_start, end))
    duration = end - start
    return {
        "active_fraction": rounded(active / duration) if duration else 0.0,
        "mean_polyphony_including_silence": rounded(area / duration) if duration else 0.0,
        "mean_polyphony_while_active": rounded(area / active) if active else 0.0,
        "max_polyphony": peak,
        "longest_silence_seconds": rounded(max((b - a for a, b in gaps), default=0.0)),
        "silences_over_two_seconds": [
            {"start": rounded(a), "end": rounded(b), "duration": rounded(b - a)}
            for a, b in gaps
            if b - a > 2
        ],
    }


def analyze(path, requested_seconds):
    path = path.resolve()
    result = {"midi": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    try:
        notes, info = read_midi(path, sustain=False)
    except (ValueError, IndexError) as exc:
        return {**result, "valid_midi": False, "error": str(exc)}
    tempos, duration = info["tempos"], info["duration_seconds"]
    timed = [(beat_to_seconds(n.start, tempos), beat_to_seconds(n.end, tempos), n) for n in notes]
    pitched = [n for n in notes if n.program != 128]
    pitch_counts = Counter(n.pitch for n in pitched)
    pc_counts = Counter(n.pitch % 12 for n in pitched)
    durations = [b - a for a, b, _ in timed]
    onsets = sorted({a for a, _, _ in timed})
    program_counts = Counter(n.program for n in notes)
    early_velocity_counts = Counter(n.velocity for a, _, n in timed if a < duration * 0.8)
    onset_groups = defaultdict(list)
    for n in notes:
        onset_groups[n.start].append((n.program, n.pitch))
    onset_patterns = Counter(tuple(sorted(group)) for group in onset_groups.values())
    flags = []
    if not notes:
        flags.append("no_completed_notes")
    if info["unmatched_off"] or info["dangling_on"]:
        flags.append("unmatched_note_events")
    if abs(duration - requested_seconds) > 0.15:
        flags.append("file_duration_differs_from_requested_by_over_150ms")
    activity = timing_activity(timed, 0.0, duration)
    if activity["longest_silence_seconds"] > 5:
        flags.append("silence_over_five_seconds")

    # An onset's highest note is a reproducible contour proxy, not a voice separation.
    top_by_program = defaultdict(dict)
    for n in pitched:
        top_by_program[n.program][n.start] = max(
            n.pitch, top_by_program[n.program].get(n.start, n.pitch)
        )
    leaps = []
    for starts in top_by_program.values():
        pitches = [pitch for _, pitch in sorted(starts.items())]
        leaps.extend(abs(b - a) for a, b in zip(pitches, pitches[1:]))

    windows = []
    for start in range(0, math.ceil(duration), 10):
        end = min(start + 10, duration)
        attacks = [(a, b, n) for a, b, n in timed if start <= a < end]
        melodic_attacks = [n.pitch for _, _, n in attacks if n.program != 128]
        windows.append(
            {
                "start_seconds": start,
                "end_seconds": rounded(end),
                "note_attacks": len(attacks),
                "program_note_counts": dict(
                    sorted(Counter(n.program for _, _, n in attacks).items())
                ),
                "pitched_note_range": [min(melodic_attacks), max(melodic_attacks)]
                if melodic_attacks
                else None,
                "mean_attack_velocity": rounded(statistics.mean(n.velocity for _, _, n in attacks))
                if attacks
                else None,
                **timing_activity(timed, start, end),
            }
        )

    constant_meter = len({(n, d) for _, n, d in info["meters"]}) == 1
    constant_tempo = len({us for _, us in tempos}) == 1
    structure = None
    if notes and constant_meter and constant_tempo:
        meter = info["meters"][0][1:]
        structure = structure_metrics(
            notes, target_beats=info["duration_beats"], meter=meter, bpm=60e6 / tempos[0][1]
        )
        bar_beats = meter[0] * 4 / meter[1]
        total_complete = int((info["duration_beats"] + 1e-8) // bar_beats)
        patterns = defaultdict(list)
        for n in notes:
            bar = int(n.start // bar_beats)
            if bar < total_complete:
                patterns[bar].append(
                    (
                        rounded(n.start - bar * bar_beats),
                        n.program,
                        n.pitch,
                        rounded(n.end - n.start),
                    )
                )
        longest_run, run, prior = 0, 0, None
        for bar in range(total_complete):
            pattern = tuple(sorted(patterns[bar]))
            run = run + 1 if pattern and pattern == prior else int(bool(pattern))
            longest_run = max(longest_run, run)
            prior = pattern
        structure["longest_consecutive_identical_nonempty_bars"] = longest_run
        structure["total_complete_bars"] = total_complete

    result.update(
        {
            "valid_midi": True,
            "interpretation": "descriptive_only_not_a_musical_quality_score",
            "review_flags": flags,
            "integrity": info,
            "duration": {
                "requested_seconds": requested_seconds,
                "file_seconds": rounded(duration),
                "difference_seconds": rounded(duration - requested_seconds),
                "first_attack_seconds": rounded(min((a for a, _, _ in timed), default=duration)),
                "last_note_end_seconds": rounded(max((b for _, b, _ in timed), default=0)),
                "tail_silence_seconds": rounded(
                    duration - max((b for _, b, _ in timed), default=0)
                ),
            },
            "notes": len(notes),
            "pitched_notes": len(pitched),
            "percussion_notes": len(notes) - len(pitched),
            "note_attacks_per_second": rounded(len(notes) / duration) if duration else 0,
            "distinct_onsets": len(onsets),
            "instruments": [
                {
                    "program_zero_based": program,
                    "family": "drums" if program == 128 else GM_FAMILIES[program // 8],
                    "notes": count,
                    "fraction": rounded(count / len(notes)),
                }
                for program, count in program_counts.most_common()
            ],
            "pitch": {
                "midi_range": [min(pitch_counts), max(pitch_counts)] if pitch_counts else None,
                "unique_pitches": len(pitch_counts),
                "unique_pitch_classes": len(pc_counts),
                "most_common_pitches": [
                    {"pitch": pitch, "count": count, "fraction": rounded(count / len(pitched))}
                    for pitch, count in pitch_counts.most_common(10)
                ],
                "pitch_entropy_bits": entropy(pitch_counts),
                "pitch_class_entropy_bits": entropy(pc_counts),
                "pitch_class_counts": {PITCH_CLASSES[pc]: pc_counts[pc] for pc in range(12)},
                "highest_note_per_instrument_onset_contour_absolute_semitones": summary(leaps),
                "contour_leaps_over_octave_fraction": rounded(
                    sum(v > 12 for v in leaps) / len(leaps)
                )
                if leaps
                else None,
                "contour_same_pitch_fraction": rounded(sum(v == 0 for v in leaps) / len(leaps))
                if leaps
                else None,
                "contour_limitation": (
                    "Highest note at each instrument onset is a proxy; "
                    "polyphonic voices are not separated."
                ),
            },
            "rhythm": {
                "note_duration_seconds": summary(durations),
                "distinct_note_durations_beats": len({rounded(n.end - n.start) for n in notes}),
                "most_common_note_durations_beats": Counter(
                    rounded(n.end - n.start) for n in notes
                ).most_common(10),
                "inter_onset_seconds": summary([b - a for a, b in zip(onsets, onsets[1:])]),
            },
            "velocity": summary([n.velocity for n in notes]),
            "velocity_before_final_twenty_percent": {
                "counts": dict(sorted(early_velocity_counts.items())),
                "unique_values": len(early_velocity_counts),
                "dominant_fraction": rounded(
                    max(early_velocity_counts.values()) / sum(early_velocity_counts.values())
                )
                if early_velocity_counts
                else None,
                "interpretation": (
                    "Early encoded note velocities; excludes ending taper but does not model "
                    "instrument loudness."
                ),
            },
            "common_onset_pitch_groups": [
                {
                    "program_pitch_pairs": pattern,
                    "count": count,
                    "fraction_of_onsets": rounded(count / len(onset_groups)),
                }
                for pattern, count in onset_patterns.most_common(10)
            ],
            "activity": activity,
            "structure": structure,
            "ten_second_sections": windows,
        }
    )
    sidecars = [
        path.with_suffix(".metadata.json"),
        path.with_suffix(".json"),
        path.parent / "metadata.json",
    ]
    for sidecar in sidecars:
        if sidecar.is_file():
            try:
                metadata = json.loads(sidecar.read_text())
                result["generation_metadata"] = {
                    k: metadata[k]
                    for k in (
                        "genre",
                        "seed",
                        "checkpoint_epoch",
                        "checkpoint_sha256",
                        "device",
                        "generation_seconds",
                        "tokens",
                        "first_context_overflow_seconds",
                        "ending",
                        "repetition_control",
                        "allowed_programs",
                        "palette",
                    )
                    if k in metadata
                }
            except (ValueError, TypeError):
                pass
            else:
                break
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("midi", type=Path, nargs="+")
    parser.add_argument("--requested-seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "interpretation": (
            "Objective MIDI descriptions; genre fidelity and musical quality require listening."
        ),
        "files": [analyze(path, args.requested_seconds) for path in args.midi],
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
        print(args.output.resolve())
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
