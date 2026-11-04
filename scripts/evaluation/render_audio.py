#!/usr/bin/env python3
"""Render MIDI listening previews with the locally installed FluidSynth library.

This is an evaluation helper, not a model-quality score. It preserves parsed
notes, velocities, General MIDI programs, percussion, and the MIDI tempo map.
It does not reproduce arbitrary MIDI automation or effects messages.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY / "src"))

from musicgen.generation.runtime.midi import beat_to_seconds, read_midi  # noqa: E402


def load_library():
    name = ctypes.util.find_library("fluidsynth")
    if not name:
        raise RuntimeError("The FluidSynth shared library is not installed.")
    library = ctypes.CDLL(name)
    pointer = ctypes.c_void_p
    signatures = {
        "new_fluid_settings": ([], pointer),
        "delete_fluid_settings": ([pointer], None),
        "fluid_settings_setnum": ([pointer, ctypes.c_char_p, ctypes.c_double], ctypes.c_int),
        "fluid_settings_setint": ([pointer, ctypes.c_char_p, ctypes.c_int], ctypes.c_int),
        "new_fluid_synth": ([pointer], pointer),
        "delete_fluid_synth": ([pointer], None),
        "fluid_synth_sfload": ([pointer, ctypes.c_char_p, ctypes.c_int], ctypes.c_int),
        "fluid_synth_program_change": ([pointer, ctypes.c_int, ctypes.c_int], ctypes.c_int),
        "fluid_synth_noteon": ([pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int], ctypes.c_int),
        "fluid_synth_noteoff": ([pointer, ctypes.c_int, ctypes.c_int], ctypes.c_int),
        "fluid_synth_write_s16": (
            [
                pointer,
                ctypes.c_int,
                pointer,
                ctypes.c_int,
                ctypes.c_int,
                pointer,
                ctypes.c_int,
                ctypes.c_int,
            ],
            ctypes.c_int,
        ),
    }
    for function, (arguments, result) in signatures.items():
        getattr(library, function).argtypes = arguments
        getattr(library, function).restype = result
    return library


def render(
    midi: Path,
    destination: Path,
    *,
    soundfont: Path,
    sample_rate: int,
    tail_seconds: float,
    mp3: bool = False,
):
    notes, info = read_midi(midi, sustain=True)
    if not notes:
        raise ValueError(f"No notes in {midi}")
    melodic = sorted({note.program for note in notes if note.program != 128})
    if len(melodic) > 15:
        raise ValueError("Preview rendering supports at most 15 melodic programs.")
    channels = dict(zip(melodic, [channel for channel in range(16) if channel != 9]))
    channels[128] = 9
    events = []
    for note in notes:
        channel = channels[note.program]
        start = round(beat_to_seconds(note.start, info["tempos"]) * sample_rate)
        end = round(beat_to_seconds(note.end, info["tempos"]) * sample_rate)
        events.extend(
            [(start, 1, channel, note.pitch, note.velocity), (end, 0, channel, note.pitch, 0)]
        )
    events.sort()
    final_frame = max(events[-1][0], round(info["duration_seconds"] * sample_rate))
    final_frame += round(tail_seconds * sample_rate)
    library = load_library()
    settings = library.new_fluid_settings()
    if not settings:
        raise RuntimeError("Could not initialize FluidSynth settings.")
    synth = None
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / (midi.stem + ".wav")
    peak = 0
    clipped = 0
    try:
        library.fluid_settings_setnum(settings, b"synth.sample-rate", sample_rate)
        library.fluid_settings_setnum(settings, b"synth.gain", 0.4)
        library.fluid_settings_setint(settings, b"synth.cpu-cores", 1)
        library.fluid_settings_setint(settings, b"synth.polyphony", 256)
        library.fluid_settings_setint(settings, b"synth.reverb.active", 0)
        library.fluid_settings_setint(settings, b"synth.chorus.active", 0)
        synth = library.new_fluid_synth(settings)
        if not synth:
            raise RuntimeError("Could not initialize FluidSynth.")
        if library.fluid_synth_sfload(synth, str(soundfont).encode(), 1) < 0:
            raise RuntimeError(f"Could not load soundfont: {soundfont}")
        for program in melodic:
            library.fluid_synth_program_change(synth, channels[program], program)
        buffer = (ctypes.c_short * (4096 * 2))()
        frame = 0
        with wave.open(str(output), "wb") as audio:
            audio.setnchannels(2)
            audio.setsampwidth(2)
            audio.setframerate(sample_rate)

            def advance(target):
                nonlocal frame, peak, clipped
                while frame < target:
                    count = min(4096, target - frame)
                    result = library.fluid_synth_write_s16(synth, count, buffer, 0, 2, buffer, 1, 2)
                    if result < 0:
                        raise RuntimeError("FluidSynth audio rendering failed.")
                    values = buffer[: count * 2]
                    peak = max(peak, max(map(abs, values), default=0))
                    clipped += sum(value <= -32768 or value >= 32767 for value in values)
                    audio.writeframesraw(ctypes.string_at(buffer, count * 4))
                    frame += count

            for at, kind, channel, pitch, velocity in events:
                advance(at)
                if kind:
                    library.fluid_synth_noteon(synth, channel, pitch, velocity)
                else:
                    library.fluid_synth_noteoff(synth, channel, pitch)
            advance(final_frame)
    finally:
        if synth:
            library.delete_fluid_synth(synth)
        library.delete_fluid_settings(settings)
    result = {
        "source_midi": str(midi.resolve()),
        "wav": str(output.resolve()),
        "soundfont": str(soundfont.resolve()),
        "sample_rate": sample_rate,
        "channels": 2,
        "midi_duration_seconds": info["duration_seconds"],
        "audio_duration_seconds": final_frame / sample_rate,
        "release_tail_seconds": tail_seconds,
        "notes": len(notes),
        "programs": info["programs"],
        "pcm_peak": peak,
        "pcm_clipped_samples": clipped,
        "synth_gain": 0.4,
        "reverb": False,
        "chorus": False,
        "limitations": (
            "General MIDI preview; arbitrary MIDI automation/effects are not reproduced."
        ),
    }
    if mp3:
        executable = shutil.which("ffmpeg")
        if not executable:
            raise RuntimeError("MP3 requested but ffmpeg is not installed.")
        compressed = output.with_suffix(".mp3")
        subprocess.run(
            [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(output),
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "128k",
                str(compressed),
            ],
            check=True,
        )
        result["mp3"] = str(compressed.resolve())
    output.with_suffix(".audio.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("midi", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--soundfont", type=Path, default=Path("/usr/share/sounds/sf2/TimGM6mb.sf2")
    )
    parser.add_argument("--sample-rate", type=int, choices=[22050, 44100], default=22050)
    parser.add_argument("--tail-seconds", type=float, default=1.0)
    parser.add_argument("--mp3", action="store_true")
    arguments = parser.parse_args()
    if arguments.tail_seconds < 0:
        parser.error("--tail-seconds must be nonnegative")
    results = [
        render(
            path,
            arguments.output_dir,
            soundfont=arguments.soundfont,
            sample_rate=arguments.sample_rate,
            tail_seconds=arguments.tail_seconds,
            mp3=arguments.mp3,
        )
        for path in arguments.midi
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
