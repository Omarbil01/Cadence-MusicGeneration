"""One inference entry point for the HTTP worker and terminal CLI."""

import hashlib
import os
import time
from collections import Counter
from pathlib import Path

from musicgen.shared.config import Settings

from .musical import apply_ending, event_repetition_penalties, plan_ending, structure_metrics
from .runtime.context import framed_context, instrument_palette
from .runtime.grammar import NoteGrammar, resolve_render_overlaps
from .runtime.midi import activity, read_midi, write_midi
from .runtime.structure import progress_bin, prompt_features
from .runtime.tokenizer import EOS, GENRES, GRID, VOCAB_SHA


class ModelUnavailable(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class MidiGenerator:
    """Load once per serial worker; PyTorch is imported only when loading."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.checkpoint = None
        self.info = {}

    def load(self):
        if self.model is not None:
            return self.info
        path = self.settings.checkpoint_path
        if not path.is_file():
            raise ModelUnavailable(
                "checkpoint_missing", "Place a compatible trained checkpoint at models/best.pt."
            )
        try:
            import torch
        except ImportError as exc:
            raise ModelUnavailable(
                "pytorch_missing", "Install the worker dependencies to run generation."
            ) from exc
        from .runtime.model import MusicTransformer

        try:
            torch.set_num_threads(self.settings.torch_threads)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            signature = checkpoint["signature"]
            metadata = checkpoint["dataset_metadata"]
            if signature["vocab_sha256"] != VOCAB_SHA:
                raise ValueError("Event vocabulary differs from the supported checkpoint")
            if metadata["dataset_kind"] not in {"EXP9_APPROVED_MIDI", "PDMX_CURATED_MIDI"}:
                raise ValueError("Expected a compatible full-data MIDI training checkpoint")
            if signature["training_config"].get("smoke"):
                raise ValueError("Smoke-test checkpoints are not application models")
            if signature["dataset_sha256"] != metadata["dataset_sha256"]:
                raise ValueError("Checkpoint dataset metadata is inconsistent")
            for genre in GENRES:
                if not metadata["program_counts_train"][genre]:
                    raise ValueError(f"Missing instrument metadata for {genre}")
                if not metadata["meter_counts_train"][genre]:
                    raise ValueError(f"Missing meter metadata for {genre}")
                if genre not in metadata["drum_pitch_counts_train"]:
                    raise ValueError(f"Missing percussion metadata for {genre}")
            device = self.settings.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA was selected but is unavailable")
            model = MusicTransformer(**signature["model_config"])
            model.load_state_dict(checkpoint["model"], strict=True)
            model.to(device)
            model.eval()
            sha = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    sha.update(block)
        except Exception as exc:
            raise ModelUnavailable("checkpoint_incompatible", str(exc)) from exc
        self.checkpoint = checkpoint
        self.model = model
        self.info = {
            "checkpoint_sha256": sha.hexdigest(),
            "checkpoint_epoch": checkpoint["epoch"],
            "dataset_sha256": signature["dataset_sha256"],
            "vocab_sha256": VOCAB_SHA,
            "device": device,
            "model_context": model.context,
            "progress_bins": getattr(model, "progress_bins", 0),
            "relative_attention": getattr(model, "relative_attention", False),
            "dataset_kind": metadata["dataset_kind"],
            "inference_policy_version": "cadence-release-events-v1",
        }
        return self.info

    def generate(
        self, *, genre: str, duration_seconds: int, output_path: Path, seed: int, cancelled=None
    ) -> dict:
        import torch

        if genre not in GENRES or not 5 <= duration_seconds <= self.settings.max_duration_seconds:
            raise ValueError("Unsupported genre or duration")
        self.load()
        metadata = self.checkpoint["dataset_metadata"]
        palette = self.settings.palette if genre == "rock_pop" else "original"
        programs = instrument_palette(metadata["program_counts_train"][genre], palette)
        drum_pitches = [int(k) for k in metadata["drum_pitch_counts_train"][genre]]
        meter_counts = metadata["meter_counts_train"][genre]
        meter = tuple(map(int, max(meter_counts, key=meter_counts.get).split("/")))
        bpm = {"classical": 112, "romantic": 96, "rock_pop": 120}[genre]
        ending = plan_ending(duration_seconds * bpm / 60, bpm, meter, self.settings.ending_style)
        render_target_beats = ending.target_beats
        state = NoteGrammar(
            target_beats=ending.generation_beats,
            bpm=bpm,
            meter=meter,
            programs=programs,
            drum_pitches=drum_pitches,
            max_gap_beats=2.0,
        )
        device = next(self.model.parameters()).device
        condition = torch.tensor([GENRES.index(genre)], device=device)
        torch.manual_seed(seed)
        started = time.monotonic()
        overflow_at = None
        bins = getattr(self.model, "progress_bins", 0)
        progress = [0] * len(state.ids)
        repetition_steps = 0
        maximum_penalty = 0.0
        with torch.inference_mode():
            for _ in range(self.settings.max_tokens):
                if cancelled and cancelled():
                    raise RuntimeError("Generation was interrupted.")
                if time.monotonic() - started > self.settings.generation_timeout_seconds:
                    raise TimeoutError("Generation exceeded its time limit.")
                if state.done:
                    break
                if len(state.ids) > self.model.context and overflow_at is None:
                    overflow_at = (state.bar * state.bar_steps + state.position) / GRID * 60 / bpm
                if bins:
                    prompt, features = prompt_features(state.ids, progress, self.model.context)
                    raw = self.model(
                        torch.tensor([prompt], device=device),
                        condition,
                        torch.tensor([features], device=device),
                    )[0, -1]
                else:
                    prompt = framed_context(state.ids, self.model.context)
                    raw = self.model(torch.tensor([prompt], device=device), condition)[0, -1]
                candidate_ids = state.allowed()
                allowed = torch.tensor(candidate_ids, device=device)
                values = raw[allowed].float() / 0.85
                penalties = event_repetition_penalties(
                    state.notes,
                    phase=state.phase,
                    pending=state.pending,
                    candidate_ids=candidate_ids,
                    strength=self.settings.repetition_penalty,
                )
                if any(penalties):
                    values -= torch.tensor(penalties, device=device) / 0.85
                    repetition_steps += 1
                    maximum_penalty = max(maximum_penalty, max(penalties))
                if not bool(torch.isfinite(values).all()):
                    raise FloatingPointError("The model produced nonfinite scores.")
                values, locations = torch.topk(values, min(24, len(values)))
                sampled = torch.multinomial(torch.softmax(values, dim=0), 1)
                choice = int(allowed[locations[sampled]].item())
                state.accept(choice)
                if bins:
                    current_step = state.bar * state.bar_steps + state.position
                    progress.append(
                        bins - 1
                        if choice == EOS
                        else progress_bin(current_step, state.target, bins)
                    )
        if not state.done:
            raise RuntimeError("Generation reached its token limit.")
        notes, ending_report = apply_ending(state.notes, ending)
        notes, collisions = resolve_render_overlaps(notes)
        if not notes:
            raise RuntimeError("The model did not generate usable MIDI notes.")
        target_beats = render_target_beats
        output_path = Path(output_path)
        temporary = output_path.with_suffix(".partial.mid")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            write_midi(temporary, notes, bpm=bpm, meter=meter, end_beat=target_beats)
            reread, integrity = read_midi(temporary, sustain=False)
            if integrity["unmatched_off"] or integrity["dangling_on"] or not reread:
                raise RuntimeError("Generated MIDI failed its integrity check.")
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)
        return {
            **self.info,
            "genre": genre,
            "requested_duration_seconds": duration_seconds,
            "actual_duration_seconds": target_beats * 60 / bpm,
            "seed": seed,
            "bpm": bpm,
            "meter": list(meter),
            "notes": len(reread),
            "tokens": len(state.ids),
            "allowed_programs": programs,
            "program_note_counts": dict(Counter(n.program for n in reread)),
            "palette": palette,
            "context_mode": "framed",
            "first_context_overflow_seconds": overflow_at,
            "same_position_collisions_removed": collisions,
            "ending": ending_report,
            "structure": structure_metrics(reread, target_beats=target_beats, meter=meter, bpm=bpm),
            "repetition_control": {
                "strength": self.settings.repetition_penalty,
                "event_ngram": 4,
                "free_prior_occurrences": 2,
                "penalized_sampling_steps": repetition_steps,
                "maximum_logit_penalty": round(maximum_penalty, 6),
                "percussion_exempt": True,
            },
            "generation_seconds": round(time.monotonic() - started, 3),
            "activity": activity(reread, 0, target_beats),
            "midi_integrity": integrity,
        }
