import pytest

from musicgen.generation.musical import (
    apply_ending,
    event_repetition_penalties,
    plan_ending,
    structure_metrics,
)
from musicgen.generation.runtime.midi import Note, read_midi, write_midi
from musicgen.generation.runtime.tokenizer import BAR, TO_ID


def test_release_reduces_final_density_and_leaves_silence_inside_duration(tmp_path):
    notes = [Note(step / 4, step / 4 + 0.25, 60 + step % 4, 0, 100) for step in range(80)]
    plan = plan_ending(20, 120, (4, 4))
    finished, report = apply_ending(notes, plan)
    assert report["harmonic_cadence"] is False
    assert report["notes_removed"] > 0
    assert max(note.start for note in finished) <= plan.last_onset
    assert max(note.end for note in finished) == plan.release_start < plan.target_beats
    assert report["tail_silence_beats"] >= 0.25
    assert finished[0] == notes[0]
    assert finished[-1].velocity < notes[-1].velocity
    assert all(note.end > note.start and 1 <= note.velocity <= 127 for note in finished)
    path = tmp_path / "release.mid"
    write_midi(path, finished, end_beat=plan.target_beats)
    reread, info = read_midi(path)
    assert reread and not info["unmatched_off"] and not info["dangling_on"]
    assert info["duration_seconds"] == pytest.approx(10)
    assert max(note.end for note in reread) < info["duration_beats"]


@pytest.mark.parametrize(
    "seconds,bpm,meter",
    [
        (5, 96, (4, 4)),
        (5, 112, (2, 4)),
        (60, 120, (4, 4)),
        (240, 112, (3, 4)),
    ],
)
def test_generation_stop_precedes_release_without_shortening_requested_file(seconds, bpm, meter):
    plan = plan_ending(seconds * bpm / 60, bpm, meter)
    assert 0 <= plan.taper_start < plan.last_onset < plan.release_start < plan.target_beats
    assert plan.generation_beats <= plan.release_start
    assert abs(plan.target_beats * 60 / bpm - seconds) <= 0.125 * 60 / bpm


def test_release_preserves_whole_chords_and_never_holds_drums():
    notes = [
        Note(step / 4, step / 4 + 0.25, pitch, program, 80)
        for step in range(64)
        for pitch, program in [(60, 0), (64, 0), (36, 128)]
    ]
    plan = plan_ending(16, 120, (4, 4))
    finished, report = apply_ending(notes, plan)
    assert report["final_melodic_notes_held"] == 2
    assert all(note.end - note.start <= 0.25 for note in finished if note.program == 128)
    for onset in {note.start for note in finished}:
        assert {note.pitch for note in finished if note.start == onset} == {36, 60, 64}


def test_release_does_not_resurrect_old_melody_under_percussion_ending():
    notes = [Note(0, 1, 60, 0, 80)]
    notes.extend(Note(beat, beat + 0.25, 36, 128, 80) for beat in range(1, 120))
    plan = plan_ending(120, 120, (4, 4))
    finished, report = apply_ending(notes, plan)
    assert report["final_melodic_notes_held"] == 0
    assert next(note for note in finished if note.program == 0).end == 1


def test_none_preserves_notes_and_original_generation_target():
    notes = [Note(0, 2, 60, 0, 80), Note(3, 4, 64, 0, 96)]
    plan = plan_ending(4, 120, (4, 4), "none")
    finished, report = apply_ending(notes, plan)
    assert finished == notes and plan.generation_beats == plan.target_beats == 4
    assert report == {"style": "none", "harmonic_cadence": False}


def test_structure_metrics_identify_exact_repeated_bars_without_a_quality_claim():
    notes = [Note(index, index + 0.5, (60, 62, 64, 65)[index % 4], 0, 80) for index in range(12)]
    metrics = structure_metrics(notes, target_beats=12, meter=(4, 4), bpm=120)
    assert metrics["nonempty_complete_bars"] == 3
    assert metrics["repeated_bar_fraction"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["silence_tail_seconds"] == 0.25
    assert metrics["interpretation"] == "descriptive_only_not_a_musical_quality_score"
    assert metrics["earlier_section"]["mean_velocity"] == 80


def test_structure_metrics_report_lighter_quieter_ending_and_release():
    notes = [Note(step / 4, step / 4 + 0.25, 60 + step % 4, 0, 100) for step in range(80)]
    plan = plan_ending(20, 120, (4, 4))
    finished, _ = apply_ending(notes, plan)
    metrics = structure_metrics(finished, target_beats=20, meter=(4, 4), bpm=120)
    assert (
        metrics["ending_section"]["note_attacks_per_beat"]
        < metrics["earlier_section"]["note_attacks_per_beat"]
    )
    assert metrics["ending_section"]["mean_velocity"] < metrics["earlier_section"]["mean_velocity"]
    assert metrics["silence_tail_seconds"] > 0


def repeated_notes(cycles=5, program=0):
    return [
        Note(index, index + 0.5, (60, 62, 64, 65)[index % 4], program, 80)
        for index in range(4 * cycles)
    ]


def test_repetition_penalty_matches_events_and_exempts_format_tokens():
    notes = repeated_notes()
    candidates = [TO_ID["PITCH_60"], TO_ID["PITCH_67"], BAR, TO_ID["POS_0"], TO_ID["PROGRAM_0"]]
    penalties = event_repetition_penalties(
        notes,
        phase="pitch",
        pending={"program": 0, "start": 80},
        candidate_ids=candidates,
    )
    assert 0 < penalties[0] <= 0.9
    assert penalties[1:] == [0.0] * 4
    duration_penalties = event_repetition_penalties(
        notes,
        phase="duration",
        pending={"program": 0, "pitch": 60, "start": 80},
        candidate_ids=[TO_ID["DUR_2"], TO_ID["DUR_4"], BAR],
    )
    assert duration_penalties[0] > 0 and duration_penalties[1:] == [0, 0]


def test_repetition_preserves_first_motif_repeats_and_different_rhythm():
    candidates = [TO_ID["PITCH_60"]]
    for notes, pending in [
        (repeated_notes(2), {"program": 0, "start": 32}),
        (repeated_notes(), {"program": 0, "start": 81}),
        (repeated_notes(program=128), {"program": 128, "start": 80}),
    ]:
        assert event_repetition_penalties(
            notes,
            phase="pitch",
            pending=pending,
            candidate_ids=candidates,
        ) == [0]
    assert event_repetition_penalties(
        repeated_notes(),
        phase="program",
        pending={"program": 0, "start": 80},
        candidate_ids=[TO_ID["PROGRAM_0"]],
    ) == [0]


@pytest.mark.checkpoint
def test_generation_supports_progress_checkpoint_and_short_framed_context(
    settings, checkpoint, tmp_path
):
    torch = pytest.importorskip("torch")
    from musicgen.generation.generator import MidiGenerator
    from musicgen.generation.runtime.model import MusicTransformer

    record = torch.load(checkpoint, weights_only=True)
    model = MusicTransformer(
        context=32,
        d_model=32,
        layers=1,
        heads=4,
        dropout=0,
        progress_bins=8,
        relative_attention=True,
    )
    record["model"] = model.state_dict()
    record["signature"]["model_config"] = model.config
    settings.checkpoint_path = tmp_path / "progress.pt"
    torch.save(record, settings.checkpoint_path)
    output = tmp_path / "progress.mid"
    metadata = MidiGenerator(settings).generate(
        genre="romantic",
        duration_seconds=5,
        output_path=output,
        seed=42,
    )
    assert metadata["progress_bins"] == 8
    assert metadata["first_context_overflow_seconds"] is not None
    notes, info = read_midi(output)
    assert notes and not info["unmatched_off"] and not info["dangling_on"]
    assert info["duration_seconds"] == pytest.approx(5, abs=0.1)
    assert max(note.end for note in notes) < info["duration_beats"]
