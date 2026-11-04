# One-minute MIDI examples

These are generated samples from the balanced CPU checkpoint:
one 60-second composition for each supported genre. Each is the first frontend
trial for its genre from the recorded evaluation, retained without editing.

| Sample | Genre | 
|---|---|---:|---:|
| [classical_60s.mid](classical_60s.mid) | Classical | 
| [romantic_60s.mid](romantic_60s.mid) | Romantic | 
| [rock_pop_60s.mid](rock_pop_60s.mid) | Rock & Pop | 

Open the files in a DAW or MIDI player with a synthesizer/soundfont to hear them.
MIDI stores notes and performance instructions; audio previews are not included.
The Classical file encodes 59.999968 seconds because of MIDI tempo precision;
the other files encode 60 seconds. All have balanced note-on/note-off events.

[metadata.json](metadata.json) records checkpoint identity, model configuration,
random seeds, sampling controls, timing observations and file hashes. Weights and
training data are not included. See [evaluation results](../docs/evaluation.md)
for the larger batch, timing ranges and remaining music-quality observations.

The release ending and repetition penalty are inference controls, so a tapered
ending does not establish a learned harmonic cadence. These examples illustrate
the model's output and are not quality scores.

Only these three named MIDI examples are intentionally included in Git. Keep
routine generations under ignored `storage/generated/`; adding another public
MIDI example requires updating the explicit ignore-rule allowance and provenance.
