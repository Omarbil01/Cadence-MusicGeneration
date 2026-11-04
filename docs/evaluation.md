# Checkpoint evaluation

The balanced CPU run completed 500 epochs. On 4 November 2026, its final checkpoint was tested through the real Flask frontend and FastAPI backend. Listening feedback favored this checkpoint over the previous model; this is qualitative feedback, not a controlled comparison or a numerical listening score.

## Checkpoint and method

| Item | Evaluated configuration |
|---|---|
| Checkpoint | `balanced250_cpu_v1/last.pt`, epoch 500 |
| SHA-256 | `6ad542c47566bcf09f34d5c170a65394e8b31d30200eb98a9fce63ad143590f0` |
| Model | 747,520 parameters; context 256, width 128, three layers, four heads, relative attention, 16 progress bins |
| Hardware | AMD Ryzen 7 8840U; CPU, two Torch threads; PyTorch 2.6.0+cpu |
| Requests | 18 serial 60-second jobs: three frontend and three direct-API jobs per genre |
| Sampling controls | Original palette, release ending, repetition penalty 0.3, maximum 30,000 tokens |

All requests used normal server-assigned random seeds. Each MIDI and metadata download was checked against its job output and checkpoint identity. The first inference was retained. Model loading took 1.22 seconds; service startup to readiness took 2.07 seconds, measured separately from generation.

The run's validation-selected `best.pt` is epoch 482, with validation loss 0.930138; the evaluated final checkpoint has loss 0.930279. Checkpoint filenames and training budgets do not identify the saved epoch on their own. See [checkpoint setup](../models/README.md) and [training](training.md) before choosing an artifact for the application.

## One-minute generation timings

Times are seconds, shown as median (minimum–maximum). Generation pools all six samples per genre; each route's ready time contains three samples.

| Genre | Generation, n=6 | Frontend ready, n=3 | Direct API ready, n=3 |
|---|---:|---:|---:|
| Classical | 10.83 (2.69–18.29) | 12.43 (10.89–19.94) | 3.23 (2.83–11.52) |
| Romantic | 6.30 (3.28–15.18) | 6.36 (4.85–16.92) | 7.07 (4.86–11.12) |
| Rock & Pop | 13.31 (6.82–18.18) | 15.41 (15.41–19.95) | 7.27 (6.87–12.12) |

Generation time covers sampling, MIDI export and validation. Frontend time runs from the Generate click to the visible download link; direct-API time runs from POST to observed completion. Frontend polling is every 1.5 seconds; the benchmark's direct-API polling is every 0.2 seconds.

Independent seeds produced different note and token densities, so the route medians do not isolate frontend overhead. For example, Classical samples ranged from 117 to 648 notes. Paired frontend observation delay above each job's own worker time had medians of 0.62 seconds (Classical), 1.57 seconds (Romantic), and 0.90 seconds (Rock & Pop). These are small-sample local-mode results; Docker, Redis/Celery, GPU and concurrent load were not measured.

## MIDI and music review

All 18 outputs were valid, nonempty MIDI files with approximately 60-second timelines, zero dangling note-on events, and zero unmatched note-off events. Every downloaded MIDI matched its native job file byte for byte. The browser recorded no JavaScript errors.

The [curated output examples](../outputs/README.md) preserve the first frontend trial for each genre from this batch. They are original generated MIDI files, accompanied by a small provenance record; the full private review batch remains outside Git.

Note-level inspection identified useful listening review points:

- **Classical:** all six samples used piano. Density varied widely; one 124-note sample used E4 for 63 notes.
- **Romantic:** one sample had 51.86% duplicate exact four-event windows, with 528 of 541 notes lasting one beat.
- **Rock & Pop:** four samples were piano-only, one brass-only, and one drums-only. No sample contained guitar or bass; the drums-only sample had no pitched melody or harmony.
- **Expression:** all Classical and Rock & Pop samples encoded velocity 80 throughout the first 80% of the timeline. The runtime supplies an ending taper and quiet tail; these do not establish a learned harmonic cadence.

These descriptions are not aesthetic scores. Repeated percussion can be a useful groove, and encoded note patterns cannot establish listener preference. The positive listening feedback and these remaining review points can guide a later music-quality iteration.

## Reproduce the evaluation

From the repository root, install the application and CPU worker dependencies as described in the [README](../README.md), then install the browser dependency:

```bash
.venv/bin/python -m pip install -e ".[ui]"
.venv/bin/python -m playwright install chromium
```

Use the actual epoch-500 checkpoint from your local training run and a new output directory. The [benchmark helper](../scripts/evaluation/evaluate_app.py) starts dedicated loopback services on ports 5050 and 8050 and stops them on exit. It fixes CPU inference to two Torch threads and retains the controls listed above.

```bash
.venv/bin/python scripts/evaluation/evaluate_app.py \
  --checkpoint storage/training/balanced250_cpu_v1/last.pt \
  --expected-epoch 500 --trials 3 \
  --output-dir storage/evaluation/epoch500_review_v2

.venv/bin/python scripts/evaluation/analyze_midi.py \
  storage/evaluation/epoch500_review_v2/frontend/*.mid \
  storage/evaluation/epoch500_review_v2/backend/*.mid \
  --output storage/evaluation/epoch500_review_v2/quality_metrics.json
```

The [audio renderer](../scripts/evaluation/render_audio.py) creates optional listening previews using an installed FluidSynth shared library and General MIDI soundfont. The default soundfont path is `/usr/share/sounds/sf2/TimGM6mb.sf2`; pass `--soundfont` for another local installation. MP3 export also requires FFmpeg.

```bash
.venv/bin/python scripts/evaluation/render_audio.py \
  storage/evaluation/epoch500_review_v2/frontend/*.mid \
  --output-dir storage/evaluation/epoch500_review_v2/audio/frontend --mp3
```

Keep full review batches and their raw timings, seeds, MIDI, audio and metadata under ignored `storage/`. This tracked summary and the three curated `outputs/` examples preserve the published results. Weights, datasets and the full generated review batch are not included in a fresh checkout.
