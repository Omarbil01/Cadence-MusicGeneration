# Training the balanced MIDI Transformer

The current workflow uses **750 MIDI files: 250 Classical/Baroque, 250 Romantic,
and 250 Rock & Pop**. Train locally with the [CPU command](../scripts/train_balanced250_cpu.py)
or use the [balanced Colab notebook](../notebooks/training/train_midi_transformer_balanced250_colab.ipynb).
These share the dataset and checkpoint format, with different model sizes and run
directories. Starting the application never starts training.

Datasets, manifests, caches, bundles, checkpoints and bulk training samples stay
under ignored `storage/` paths or outside the repository. The three curated
[`outputs/` examples](../outputs/README.md) are the published sample set. Commit
the training notebook with outputs cleared.

## Environment

Use Python 3.11 or 3.12 and run commands from the repository root. For local CPU
training:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[worker]"
python -m pip check
python scripts/train_balanced250_cpu.py --help
```

An existing project environment with CPU PyTorch can be reused. Notebook-only
dependencies are listed in [requirements.txt](../notebooks/training/requirements.txt).
The Colab notebook preserves its runtime's CUDA PyTorch instead of installing
the CPU wheel above.

## Dataset selection

| Genre | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Classical/Baroque | 175 | 50 | 25 | 250 |
| Romantic | 175 | 50 | 25 | 250 |
| Rock & Pop | 175 | 50 | 25 | 250 |
| Total | 525 | 150 | 75 | 750 |

You need the source PDMX MIDI collection, its metadata CSV, and the original
approved manifest used to preserve existing composition-family holdouts. These
inputs are not included. Replace the paths below with your local sources:

```bash
export MIDI_ROOT=/path/to/pdmx-root

python scripts/build_pdmx_manifest.py \
  --pdmx-csv "$MIDI_ROOT/PDMX.csv" \
  --midi-root "$MIDI_ROOT" \
  --baseline-manifest /path/to/Experiment_9_Selected_MIDI_APPROVED.csv \
  --classical-scope baroque_classical \
  --output storage/training/pdmx_expanded/dataset_v3

python scripts/build_balanced_manifest.py \
  --curated-manifest storage/training/pdmx_expanded/dataset_v3/expanded.csv \
  --midi-root "$MIDI_ROOT" \
  --output storage/training/pdmx_balanced250/dataset
```

Both builders preserve existing destinations; choose a new output directory to
rebuild. If the balanced `selected.csv` already exists with its matching source
files, start at preparation below. Keep source versions, hashes and builder
reports with each experiment.

The PDMX builder records versioned attribution rules and source checks. Named
Baroque/Classical composers form the first label; named Romantic composers form
the second; rock/pop tags supply the third when attribution does not conflict.
Broad classical tags alone cannot distinguish historical periods. Uncertain
attributions go to a review queue; Beethoven stays there under the default
policy. Labels and recorded rights status remain provisional.

The balanced selector samples within the curated pool's existing splits, rotates
through composer/artist strata and keeps one usable file per recorded composition
family. It freshly checks hashes and encoding, replacing technical rejects within
the same genre/split. Recorded grouping cannot exclude undocumented near-duplicates.

## Prepare, benchmark and train on CPU

The CPU runner defaults to `storage/training/pdmx_balanced250/dataset/selected.csv`.
Its MIDI root defaults to the sibling `music_generation_pytorch_v1/` directory;
the examples pass `--midi-root` explicitly. Add `--manifest /path/to/selected.csv`
to use a different selection.

Choose a new run directory; the completed local experiment used
`storage/training/balanced250_cpu_v1`.

```bash
export TRAIN_RUN=storage/training/balanced250_cpu_v2

python scripts/train_balanced250_cpu.py \
  --midi-root "$MIDI_ROOT" --prepare-only --output "$TRAIN_RUN"

python scripts/train_balanced250_cpu.py \
  --midi-root "$MIDI_ROOT" --benchmark-only --benchmark-steps 3 \
  --torch-threads 4 --output "$TRAIN_RUN"

python scripts/train_balanced250_cpu.py \
  --midi-root "$MIDI_ROOT" --smoke --torch-threads 4 \
  --output storage/training/balanced250_cpu_smoke_v2

python scripts/train_balanced250_cpu.py \
  --midi-root "$MIDI_ROOT" --epochs 500 --patience 50 \
  --torch-threads 4 --output "$TRAIN_RUN"
```

Review the preparation report, measured epoch estimate and smoke result before
running the final command. Preparation and benchmarking write preflight reports
beside the run directory. The isolated smoke run uses a tiny model; its checkpoint
is marked as a test artifact and cannot serve the application.

The epoch limit is a maximum. Early stopping follows validation improvement with
patience 50 and `min_delta=0.0001`. The optional `--max-epoch-minutes NUMBER`
checks the benchmark estimate before training; actual timing can vary. Use a new
run directory for changes to data, model or training settings.

| Setting | CPU command defaults | Balanced Colab defaults |
|---|---:|---:|
| Context tokens | 256 | 512 |
| Model width | 128 | 192 |
| Layers / attention heads | 3 / 4 | 4 / 6 |
| Batch / gradient accumulation | 8 / 4 | 16 / 2 |
| Training windows per file, before genre balancing | 8 | 16 |
| Fixed validation windows per file, maximum | 8 | 16 |
| Training transposition | ±2 semitones | ±2 semitones |
| Progress buckets / relative distance bias | 16 / enabled | 16 / enabled |
| Precision | CPU float32 | CUDA BF16 or FP16 autocast |
| Epoch limit / early-stopping patience | 500 / 50 | 500 / 50 |

CPU thread count defaults to at most four; the completed local 500-epoch run
explicitly used ten. Benchmark thread counts on the intended machine instead of
assuming more threads are faster. Colab timings require an actual GPU measurement.

Preparation normalizes tempo while retaining score-beat phrase continuity, stores
windows in SQLite and gives true piece endings EOS targets. Training samples
include ending windows and transpose melodic pitches consistently in inputs and
labels. Percussion and validation/test data are unchanged. Validation uses a fixed
composition-covering subset; the full test split stays unused during training.

## Resume and checkpoint selection

Resume an interrupted CPU run with the same sources and manifest:

```bash
python scripts/train_balanced250_cpu.py \
  --midi-root "$MIDI_ROOT" --resume "$TRAIN_RUN/last.pt"
```

Resume restores the saved configuration, optimizer, random state and history.
The output defaults to that checkpoint's original directory. Do not change model
or training settings during resume; unfinished-epoch work may need repeating.
The CPU runner accepts this workflow's full CPU/float32 checkpoints.

Each run keeps `best.pt` (validation selection), `last.pt` (resume), `history.json`
and `run.json` together. In the completed local run, `last.pt` is epoch **500** and
the saved validation-selected `best.pt` is epoch **482**. The app evaluation used
the literal epoch-500 checkpoint; see [evaluation results](evaluation.md).
The best saved checkpoint can precede the final epoch, and the minimum-improvement
threshold can leave smaller later loss improvements unselected.

## Listening, held-out evaluation and export

Compare fixed seeds for every genre, using the same renderer and inference
settings across checkpoints. The existing helper generates one-minute samples:

```bash
python - <<'PY'
import os
from pathlib import Path
from musicgen.training.training import generate_samples

run = Path(os.environ["TRAIN_RUN"])
generate_samples(run / "best.pt", run / "samples_60s", seconds=60,
                 seeds=(7, 19), device="cpu")
PY
```

Use `last.pt` explicitly when reviewing a literal final epoch. MIDI validity,
next-token loss and descriptive repetition metrics do not establish musical
quality. Listen for melody, harmony, phrase development, genre identity and
expression. The renderer's repetition control and release taper are separate
inference policies; a quiet ending does not prove a learned harmonic cadence.

After model-selection choices are final, evaluate the held-out split once:

```bash
python - <<'PY'
import json
import os
from pathlib import Path
import torch
from musicgen.training.training import evaluate_test, prepare_dataset

torch.set_num_threads(4)
run = Path(os.environ["TRAIN_RUN"])
selected = run / "best.pt"
checkpoint = torch.load(selected, map_location="cpu", weights_only=True)
config = checkpoint["signature"]["training_config"]
prepared = prepare_dataset(
    Path("storage/training/pdmx_balanced250/dataset/selected.csv"),
    Path(os.environ["MIDI_ROOT"]),
    Path("storage/training/pdmx_balanced250/cache_cpu"),
    context=config["context"], min_per_genre=250, max_per_genre=250,
    storage="sqlite", normalize_tempo=True, progress_bins=config["progress_bins"],
)
print(json.dumps(evaluate_test(selected, prepared, device="cpu"), indent=2))
PY
```

Results are saved beside the selected checkpoint in `test_metrics.json`. Repeated
calls reuse matching results and reject a different checkpoint identity in the
same directory. Keep the test score out of model/configuration selection.

Export the reviewed selection to a new inference-only file:

```bash
python - <<'PY'
import os
from pathlib import Path
from musicgen.training.training import export_checkpoint

run = Path(os.environ["TRAIN_RUN"])
print(export_checkpoint(run / "best.pt", run / "export/best.pt"))
PY
```

The export validates compatibility, removes resume-only fields and refuses to
overwrite an existing destination. Preserve the current deployed checkpoint before
installing a different one. See [checkpoint setup](../models/README.md) and
[operations](operations.md) for application paths and worker restarts.

## Balanced Colab workflow

Build a fresh private source-data ZIP after changing the code or selection:

```bash
python scripts/build_colab_bundle.py \
  --manifest storage/training/pdmx_balanced250/dataset/selected.csv \
  --midi-root "$MIDI_ROOT" --min-per-genre 250 --max-per-genre 250 \
  --output storage/training/cadence_balanced250_colab_bundle.zip
```

The bundle contains the selected MIDI, manifest and current project source with
checksums. Use a new filename for a rebuild; the builder preserves existing files.

1. Put the ZIP and [balanced notebook](../notebooks/training/train_midi_transformer_balanced250_colab.ipynb)
   in `My Drive/cadence-training/`. Keep the ZIP compressed and select a fresh GPU runtime.
2. Run the CUDA check and explicitly mount Drive. The default input is that ZIP
   on Drive; browser upload is also supported. The notebook verifies its contents
   before installing the bundled source.
3. Review the exact 750-file split audit and prepared-window report. A technical
   reject stops preparation instead of silently reducing a genre to 249 files.
4. Run the GPU benchmark and smoke test. Then enable `RUN_FULL_TRAINING`; its
   default five-minute estimated epoch budget is a preflight guard, not a timing guarantee.
5. Review learning curves and fixed-seed samples. Freeze choices, then enable the
   notebook's full-test evaluation, export and download cells in that order.

The default persistent run is `My Drive/cadence-training/balanced250_structured_v1/`.
Dataset and SQLite caches stay on the Colab VM; completed checkpoints persist on
Drive. Resume uses the same bundle, saved configuration and run directory after a
disconnect. BF16 is selected on supporting GPUs; otherwise FP16 uses a saved
gradient scaler. Exported parameters remain float32. This larger GPU configuration
is a separate experiment from the completed compact CPU run.
