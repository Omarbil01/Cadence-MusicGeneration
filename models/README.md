# Model checkpoints

Trained weights are not included in the repository. Place a compatible checkpoint at `models/best.pt`, or set `MUSICGEN_CHECKPOINT_PATH` to its location in `.env`. Checkpoint files are excluded from Git.

The application starts without weights and reports generation as unavailable. After adding or replacing a checkpoint, restart the local API or the Compose worker:

```bash
docker compose restart worker
```

Compose mounts this directory read-only in the worker. See the [main README](../README.md) for dependency installation and local startup.

Choose weights from the intended training run: filenames such as `best.pt` do not identify an epoch by themselves. Generated metadata records the loaded checkpoint's epoch and SHA-256. The current balanced CPU run's final checkpoint is epoch 500; its validation-selected checkpoint is epoch 482. See [evaluation results](../docs/evaluation.md) and [training/export instructions](../docs/training.md).

## Compatibility contract

The runtime supports the baseline `MusicTransformer` and its optional progress-conditioned, relative-attention configuration, with the same event vocabulary, genre order, and checkpoint schema. Model dimensions and optional features are recorded in `model.config`; a different model class, tokenizer, or genre mapping requires a runtime update.

Export a dictionary with `torch.save`. Use tensors and plain Python containers compatible with `torch.load(..., weights_only=True)`.

| Field | Required content |
|---|---|
| `model` | A state dictionary that loads strictly into the runtime `MusicTransformer`. |
| `epoch` | The training epoch, recorded in inference metadata. |
| `signature.vocab_sha256` | The runtime tokenizer's `VOCAB_SHA`. |
| `signature.dataset_sha256` | The dataset digest, matching `dataset_metadata.dataset_sha256`. |
| `signature.model_config` | `context`, `d_model`, `layers`, `heads`, `dropout`, and enabled optional `progress_bins`/`relative_attention`. Use `model.config`. |
| `signature.training_config` | A dictionary of training settings; `smoke` must be absent or false. |
| `dataset_metadata.dataset_kind` | `EXP9_APPROVED_MIDI` for the baseline or `PDMX_CURATED_MIDI` for the expanded selection. Neither identifier establishes dataset rights or model quality. |
| `dataset_metadata.dataset_sha256` | The same dataset digest as the signature. |
| `dataset_metadata.program_counts_train` | A nonempty program-count mapping for each supported genre. |
| `dataset_metadata.drum_pitch_counts_train` | A percussion-pitch-count mapping for each genre; an individual mapping may be empty. |
| `dataset_metadata.meter_counts_train` | A nonempty time-signature-count mapping for each genre. |

The genre keys are `classical`, `romantic`, and `rock_pop`, in that order for model conditioning. Count mappings come from the training split: use integer-like string keys for programs (`0`–`127`, with `128` representing percussion) and drum pitches (`0`–`127`), and supported meter strings such as `4/4`. Counts should be positive integers. Keep the tokenizer's event order and quantization unchanged, and use a context of at least 16 tokens for inference framing.

The export structure is:

```python
import torch

from musicgen.generation.runtime.tokenizer import VOCAB_SHA

torch.save(
    {
        "model": model.state_dict(),
        "epoch": epoch,
        "signature": {
            "vocab_sha256": VOCAB_SHA,
            "dataset_sha256": dataset_sha256,
            "model_config": model.config,
            "training_config": training_config,
        },
        "dataset_metadata": dataset_metadata,
    },
    "models/best.pt",
)
```

Here, `model`, `epoch`, the digest, training settings, and dataset metadata must come from the training run. The snippet defines the serialization boundary; it does not train a model.

The [checkpoint loader](../src/musicgen/generation/generator.py), [model](../src/musicgen/generation/runtime/model.py), and [tokenizer](../src/musicgen/generation/runtime/tokenizer.py) define compatibility. The [test fixture](../tests/conftest.py) demonstrates the schema with a tiny untrained checkpoint used only for integration tests. Passing compatibility checks does not demonstrate musical quality.

See [the training guide](../docs/training.md) for data preparation, CPU and Colab training, checkpoint selection, and export.
