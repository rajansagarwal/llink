# LLINK

Implementation of paper [Languages are New Modalities: Cross-Lingual Alignment via Encoder Injection](https://example.com)

![LLINK Architecture](images/architecture.png)

#### Abstract

Instruction-tuned Large Language Models (LLMs) underperform on low‑resource, non‑Latin scripts due to tokenizer fragmentation and weak cross‑lingual coupling. We present LLINK (Large Language Lnjection for Non-English Knowledge), a compute-efficient language-as-modality method that conditions an instruction-tuned decoder without changing the tokenizer or retraining the decoder, operating in two key steps. In the first stage, we align sentence embeddings from a frozen multilingual encoder to the decoder’s hidden space at a reserved position via a lightweight contrastive projector; we follow by expanding this vector into a small set of soft slots and training minimal adapters so the frozen decoder consumes the signal. LLINK substantially improves bilingual retrieval and achieves 84\% preference over the base model and 64\% over direct finetuning in LLM-judged Q\&A evaluations. We further find that improvements can be attributed to reduced tokenization inflation and a stronger cross-lingual alignment, despite the model having residual weaknesses in numeric fidelity. Treating low-resource languages as a modality offers a practical path to stronger cross-lingual alignment in lightweight LLMs.

## Prerequisites

- Python 3.13 or newer for local tooling. (Modal functions build their own CUDA-enabled Python 3.10 image.)
- [uv](https://docs.astral.sh/uv/) or `pip` for dependency management.
- [Modal CLI](https://modal.com/docs/guide/cli) (`pip install modal`) with an authenticated account (`modal token set`).
- A Hugging Face access token stored in Modal as a secret named `hf-token`:
  ```bash
  echo "hf_xxx" | modal secret create hf-token --stdin
  ```
- A Modal volume named `khmer-bridge-vol` to persist checkpoints and trained artifacts:
  ```bash
  modal volume create khmer-bridge-vol
  ```
  Populate the volume with the required model checkpoints before running remote jobs; asset preparation is tracked separately.

## Local Environment

```bash
git clone <repo-url>
cd llink
uv venv  # or: python3.13 -m venv .venv
source .venv/bin/activate
uv sync  # or: pip install -e .
pip install modal  # ensures CLI + SDK available inside the venv
```

`uv sync` installs the base visualization utilities (`matplotlib`, `pandas`) used for log analysis and plots. Add any project-specific extras inside the virtual environment as needed.

## Modal Resources

The remote functions expect Modal to provide:

- GPU type `H100`.
- Volume `khmer-bridge-vol` mounted at `/vol` (checkpoints, adapters, and projector weights).
- Secret `hf-token` exposing the Hugging Face token as `HUGGINGFACE_HUB_TOKEN`.

Create or update these resources before launching jobs. Data curation and uploads are handled separately (TBD).

## Running

### Stage A: Projection Training

Launch the Stage A contrastive projector job on Modal:

```bash
modal run projection/train.py::train_projection_model
```

The job streams logs to the terminal. Checkpoints and metadata are written under `/vol/ckpts` in the attached Modal volume.

### Stage B: Inference Functions

Invoke the remote inference function with a string in the foreign language:

```bash
modal run inference/main.py::infer --kwargs '{"foreign_text": "សួស្តី", "task_type": "translate_to_english"}'
```

Optional flags:

- `--strict` (boolean) toggles conservative prompting.
- `--gate-boost <float>` scales the injected slot magnitude.

The function compares injected vs. ablated outputs and prints warnings when the injection has no measurable effect.

To inspect slot neighborhoods or token-level deltas during debugging, call the probing utility:

```bash
modal run inference/main.py::lexeme_probe --kwargs '{"foreign_text": "សួស្តី"}'
```

### Batch Diagnostics

Supply a list of foreign-language examples to the batch tester to measure pairwise cosine similarity and slot norms:

```bash
modal run inference/main.py::test_batch --kwargs '{"foreign_texts": ["ខ្ញុំស្រឡាញ់ភាសាខ្មែរ", "សួស្តី"]}'
```

Outputs include diversity statistics that help tune batching and normalization.

### Local Utilities

The repository also includes helper scripts that can be run locally once the virtual environment is active, for example:

```bash
python parse_logs.py  # summarize training runs into CSV/plots
```

Plotting and analysis notebooks reference artifacts written to the Modal volume. Mount or sync those assets locally before post-processing (details TBD).
