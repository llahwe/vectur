### VecTur: Vector Turing Machines

This folder contains a research codebase and paper draft for **VecTur (Vector Turing Machines)**: a differentiable, Turing-machine-inspired transition system designed to be used as a **drop-in computational block inside a decoder-only LLM** (Llama-style macro architecture).

At a high level, VecTur aims to combine:

- **Iterative “System 2”-style computation**: explicit unrolled transition steps inside the forward pass.
- **Sparse, strictly local external memory access**: a continuous “head on a circle” (\(S^1\)) with local \(2k\)-sparse gather/scatter updates to avoid dense memory attention and reduce “memory blurring”.
- **Composable integration**: VecTur/VecSTur are implemented as modular blocks that can replace standard attention/MLP-style blocks in a fixed macro architecture.
- **Mechanistic interpretability hooks**: explicit state variables (finite control + tape + head index) make it easier to analyze learned algorithms.

VecSTur is a stochastic extension that consumes random tape symbols to target **randomized algorithms** as a computational resource.

### Repository layout

- **Paper**: `vectur/paper/`
  - Main LaTeX: `vectur/paper/paper.tex`
  - PDF output: `vectur/paper/paper.pdf`
- **Training / eval code**: `vectur/code/`
  - Training: `vectur/code/train.py`
  - Fine-tuning: `vectur/code/fine_tune.py`
  - Evaluation: `vectur/code/eval.py`
  - Blocks (architectures): `vectur/code/blocks/`
  - Experiment configs: `vectur/code/experiments/`
  - Multi-worker experiment DAG manager: `vectur/code/experiment_manager.py` + `vectur/code/run_experiments.py`
- **Synthetic dataset generator**: `vectur/datasets/compgen.py`
- **Vast helper**: `vectur/vast_setup.sh` (SSH + optional `rclone config` port-forward setup)

### What experiments are run (as described in the paper)

The paper defines three complementary protocols (run per block / architecture):

- **Protocol 1: language pretraining → downstream evaluation**
  - Train: FineWeb(-EDU)
  - Test: GSM8K, ARC, HellaSwag, WikiText-103
- **Protocol 2: algorithmic transfer between CLRS and CompGen**
  - (a) Train on CLRS-Text, test on CompGen under:
    - **Zero-shot** (no CompGen fine-tuning)
    - **Few-shot** (in-context demonstrations at test time)
    - **Fine-tune** (supervised adaptation on CompGen train)
  - (b) Train on CompGen, test on held-out CompGen
- **Protocol 3: in-domain CLRS generalization**
  - Train on CLRS-Text train, test on CLRS-Text held-out

The intended “blocks” compared include:

- Attention (Transformer)
- LSTM
- NTM/DNC-style differentiable memory baseline
- VecTur
- VecSTur

### Running training / fine-tuning / evaluation (single machine)

All three entrypoints support either a JSON config **or** explicit CLI flags (but **not both**):

- `vectur/code/train.py`: pretraining / training runs
- `vectur/code/fine_tune.py`: fine-tune a trained checkpoint (stage is explicit)
- `vectur/code/eval.py`: evaluation runs (can build HF token buffers on demand)

Outputs are written locally under:

- **Runs**: `runs/<run_name>/`
  - `runs/<run_name>/checkpoints/` (checkpoints)
  - `runs/<run_name>/metrics.jsonl` (training logs)
  - `runs/<run_name>/config.json` (resolved config)
  - `runs/<run_name>/token_buffer.pt` (+ `token_buffer_meta.json` when using HF buffers)
- **Eval caches**: `eval_cache/<experiment>/<test_name>/token_buffer.pt` (when eval builds HF buffers)

Configs live in `vectur/code/experiments/`. See `vectur/code/experiments/README.md` for the config conventions and dataset IDs used by those configs.

### Generating CompGen and uploading splits to Hugging Face (optional)

`vectur/datasets/compgen.py` can generate `.jsonl` examples and (optionally) push them to the Hugging Face Hub as a **datasets** repo split. It supports deterministic splitting during generation:

- **`--split-frac`**: train fraction (default `0.98`), writes `train/` and `test/` subfolders and pushes both splits.

Example (placeholders):

```bash
python3 -m vectur.datasets.compgen generate \
  --run-name compgen_v1 \
  --fmt jsonl \
  --dataset-size 1000000 \
  --hf-repo "<YOUR_HF_USERNAME>/<YOUR_DATASET_NAME>" \
  --hf-upload --hf-as-dataset \
  --split-frac 0.98
```

### Running the full paper experiment DAG (multi-worker; multiple Vast instances)

The experiment manager treats the paper’s work as a **directed acyclic graph (DAG)** of stages. Example stage chain per model:

- `dry_run_train → train → (dry_run_finetune → finetune)? → dry_run_eval(test) → eval(test) → paper bookkeeping`

The manager persists the DAG state as **human-readable JSON** and supports:

- **Automatic stage completion** (exit code 0)
- **Automatic failure** (non-zero exit code)
- **Manual overrides** (mark a stage completed/failed without running it)

#### Remote state + mutexes (to avoid duplicate work)

When running on multiple machines, you can store the authoritative work graph on Google Drive (or any rclone-supported remote) and use **remote lockfiles** as mutexes:

- Graph state: `work_graph.json`
- Global lock: `locks/graph.lock.json` (serializes graph updates)
- Per-stage locks: `locks/stages/<node>.lock.json` (prevents two workers from running the same stage)

Locks are JSON with an expiry (`expires_utc`) so stale locks can be removed or edited manually if needed.

#### Setting up rclone on Vast

Use `vectur/vast_setup.sh --rclone` to run `rclone config` on the Vast instance with the OAuth callback port forwarded to your local browser (the script prints instructions).

#### Running the scheduler on each worker

On each Vast instance, set **placeholders** (do not paste secrets into this repo):

```bash
export RCLONE_REMOTE="<YOUR_RCLONE_REMOTE_PREFIX>"   # e.g. "gdrive:"
export RCLONE_ROOT="<REMOTE_FOLDER_PATH>"            # e.g. "research/papers/vectur/code"
export OWNER_ID="$(hostname):$$"
```

Then run:

```bash
python3 vectur/code/run_experiments.py --refresh --list
python3 vectur/code/run_experiments.py --max-stages 999999
```

The CLI prints status lines like:

- `Running Stage <node_id> ...`
- `Success!`
- `Skipped (locked by another worker).`

#### Manual overrides (unblocking)

To manually override a node status:

```bash
python3 vectur/code/run_experiments.py --manual-complete "<node_id>" --manual-note "override"
python3 vectur/code/run_experiments.py --manual-fail "<node_id>" --manual-note "override"
```

Or (advanced) edit the remote `work_graph.json` / lockfiles directly via Google Drive to unblock a stuck run.

### Security / privacy notes (important)

- **Do not commit** credentials, access tokens, SSH keys, or `rclone.conf`.
- Treat `rclone config` and any cloud credentials as sensitive.
- Prefer passing remotes/paths via environment variables (`RCLONE_REMOTE`, `RCLONE_ROOT`) rather than hardcoding them into configs or docs.