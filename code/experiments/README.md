### Work graph (source of truth)

This directory uses **`work_graph.json` as the sole source of truth** for experiments and their current status.

`vectur/code/run_experiments.py` + `vectur/code/experiment_manager.py`:

- read `work_graph.json`
- choose a runnable node whose dependencies are complete
- run the right action (`train`, `finetune`, `eval`)
- update the node status in `work_graph.json`

No per-stage config files are committed here.

### Remote persistence (Google Drive via rclone)

`run_experiments.py` supports persisting the authoritative `work_graph.json` and using remote lockfiles via:
- `--rclone-remote gdrive:` (or any rclone remote prefix)
- `--rclone-root "research/papers/vectur/code"` (folder under that remote)

When remote is enabled, `train.py` and `eval.py` also use the same remote settings to:
- upload checkpoints/metadata under `runs/**`
- download missing checkpoints on demand (helpful after preemption/restart)

### Status model (manual intervention required)

Each node has exactly one status:

- `not_started` (only this state can be claimed/run automatically)
- `in_progress`
- `completed`
- `failed`
- `interrupted`

If a node is in any state other than `not_started`, it will not be auto-claimed again; you must manually edit the status (or use the CLI override flags) to unblock.

### Node schema (high level)

`work_graph.json` contains:

- `version`: integer
- `updated_utc`: float
- `nodes`: object keyed by `node_id`

Each node contains:

- `id`: string (must match the key)
- `action`: `"train" | "finetune" | "eval" | "paper" | "noop"`
- `depends_on`: list of node ids
- `status`: one of the statuses above
- `spec`: action-specific payload

Action `spec` conventions:

- **train**: `{ "train_config": { ...TrainConfig fields... } }`
- **finetune**: `{ "experiment_config": { ... } }` (consumed by `fine_tune.py --config`)
- **eval**: `{ "eval_config": { ... } }` (consumed by `eval.py --config`)

### Common HF dataset IDs used by work items

- FineWeb-EDU: `HuggingFaceFW/fineweb-edu` (config `sample-10BT`, split `train`)
- CLRS-Text train: `tomg-group-umd/CLRS-Text-train` (split `train`)
- CLRS-Text held-out: `tomg-group-umd/CLRS-Text-test` (split `test`)
- CompGen: `ethanhallphd/compgen-compgen_v1_m` (splits `train`/`test`, if present)
- GSM8K: `openai/gsm8k` (config `main`, split `test`)
- ARC: `allenai/ai2_arc` (config `ARC-Challenge`, split `test`)
- HellaSwag: `Rowan/hellaswag` (split `validation`)
- WikiText-103: `Salesforce/wikitext` (config `wikitext-103-v1`, split `test`)

