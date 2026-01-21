# Vast.ai + Google Drive persistence (Vectur)

This repo supports **persisting**:
- `vectur/code/experiments/work_graph.json` (the work-graph state + remote locks)
- `vectur/code/runs/**` (checkpoints + run metadata)

…to **Google Drive** using **rclone**, so you can survive Vast preemption / restarts.

## 1) One-time rclone setup on the Vast instance (personal Google Drive)

### Install rclone (on the instance)

```bash
apt-get update -y
apt-get install -y rclone
```

### Authenticate rclone on the instance (OAuth via your local browser)

From your laptop, run:

```bash
bash vectur/vast_setup.sh --rclone
```

That script SSHes with a port-forward so the URL rclone prints (with `127.0.0.1:53682`) can be opened in your **local** browser to complete auth for the **remote** instance.

In `rclone config`:
- Create a new remote named `gdrive`
- Type: `drive`
- Scope: full drive access (recommended)

Verify on the instance:

```bash
rclone lsd gdrive:
```

## 2) Pick where on Drive the state should live

Choose a Drive folder root (example):
- `research/papers/vectur/code`

Under that root, this code will write:
- `experiments/work_graph.json`
- `locks/**` (remote mutex/lock files)
- `runs/**` (checkpoints, config, metrics)

## 3) Run the work graph with persistence enabled

On the Vast instance, `cd` into `vectur/code` and run:

```bash
python3 run_experiments.py \
  --rclone-remote gdrive: \
  --rclone-root "research/papers/vectur/code" \
  --max-stages 1
```

### Optional: one-command launch helper

From your laptop, you can also start the scheduler detached via:

```bash
bash vectur/vash_setup.sh --launch --remote-repo-dir /workspace/vectur
```

If `tmux` is installed on the instance, this starts a session named `vectur_sched`.

Notes:
- `run_experiments.py` uses rclone for **remote locks** + keeping `work_graph.json` authoritative on Drive.
- `train.py` and `eval.py` automatically:
  - **push** checkpoints + metadata to the same Drive root (under `runs/**`)
  - **pull** missing checkpoints from Drive when needed (e.g. after a restart)

## 4) Common Vast workflow (what you described)

Typical sequence:
- Launch instance using your normal Vast template.
- Sync your repo/code onto the instance (your “git sync” step).
- Run `bash vectur/vast_setup.sh --rclone` once per *fresh* instance (or restore rclone config).
- Start the work graph using the command above.

## 5) (Optional) Fully non-interactive auth (service account)

If you want “no browser auth ever”:
- Create a Google Cloud service account + JSON key
- Share a Drive folder with the service account email
- Set rclone env vars in your Vast template, e.g.:

```bash
RCLONE_CONFIG_GDRIVE_TYPE=drive
RCLONE_CONFIG_GDRIVE_SCOPE=drive
RCLONE_CONFIG_GDRIVE_SERVICE_ACCOUNT_FILE=/workspace/secrets/gdrive-sa.json
```

Then `--rclone-remote gdrive:` works without `rclone config`.

