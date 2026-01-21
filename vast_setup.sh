#!/bin/bash
set -euo pipefail

###############################################################################
# vast_setup.sh
#
# What this does:
# - Find the only Vast instance (or prompt if there are multiple)
# - Ensure SSH access (optionally attaches your SSH pubkey to the instance)
# - By default: SSH into the instance (no rclone setup).
# - Optional: run `rclone config` on the remote with a local port-forward so the
#   rclone OAuth URL works in your LOCAL browser.
#
# Why port-forward 53682?
# - rclone's Google Drive OAuth flow uses a local callback server on:
#     http://127.0.0.1:53682/...
# - If rclone runs on the REMOTE instance, it listens on REMOTE localhost:53682.
# - `ssh -L 53682:127.0.0.1:53682 ...` forwards LOCAL 53682 -> REMOTE 53682,
#   so opening the URL in your local browser completes the remote OAuth flow.
###############################################################################

# Optional: load env vars from .env if present (don’t fail if missing).
# NOTE: This script does not require .env for the port-forward OAuth flow.
if [[ -f ".env" ]]; then
  # shellcheck disable=SC1091
  source ".env"
fi

echo "[vast_setup] starting..."

###############################################################################
# Flags / options
#
# Default behavior is intentionally minimal: pick instance and SSH into it.
# To run the rclone OAuth setup, pass: --rclone
# To sync this git repo onto the instance, pass: --sync-git
###############################################################################

RUN_RCLONE_SETUP=false
SYNC_GIT=false
LAUNCH_SCHEDULER=false
LAUNCH_MAX_STAGES=999999
REMOTE_REPO_DIR_OVERRIDE="workspace/vectur"

usage() {
  cat <<'EOF'
Usage:
  bash vast_setup.sh [--rclone] [--sync-git] [--launch] [--max-stages N] [--remote-repo-dir DIR] [--help]

Options:
  --rclone     Run `rclone config` on the remote with port-forwarding (53682).
               This is needed if you want to authenticate rclone via your local browser.
  --sync-git   Sync this repo onto the Vast instance:
               - Detect repo URL from `git remote -v` (asks you to confirm)
               - If not already present on the instance, `git clone` into /workspace/<repo>
               - Otherwise `git pull` to update it
  --launch     Start the experiment scheduler on the instance (detached).
               Uses tmux if available (session: vectur_sched), otherwise falls back to nohup.
               If you don't pass --sync-git, you should also pass --remote-repo-dir.
  --max-stages Max number of runnable stages to execute (default: 999999).
  --remote-repo-dir
               Absolute path to the repo root on the instance (only used when --sync-git is not set).
               Example: /workspace/vectur
  --help       Show this help.

Defaults:
  - Without flags, the script just selects an instance and SSHes into it.
EOF
}

ask_confirm_default_yes() {
  # Usage: ask_confirm_default_yes "Prompt text"
  # Returns 0 for yes, 1 for no.
  local prompt="$1"
  local ans=""
  while true; do
    read -r -p "${prompt} [Y/n] " ans
    case "${ans}" in
      ""|y|Y|yes|YES) return 0 ;;
      n|N|no|NO) return 1 ;;
      *) echo "Please answer y or n." ;;
    esac
  done
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rclone)
      RUN_RCLONE_SETUP=true
      shift
      ;;
    --sync-git)
      SYNC_GIT=true
      shift
      ;;
    --launch)
      LAUNCH_SCHEDULER=true
      shift
      ;;
    --max-stages)
      LAUNCH_MAX_STAGES="${2:-}"
      shift 2
      ;;
    --remote-repo-dir)
      REMOTE_REPO_DIR_OVERRIDE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[vast_setup] error: unknown argument: $1"
      echo
      usage
      exit 2
      ;;
  esac
done

###############################################################################
# Optional git sync configuration (local side)
###############################################################################

REPO_URL=""
REPO_NAME=""
LOCAL_BRANCH=""
REMOTE_REPO_BASE_DIR="${REMOTE_REPO_BASE_DIR:-/workspace}"
# Optional override when not using --sync-git.
REMOTE_REPO_DIR_OVERRIDE="${REMOTE_REPO_DIR_OVERRIDE:-${REMOTE_REPO_DIR:-}}"

# Convenience: support the older env var names from README drafts.
LOCAL_RCLONE_REMOTE="${RCLONE_REMOTE:-${VECTUR_RCLONE_REMOTE:-}}"
LOCAL_RCLONE_ROOT="${RCLONE_ROOT:-${VECTUR_RCLONE_ROOT:-}}"
LOCAL_OWNER_ID="${OWNER_ID:-${VECTUR_OWNER_ID:-}}"

if [[ "${SYNC_GIT}" == "true" ]]; then
  if ! command -v git >/dev/null 2>&1; then
    echo "[vast_setup] error: --sync-git requested but git is not installed locally"
    exit 1
  fi

  # Ensure we're in a git repo and run git commands from the repo root.
  REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
  if [[ -z "${REPO_ROOT}" ]]; then
    echo "[vast_setup] error: --sync-git requested but current directory is not a git repo"
    exit 1
  fi
  cd "${REPO_ROOT}"

  # Detect a clone URL from `git remote -v` (prefer origin fetch).
  DETECTED_URL="$(git remote -v | awk '$1=="origin" && $3=="(fetch)" {print $2; exit}')"
  if [[ -z "${DETECTED_URL}" ]]; then
    DETECTED_URL="$(git remote -v | awk '$3=="(fetch)" {print $2; exit}')"
  fi

  if [[ -n "${DETECTED_URL}" ]]; then
    echo "[vast_setup] detected repo URL from git remote -v:"
    echo "  ${DETECTED_URL}"
    if ask_confirm_default_yes "Use this repo URL for syncing to the instance?"; then
      REPO_URL="${DETECTED_URL}"
    else
      read -r -p "Paste a repo URL to clone on the instance: " REPO_URL
    fi
  else
    echo "[vast_setup] could not detect a repo URL from git remote -v"
    read -r -p "Paste a repo URL to clone on the instance: " REPO_URL
  fi

  if [[ -z "${REPO_URL}" ]]; then
    echo "[vast_setup] error: repo URL is empty; cannot sync"
    exit 1
  fi

  # Choose a folder name on the remote instance based on the repo URL.
  # Works for URLs like:
  # - git@github.com:user/repo.git
  # - https://github.com/user/repo.git
  # - https://github.com/user/repo
  REPO_NAME="${REPO_URL##*/}"
  REPO_NAME="${REPO_NAME%.git}"

  # Syncing the current local branch is usually what you want.
  LOCAL_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  if [[ -z "${LOCAL_BRANCH}" || "${LOCAL_BRANCH}" == "HEAD" ]]; then
    LOCAL_BRANCH=""
  fi
fi

echo "[vast_setup] discovering instances..."

# Get a clean list of instance IDs. `-q` prints only numeric ids (one per line).
# This is robust across Vast CLI output formatting.
INSTANCE_IDS=()
while IFS= read -r line; do
  [[ -n "${line}" ]] && INSTANCE_IDS+=("${line}")
done < <(vastai show instances -q)

if [[ "${#INSTANCE_IDS[@]}" -eq 0 ]]; then
  echo "[vast_setup] error: no Vast instances found"
  exit 1
fi

# If there is exactly one instance, pick it automatically.
# Otherwise, show the full table and prompt.
INSTANCE_ID=""
if [[ "${#INSTANCE_IDS[@]}" -eq 1 ]]; then
  INSTANCE_ID="${INSTANCE_IDS[0]}"
else
  echo "[vast_setup] multiple instances found:"
  vastai show instances
  echo
  echo "[vast_setup] choose an instance id to use:"
  select choice in "${INSTANCE_IDS[@]}" "Enter manually"; do
    if [[ "${choice}" == "Enter manually" ]]; then
      read -r -p "Instance id: " INSTANCE_ID
      break
    fi
    if [[ -n "${choice}" ]]; then
      INSTANCE_ID="${choice}"
      break
    fi
  done
fi

echo "[vast_setup] instance id: $INSTANCE_ID"

# SSH key handling:
# - If your instance was created with `--ssh`, you may already have access.
# - If not, you may need to attach your public key to the instance.
# - We try to attach ~/.ssh/vast_ed25519.pub if it exists; otherwise we proceed.
VAST_SSH_KEY_PATH="${VAST_SSH_KEY_PATH:-$HOME/.ssh/vast_ed25519}"
VAST_SSH_PUBKEY_PATH="${VAST_SSH_PUBKEY_PATH:-${VAST_SSH_KEY_PATH}.pub}"

if [[ -f "${VAST_SSH_PUBKEY_PATH}" ]]; then
  echo "[vast_setup] attaching SSH public key to instance (best-effort): ${VAST_SSH_PUBKEY_PATH}"
  PUBKEY_CONTENT="$(<"${VAST_SSH_PUBKEY_PATH}")"
  if ! vastai attach "${INSTANCE_ID}" "${PUBKEY_CONTENT}"; then
    echo "[vast_setup] note: could not attach key (it may already be attached). Continuing..."
  fi
else
  echo "[vast_setup] note: public key not found at ${VAST_SSH_PUBKEY_PATH}; skipping vastai attach"
fi

# rclone OAuth callback port (local forwarded -> remote).
RCLONE_AUTH_PORT="${RCLONE_AUTH_PORT:-53682}"

# Get SSH connection info from Vast.
# Depending on Vast CLI version, this may be:
# - "ssh://root@HOST:PORT"
# - "ssh -p PORT root@HOST"
SSH_URL="$(vastai ssh-url "${INSTANCE_ID}")"
echo "[vast_setup] ssh-url: ${SSH_URL}"

# Use our SSH identity if present; otherwise let ssh use your defaults.
SSH_IDENTITY_ARGS=()
if [[ -f "${VAST_SSH_KEY_PATH}" ]]; then
  SSH_IDENTITY_ARGS=(-i "${VAST_SSH_KEY_PATH}")
fi

# Remote command:
# - Default: open a login shell.
# - With --sync-git: ensure the repo exists and is updated on the instance.
# - With --rclone: run `rclone config` interactively (typically after git sync).
REMOTE_CMD=""
REMOTE_CMD+=$'set -euo pipefail\n'

# Propagate common remote-state env vars to the instance (best-effort).
if [[ -n "${LOCAL_RCLONE_REMOTE}" ]]; then
  REMOTE_CMD+="export RCLONE_REMOTE=$(printf '%q' "${LOCAL_RCLONE_REMOTE}")"$'\n'
fi
if [[ -n "${LOCAL_RCLONE_ROOT}" ]]; then
  REMOTE_CMD+="export RCLONE_ROOT=$(printf '%q' "${LOCAL_RCLONE_ROOT}")"$'\n'
fi
if [[ -n "${LOCAL_OWNER_ID}" ]]; then
  REMOTE_CMD+="export OWNER_ID=$(printf '%q' "${LOCAL_OWNER_ID}")"$'\n'
fi

if [[ "${SYNC_GIT}" == "true" ]]; then
  # Pass local values safely into the remote script.
  REMOTE_CMD+="REMOTE_REPO_BASE_DIR=$(printf '%q' "${REMOTE_REPO_BASE_DIR}")"$'\n'
  REMOTE_CMD+="REPO_URL=$(printf '%q' "${REPO_URL}")"$'\n'
  REMOTE_CMD+="REPO_NAME=$(printf '%q' "${REPO_NAME}")"$'\n'
  if [[ -n "${LOCAL_BRANCH}" ]]; then
    REMOTE_CMD+="LOCAL_BRANCH=$(printf '%q' "${LOCAL_BRANCH}")"$'\n'
  else
    REMOTE_CMD+=$'LOCAL_BRANCH=""\n'
  fi

  REMOTE_CMD+=$'echo\n'
  REMOTE_CMD+=$'echo "[vast_setup][remote] Git sync enabled."\n'
  REMOTE_CMD+=$'echo "[vast_setup][remote] Repo URL: ${REPO_URL}"\n'
  REMOTE_CMD+=$'echo "[vast_setup][remote] Repo dir: ${REMOTE_REPO_BASE_DIR}/${REPO_NAME}"\n'
  REMOTE_CMD+=$'echo\n'
  REMOTE_CMD+=$'if ! command -v git >/dev/null 2>&1; then\n'
  REMOTE_CMD+=$'  echo "[vast_setup][remote] error: git not found on instance."\n'
  REMOTE_CMD+=$'  exit 1\n'
  REMOTE_CMD+=$'fi\n'
  REMOTE_CMD+=$'REPO_DIR="${REMOTE_REPO_BASE_DIR}/${REPO_NAME}"\n'
  REMOTE_CMD+=$'mkdir -p "${REMOTE_REPO_BASE_DIR}"\n'
  REMOTE_CMD+=$'if [[ -d "${REPO_DIR}" && ! -d "${REPO_DIR}/.git" ]]; then\n'
  REMOTE_CMD+=$'  echo "[vast_setup][remote] error: ${REPO_DIR} exists but is not a git repo (no .git directory)."\n'
  REMOTE_CMD+=$'  echo "[vast_setup][remote]        Move it aside or delete it, then re-run with --sync-git."\n'
  REMOTE_CMD+=$'  exit 1\n'
  REMOTE_CMD+=$'fi\n'
  REMOTE_CMD+=$'if [[ -d "${REPO_DIR}/.git" ]]; then\n'
  REMOTE_CMD+=$'  echo "[vast_setup][remote] Repo already exists; pulling latest..."\n'
  REMOTE_CMD+=$'  git -C "${REPO_DIR}" fetch --all --prune\n'
  REMOTE_CMD+=$'  if [[ -n "${LOCAL_BRANCH}" ]]; then\n'
  REMOTE_CMD+=$'    git -C "${REPO_DIR}" checkout "${LOCAL_BRANCH}" || true\n'
  REMOTE_CMD+=$'  fi\n'
  REMOTE_CMD+=$'  git -C "${REPO_DIR}" pull --ff-only || git -C "${REPO_DIR}" pull\n'
  REMOTE_CMD+=$'else\n'
  REMOTE_CMD+=$'  echo "[vast_setup][remote] Cloning repo..."\n'
  REMOTE_CMD+=$'  git clone "${REPO_URL}" "${REPO_DIR}"\n'
  REMOTE_CMD+=$'  if [[ -n "${LOCAL_BRANCH}" ]]; then\n'
  REMOTE_CMD+=$'    git -C "${REPO_DIR}" checkout "${LOCAL_BRANCH}" || true\n'
  REMOTE_CMD+=$'  fi\n'
  REMOTE_CMD+=$'  # Clone already fetches the latest commit, but we also pull to satisfy "clone then pull".\n'
  REMOTE_CMD+=$'  git -C "${REPO_DIR}" pull --ff-only || true\n'
  REMOTE_CMD+=$'fi\n'
  REMOTE_CMD+=$'cd "${REPO_DIR}"\n'
fi

if [[ "${RUN_RCLONE_SETUP}" == "true" ]]; then
  REMOTE_CMD+=$'echo\n'
  REMOTE_CMD+=$'echo "[vast_setup][remote] Port-forward is active."\n'
  REMOTE_CMD+=$'echo "[vast_setup][remote] Running: rclone config"\n'
  REMOTE_CMD+=$'echo "[vast_setup][remote] When rclone prints a URL with 127.0.0.1:53682, open it in your LOCAL browser."\n'
  REMOTE_CMD+=$'echo\n'
  REMOTE_CMD+=$'if ! command -v rclone >/dev/null 2>&1; then\n'
  REMOTE_CMD+=$'  echo "[vast_setup][remote] rclone not found; attempting install (best-effort)..."\n'
  REMOTE_CMD+=$'  if command -v apt-get >/dev/null 2>&1; then\n'
  REMOTE_CMD+=$'    apt-get update -y && apt-get install -y rclone\n'
  REMOTE_CMD+=$'  elif command -v yum >/dev/null 2>&1; then\n'
  REMOTE_CMD+=$'    yum install -y rclone || true\n'
  REMOTE_CMD+=$'  fi\n'
  REMOTE_CMD+=$'fi\n'
  REMOTE_CMD+=$'if ! command -v rclone >/dev/null 2>&1; then\n'
  REMOTE_CMD+=$'  echo "[vast_setup][remote] error: rclone still not found. Install it manually and re-run with --rclone."\n'
  REMOTE_CMD+=$'  exit 1\n'
  REMOTE_CMD+=$'fi\n'
  REMOTE_CMD+=$'rclone config\n'
  REMOTE_CMD+=$'echo\n'
  REMOTE_CMD+=$'echo "[vast_setup][remote] rclone config finished."\n'
fi

if [[ "${LAUNCH_SCHEDULER}" == "true" ]]; then
  REMOTE_CMD+=$'echo\n'
  REMOTE_CMD+=$'echo "[vast_setup][remote] Launching scheduler..."\n'
  REMOTE_CMD+="LAUNCH_MAX_STAGES=$(printf '%q' "${LAUNCH_MAX_STAGES}")"$'\n'

  # If we didn't sync via git, we still need a repo dir to run from.
  if [[ "${SYNC_GIT}" != "true" ]]; then
    REMOTE_CMD+="REMOTE_REPO_DIR_OVERRIDE=$(printf '%q' "${REMOTE_REPO_DIR_OVERRIDE}")"$'\n'
    REMOTE_CMD+=$'if [[ -z "${REMOTE_REPO_DIR_OVERRIDE:-}" ]]; then\n'
    REMOTE_CMD+=$'  if [[ -d "/workspace/vectur" ]]; then\n'
    REMOTE_CMD+=$'    REMOTE_REPO_DIR_OVERRIDE="/workspace/vectur"\n'
    REMOTE_CMD+=$'  fi\n'
    REMOTE_CMD+=$'fi\n'
    REMOTE_CMD+=$'if [[ -z "${REMOTE_REPO_DIR_OVERRIDE:-}" ]]; then\n'
    REMOTE_CMD+=$'  echo "[vast_setup][remote] error: --launch requested but repo dir is unknown."\n'
    REMOTE_CMD+=$'  echo "[vast_setup][remote]        Re-run with: --remote-repo-dir /workspace/<your_repo_dir>  (or use --sync-git)"\n'
    REMOTE_CMD+=$'  exit 1\n'
    REMOTE_CMD+=$'fi\n'
    REMOTE_CMD+=$'REPO_DIR="${REMOTE_REPO_DIR_OVERRIDE}"\n'
    REMOTE_CMD+=$'cd "${REPO_DIR}"\n'
  fi

  REMOTE_CMD+=$'if [[ ! -f "code/run_experiments.py" ]]; then\n'
  REMOTE_CMD+=$'  echo "[vast_setup][remote] error: expected code/run_experiments.py under ${REPO_DIR}."\n'
  REMOTE_CMD+=$'  echo "[vast_setup][remote]        Are you in the repo root? (Try --remote-repo-dir /workspace/vectur)"\n'
  REMOTE_CMD+=$'  exit 1\n'
  REMOTE_CMD+=$'fi\n'
  REMOTE_CMD+=$'mkdir -p "code/.experiment_manager"\n'
  REMOTE_CMD+=$'SCHED_LOG="${REPO_DIR}/code/.experiment_manager/scheduler.log"\n'
  REMOTE_CMD+=$'PYBIN="python3"\n'
  REMOTE_CMD+=$'if ! command -v "${PYBIN}" >/dev/null 2>&1; then PYBIN="python"; fi\n'
  REMOTE_CMD+=$'if ! command -v "${PYBIN}" >/dev/null 2>&1; then\n'
  REMOTE_CMD+=$'  echo "[vast_setup][remote] error: python not found on instance."\n'
  REMOTE_CMD+=$'  exit 1\n'
  REMOTE_CMD+=$'fi\n'

  REMOTE_CMD+=$'if command -v tmux >/dev/null 2>&1; then\n'
  REMOTE_CMD+=$'  if tmux has-session -t vectur_sched 2>/dev/null; then\n'
  REMOTE_CMD+=$'    echo "[vast_setup][remote] Scheduler already running (tmux session: vectur_sched)."\n'
  REMOTE_CMD+=$'    echo "[vast_setup][remote] Attach with: tmux attach -t vectur_sched"\n'
  REMOTE_CMD+=$'  else\n'
  REMOTE_CMD+=$'    tmux new-session -d -s vectur_sched "cd ${REPO_DIR} && ${PYBIN} code/run_experiments.py --refresh --max-stages ${LAUNCH_MAX_STAGES} >> ${SCHED_LOG} 2>&1"\n'
  REMOTE_CMD+=$'    echo "[vast_setup][remote] Started scheduler in tmux session: vectur_sched"\n'
  REMOTE_CMD+=$'    echo "[vast_setup][remote] Log: ${SCHED_LOG}"\n'
  REMOTE_CMD+=$'  fi\n'
  REMOTE_CMD+=$'else\n'
  REMOTE_CMD+=$'  nohup bash -lc "cd ${REPO_DIR} && ${PYBIN} code/run_experiments.py --refresh --max-stages ${LAUNCH_MAX_STAGES} >> ${SCHED_LOG} 2>&1" >/dev/null 2>&1 &\n'
  REMOTE_CMD+=$'  echo "[vast_setup][remote] Started scheduler with nohup."\n'
  REMOTE_CMD+=$'  echo "[vast_setup][remote] Log: ${SCHED_LOG}"\n'
  REMOTE_CMD+=$'fi\n'
fi

REMOTE_CMD+=$'echo\n'
REMOTE_CMD+=$'echo "[vast_setup][remote] Starting interactive shell (exit to disconnect)."\n'
REMOTE_CMD+=$'exec bash -l\n'

# Build an ssh invocation robustly:
# - Prefer parsing known formats (ssh://... and "ssh -p ... user@host")
# - Fall back to best-effort execution of the returned ssh args.

# Case 1: ssh://root@HOST:PORT
if [[ "${SSH_URL}" =~ ^ssh://([^@]+)@([^:]+):([0-9]+) ]]; then
  SSH_USER="${BASH_REMATCH[1]}"
  SSH_HOST="${BASH_REMATCH[2]}"
  SSH_PORT="${BASH_REMATCH[3]}"

  if [[ "${RUN_RCLONE_SETUP}" == "true" ]]; then
    echo "[vast_setup] connecting to ${SSH_USER}@${SSH_HOST}:${SSH_PORT} with port-forward ${RCLONE_AUTH_PORT} (rclone setup enabled)..."
    ssh \
      "${SSH_IDENTITY_ARGS[@]}" \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -L "${RCLONE_AUTH_PORT}:127.0.0.1:${RCLONE_AUTH_PORT}" \
      -p "${SSH_PORT}" \
      "${SSH_USER}@${SSH_HOST}" \
      -t "bash -lc $(printf '%q' "${REMOTE_CMD}")"
  else
    echo "[vast_setup] connecting to ${SSH_USER}@${SSH_HOST}:${SSH_PORT} (no rclone setup; run with --rclone to enable)..."
    ssh \
      "${SSH_IDENTITY_ARGS[@]}" \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -p "${SSH_PORT}" \
      "${SSH_USER}@${SSH_HOST}" \
      -t "bash -lc $(printf '%q' "${REMOTE_CMD}")"
  fi

  echo "[vast_setup] done."
  exit 0
fi

# Case 2: "ssh -p PORT user@host" (or similar). Try to parse port + destination.
if [[ "${SSH_URL}" =~ -p[[:space:]]*([0-9]+)[[:space:]]+([^[:space:]]+) ]]; then
  SSH_PORT="${BASH_REMATCH[1]}"
  SSH_DEST="${BASH_REMATCH[2]}"

  if [[ "${RUN_RCLONE_SETUP}" == "true" ]]; then
    echo "[vast_setup] connecting to ${SSH_DEST} (port ${SSH_PORT}) with port-forward ${RCLONE_AUTH_PORT} (rclone setup enabled)..."
    ssh \
      "${SSH_IDENTITY_ARGS[@]}" \
      -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -L "${RCLONE_AUTH_PORT}:127.0.0.1:${RCLONE_AUTH_PORT}" \
      -p "${SSH_PORT}" \
      "${SSH_DEST}" \
      -t "bash -lc $(printf '%q' "${REMOTE_CMD}")"
  else
    echo "[vast_setup] connecting to ${SSH_DEST} (port ${SSH_PORT}) (no rclone setup; run with --rclone to enable)..."
    ssh \
      "${SSH_IDENTITY_ARGS[@]}" \
      -o ServerAliveInterval=30 \
      -o ServerAliveCountMax=3 \
      -p "${SSH_PORT}" \
      "${SSH_DEST}" \
      -t "bash -lc $(printf '%q' "${REMOTE_CMD}")"
  fi

  echo "[vast_setup] done."
  exit 0
fi

# Fallback: treat ssh-url output as ssh arguments (best-effort).
# NOTE: This uses `eval` to preserve argument splitting; ssh-url is trusted output
# from the Vast CLI. If you ever paste arbitrary text into SSH_URL, this is unsafe.
echo "[vast_setup] warning: could not parse ssh-url format; falling back to best-effort execution"
SSH_ARGS="${SSH_URL}"
SSH_ARGS="${SSH_ARGS#ssh }"

if [[ "${RUN_RCLONE_SETUP}" == "true" ]]; then
  eval ssh \
    "${SSH_IDENTITY_ARGS[@]}" \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -L "${RCLONE_AUTH_PORT}:127.0.0.1:${RCLONE_AUTH_PORT}" \
    ${SSH_ARGS} \
    -t "bash -lc $(printf '%q' "${REMOTE_CMD}")"
else
  eval ssh \
    "${SSH_IDENTITY_ARGS[@]}" \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    ${SSH_ARGS} \
    -t "bash -lc $(printf '%q' "${REMOTE_CMD}")"
fi

echo "[vast_setup] done."