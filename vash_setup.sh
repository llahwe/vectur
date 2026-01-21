#!/bin/bash
set -euo pipefail

# Backward-compatible wrapper for a common typo / alias.
# Prefer: bash vectur/vast_setup.sh ...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 ]]; then
  # Make the "just run it" path do something useful.
  # If your repo isn't at /workspace/vectur on the instance, re-run with:
  #   bash vectur/vash_setup.sh --remote-repo-dir /workspace/<your_repo> --launch
  exec bash "${SCRIPT_DIR}/vast_setup.sh" --launch --remote-repo-dir /workspace/vectur
fi
exec bash "${SCRIPT_DIR}/vast_setup.sh" "$@"

