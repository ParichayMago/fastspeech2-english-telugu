#!/usr/bin/env bash
# Local development helper for the one user-approved assignment Pod.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
REMOTE=/workspace/fastspeech2-english-telugu
DEST=root@213.192.2.107
SSH=(ssh -p 40130 -i "$HOME/.ssh/id_ed25519" -o IdentitiesOnly=yes
  -o BatchMode=yes -o StrictHostKeyChecking=yes
  -o "UserKnownHostsFile=$ROOT/.git/runpod_known_hosts")
case "${1:-}" in
  sync)
    # Never delete remote datasets, checkpoints or environment files.
    printf -v transport '%q ' "${SSH[@]}"
    rsync -az --no-owner --no-group \
      --exclude='/.git/' \
      --exclude='/.git/index.lock' --exclude='/.env' --exclude='/.env.*' \
      --exclude='/.venv/' --exclude='/.venvs/' --exclude='/Results/' \
      --exclude='/logs/' --exclude='/raw_data/' --exclude='/preprocessed_data/' \
      --exclude='/output/' --exclude='/Data/' --exclude='/data/' \
      --exclude='/Telugu/hifigan/*.pth.tar*' \
      --exclude='__pycache__/' \
      -e "$transport" "$ROOT/" "$DEST:$REMOTE/"
    ;;
  run)
    shift
    (( $# )) || { printf 'Usage: bash Telugu/setup/gpu.sh run COMMAND [ARGS...]\n' >&2; exit 2; }
    printf -v command '%q ' "$@"
    "${SSH[@]}" "$DEST" "cd '$REMOTE' && export NLTK_DATA=/workspace/.cache/nltk_data MPLBACKEND=Agg HF_HOME=/workspace/.cache/huggingface && export PATH=/workspace/.venvs/fastspeech2/bin:\$PATH && $command"
    ;;
  pull-checks)
    target="$ROOT/Results/SetupChecks/2026-11-02-rtx4000"
    mkdir -p "$target"
    printf -v transport '%q ' "${SSH[@]}"
    rsync -az -e "$transport" "$DEST:/workspace/logs/" "$target/"
    ;;
  *)
    printf 'Usage: bash Telugu/setup/gpu.sh {sync|run COMMAND [ARGS...]|pull-checks}\n' >&2
    exit 2
    ;;
esac
