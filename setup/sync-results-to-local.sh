#!/usr/bin/env bash
# Run on the Mac/workstation. Does not stop the Pod or delete local files.
set -euo pipefail
if [[ $# -lt 3 || $# -gt 4 ]]; then
  printf 'Usage: bash setup/sync-results-to-local.sh USER@HOST PORT PRIVATE_KEY_PATH [BACKUP_DIRECTORY]\n' >&2
  exit 2
fi
REMOTE=$1
PORT=$2
KEY=$3
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BACKUP=${4:-"$ROOT/../fastspeech2-backups"}
[[ "$REMOTE" =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+$ ]] || { printf 'Expected USER@HOST; use the exact direct SSH address from Runpod.\n' >&2; exit 2; }
[[ "$PORT" =~ ^[0-9]+$ ]] || { printf 'PORT must be numeric.\n' >&2; exit 2; }
[[ -f "$KEY" ]] || { printf 'Private key file not found.\n' >&2; exit 2; }
command -v rsync >/dev/null
mkdir -p "$BACKUP"
# rsync calls this wrapper with separate arguments; no key-path shell interpolation.
WRAPPER=$(mktemp)
trap 'rm -f "$WRAPPER"' EXIT
cat > "$WRAPPER" <<'SH'
#!/usr/bin/env bash
exec ssh -i "$FS2_SYNC_KEY" -p "$FS2_SYNC_PORT" -o BatchMode=yes "$@"
SH
chmod 700 "$WRAPPER"
export FS2_SYNC_KEY="$KEY" FS2_SYNC_PORT="$PORT"
rsync -av --partial --timeout=60 -e "$WRAPPER" \
  --include='/English/***' --include='/Telugu/***' \
  --include='/Results/***' --include='/Report/***' \
  --include='/logs/***' --include='/README.md' \
  --include='/requirements.txt' --include='/setup/***' \
  --exclude='*' \
  "$REMOTE:/workspace/fastspeech2-english-telugu/" "$BACKUP/"
date -u '+Completed result sync at %Y-%m-%dT%H:%M:%SZ' > "$BACKUP/last-successful-sync.txt"
printf 'Results copied to %s. Inspect required checkpoints/results before stopping the Pod.\n' "$BACKUP"
