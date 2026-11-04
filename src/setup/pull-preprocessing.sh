#!/usr/bin/env bash
# Back up preprocessing artifacts; no Pod lifecycle changes and no deletions.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
DEST=root@157.157.221.29
REMOTE=/workspace/fastspeech2-english-telugu
SSH=(ssh -p 22117 -i "$HOME/.ssh/id_ed25519" -o IdentitiesOnly=yes
  -o BatchMode=yes -o StrictHostKeyChecking=yes
  -o "UserKnownHostsFile=$ROOT/.git/runpod_known_hosts")
printf -v transport '%q ' "${SSH[@]}"
for language in english telugu; do
  mkdir -p "$ROOT/data/features/$language" "$ROOT/data/preprocessing/$language"
  rsync -az --copy-links --no-owner --no-group -e "$transport" \
    "$DEST:$REMOTE/data/features/$language/" "$ROOT/data/features/$language/"
  rsync -az --copy-links --no-owner --no-group -e "$transport" \
    "$DEST:$REMOTE/data/preprocessing/$language/" "$ROOT/data/preprocessing/$language/"
done
for language in English Telugu; do
  mkdir -p "$ROOT/src/config/$language"
  rsync -az --no-owner --no-group -e "$transport" \
    "$DEST:$REMOTE/config/$language/" "$ROOT/src/config/$language/"
done
mkdir -p "$ROOT/src/artifacts/Preprocessing" "$ROOT/data/preprocessing/logs"
rsync -az --no-owner --no-group -e "$transport" \
  "$DEST:$REMOTE/src/artifacts/Preprocessing/" "$ROOT/src/artifacts/Preprocessing/"
rsync -az --no-owner --no-group -e "$transport" \
  "$DEST:$REMOTE/text/assignment_phones.json" "$ROOT/src/text/"
rsync -az --no-owner --no-group -e "$transport" \
  "$DEST:/workspace/logs/" "$ROOT/data/preprocessing/logs/"
rsync -az --no-owner --no-group -e "$transport" \
  "$DEST:$REMOTE/data/preprocessing/artifact-sha256.json" "$ROOT/data/preprocessing/"
printf 'Preprocessing artifacts copied. Verify locally before any Pod stop.\n'
