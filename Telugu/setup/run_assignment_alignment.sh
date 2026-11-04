#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
LANGUAGE=${1:?Specify english or telugu}
case "$LANGUAGE" in english|telugu) ;; *) exit 2 ;; esac
export MFA_ROOT_DIR=/workspace/.cache/mfa-assignment-$LANGUAGE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH=/workspace/.venvs/mfa/bin:$PATH
export CONDA_PREFIX=/workspace/.venvs/mfa
MFA=/workspace/.venvs/mfa/bin/mfa
PREP="$ROOT/data/preprocessing/$LANGUAGE"
mkdir -p "$PREP/alignments"
if [[ "$LANGUAGE" == english ]]; then
  for split in train validation; do
    "$MFA" align "$PREP/corpus/$split" "$PREP/lexicon.txt" \
      /workspace/.cache/mfa/pretrained_models/acoustic/english_us_arpa.zip \
      "$PREP/alignments/$split" --config_path "$ROOT/Telugu/config/english-aligner.yaml" \
      --num_jobs 4 --clean
  done
else
  # Only training audio fits the Telugu acoustic aligner.
  "$MFA" train "$PREP/corpus/train" "$PREP/lexicon.txt" "$PREP/mfa-model.zip" \
    --output_directory "$PREP/alignments/train" \
    --config_path "$ROOT/Telugu/config/telugu-aligner.yaml" --num_jobs 4 --single_speaker --clean
  # Freeze that model for validation alignment.
  "$MFA" align "$PREP/corpus/validation" "$PREP/lexicon.txt" "$PREP/mfa-model.zip" \
    "$PREP/alignments/validation" --config_path "$ROOT/Telugu/config/telugu-aligner.yaml" \
    --num_jobs 4 --single_speaker --clean
fi
