#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
WORKSPACE=${FS2_WORKSPACE:-/workspace}
STORAGE_MODE=${FS2_STORAGE_MODE:-ephemeral}
TORCH_VERSION=${FS2_TORCH_VERSION:-2.0.1}
TORCHAUDIO_VERSION=${FS2_TORCHAUDIO_VERSION:-2.0.2}
TORCH_CUDA_INDEX=${FS2_TORCH_CUDA_INDEX:-cu118}
case "$TORCH_CUDA_INDEX" in
  cu118|cu128) ;;
  *) printf 'Supported PyTorch CUDA wheel indexes: cu118, cu128.\n' >&2; exit 1 ;;
esac
if [[ ${1:-} == --plan ]]; then
  printf 'Install isolated Python 3.10 environment under %s/.venvs/fastspeech2\n' "$WORKSPACE"
  printf 'Install PyTorch %s, torchaudio %s (%s), and pinned speech dependencies.\n' "$TORCH_VERSION" "$TORCHAUDIO_VERSION" "$TORCH_CUDA_INDEX"
  printf 'Download English NLTK resources and run GPU/model/feature checks.\n'
  printf 'Does not provision machines, start training, or download model weights or datasets.\n'
  exit 0
fi
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || {
  printf 'Run this on the Linux x86_64 GPU machine, not on your Mac.\n' >&2; exit 1;
}
[[ -f "$ROOT/train.py" && -f "$ROOT/LICENSE" ]] || { printf 'Run inside the prepared FastSpeech2 checkout.\n' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { printf 'NVIDIA driver/GPU not available.\n' >&2; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [[ "$STORAGE_MODE" == ephemeral ]]; then
  mkdir -p "$WORKSPACE"
  printf 'Ephemeral storage: copy checkpoints and results to the Mac BEFORE stopping this Pod.\n'
elif [[ "$STORAGE_MODE" != persistent ]]; then
  printf 'FS2_STORAGE_MODE must be ephemeral or persistent.\n' >&2
  exit 1
fi
[[ -d "$WORKSPACE" && -w "$WORKSPACE" ]] || { printf 'Workspace is missing or unwritable: %s\n' "$WORKSPACE" >&2; exit 1; }

# System packages belong to this disposable cloud container, not the laptop.
if [[ $(id -u) == 0 ]]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates git tmux rsync ffmpeg libsndfile1 \
    libgomp1 build-essential pkg-config
else
  for tool in python3 ffmpeg tmux g++; do
    command -v "$tool" >/dev/null || { printf 'Missing system package for %s; install it in the container first.\n' "$tool" >&2; exit 1; }
  done
fi
export UV_CACHE_DIR="$WORKSPACE/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORKSPACE/.tools/python"
export HF_HOME="$WORKSPACE/.cache/huggingface"
export NLTK_DATA="$WORKSPACE/.cache/nltk_data"
export MPLBACKEND=Agg
mkdir -p "$WORKSPACE/.tools" "$HF_HOME" "$NLTK_DATA" "$WORKSPACE/logs"
TOOLS="$WORKSPACE/.tools/uv"
if [[ ! -f "$TOOLS/uv/__init__.py" ]]; then
  python3 -m pip install --target "$TOOLS" uv==0.5.14
fi
UV=(env "PYTHONPATH=$TOOLS" python3 -m uv)
VENV="$WORKSPACE/.venvs/fastspeech2"
if [[ ! -x "$VENV/bin/python" ]]; then
  "${UV[@]}" venv --python 3.10 --seed "$VENV"
fi
PY="$VENV/bin/python"
"$PY" -c 'import sys; assert sys.version_info[:2] == (3,10), "Expected Python 3.10"'
"$PY" -m pip install pip==24.3.1 setuptools==69.5.1 wheel==0.44.0
"$PY" -m pip install numpy==1.23.5 Cython==0.29.37
"$PY" -m pip install "torch==$TORCH_VERSION" "torchaudio==$TORCHAUDIO_VERSION" \
  --index-url "https://download.pytorch.org/whl/$TORCH_CUDA_INDEX"
"$PY" -m pip install --no-build-isolation -r "$ROOT/Telugu/setup/requirements-cloud.txt"
"$PY" -m pip check
"$PY" -m nltk.downloader -d "$NLTK_DATA" cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng

"$PY" "$ROOT/Telugu/setup/check_environment.py" --repo "$ROOT" --require-cuda \
  --output "$WORKSPACE/logs/environment-check.json"
"$PY" -m pip freeze > "$WORKSPACE/logs/requirements-resolved.txt"
git -C "$ROOT" rev-parse HEAD > "$WORKSPACE/logs/source-commit.txt"
printf 'Environment checks passed. Activate: source %s/bin/activate\n' "$VENV"
printf 'Set NLTK_DATA=%s and HF_HOME=%s in future sessions.\n' "$NLTK_DATA" "$HF_HOME"
printf 'Dataset preparation, aligner setup, Telugu support and vocoder weights are separate next steps.\n'
