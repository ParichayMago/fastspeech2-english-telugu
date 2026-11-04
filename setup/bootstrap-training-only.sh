#!/usr/bin/env bash
# Install only dependencies needed to train precomputed corrected features.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export UV_CACHE_DIR=/workspace/.cache/uv
export UV_PYTHON_INSTALL_DIR=/workspace/.tools/python
export MPLBACKEND=Agg
mkdir -p /workspace/.tools/uv /workspace/logs
if [[ ! -f /workspace/.tools/uv/uv/__init__.py ]]; then
  python3 -m pip install --target /workspace/.tools/uv uv==0.5.14
fi
UV=(env PYTHONPATH=/workspace/.tools/uv python3 -m uv)
VENV=/workspace/.venvs/fastspeech2
if [[ ! -x "$VENV/bin/python" ]]; then
  "${UV[@]}" venv --python 3.10 "$VENV"
fi
"${UV[@]}" pip install --python "$VENV/bin/python" torch==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
"${UV[@]}" pip install --python "$VENV/bin/python" \
  numpy==1.23.5 scipy==1.10.1 matplotlib==3.7.3 inflect==4.1.0 \
  Unidecode==1.3.8 PyYAML==6.0.1 tensorboard==2.14.1 protobuf==4.25.8 tqdm==4.66.5
"$VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(); print({"torch":torch.__version__,"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(0),"architectures":torch.cuda.get_arch_list()})'
"${UV[@]}" pip freeze --python "$VENV/bin/python" > /workspace/logs/training-requirements-resolved.txt
cd "$ROOT"
