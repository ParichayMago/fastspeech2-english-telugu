#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || exit 1
export MAMBA_ROOT_PREFIX=/workspace/.tools/mamba-root
export MFA_ROOT_DIR=/workspace/.cache/mfa
mkdir -p /workspace/.tools/micromamba /workspace/logs "$MFA_ROOT_DIR"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends espeak-ng
/workspace/.venvs/fastspeech2/bin/python -m pip install \
  -c "$ROOT/Telugu/setup/requirements-cloud.txt" speechbrain==1.0.3 phonemizer==3.3.0 num2words==0.5.14
if [[ ! -x /workspace/.tools/micromamba/micromamba ]]; then
  python3 - <<'PY'
import io,pathlib,tarfile,urllib.request
with urllib.request.urlopen('https://micro.mamba.pm/api/micromamba/linux-64/latest',timeout=60) as r:
    raw=r.read()
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:bz2') as archive:
    member=archive.extractfile('bin/micromamba')
    p=pathlib.Path('/workspace/.tools/micromamba/micromamba')
    p.write_bytes(member.read()); p.chmod(0o755)
PY
fi
if [[ ! -x /workspace/.venvs/mfa/bin/mfa ]]; then
  /workspace/.tools/micromamba/micromamba create -y --no-rc \
    -p /workspace/.venvs/mfa --override-channels -c conda-forge \
    'python=3.10' 'montreal-forced-aligner=3.3.9'
fi
/workspace/.venvs/mfa/bin/mfa version
/workspace/.venvs/mfa/bin/mfa configure --help
espeak-ng --voices=te
/workspace/.venvs/fastspeech2/bin/python -m pip freeze > /workspace/logs/preprocessing-pip-freeze.txt
/workspace/.tools/micromamba/micromamba list -p /workspace/.venvs/mfa --explicit \
  > /workspace/logs/mfa-environment-explicit.txt
