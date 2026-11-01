# FastSpeech 2 assignment: private repository and cloud setup

Prepared 1 November 2026. No GPU has been rented, no cloud account has been
changed, and no private GitHub repository has yet been created. This kit contains
scripts to perform those repository and environment setup steps once access is
available. It does not implement the English/Telugu assignment itself.

## 1. Create the private repository from your Mac

Requirements: Git, Python 3, GitHub CLI (`gh`), a valid GitHub login, and network
access to GitHub. The current agent terminal cannot resolve GitHub; its `gh`
login check also failed. The connected GitHub app can read upstream code but does
not expose repository creation.

If login is needed, run this in your own Terminal and complete the browser flow:

```bash
gh auth login --hostname github.com --git-protocol https --web
```

Run the following from the directory containing this README:

```bash
bash create-private-repo.sh --plan
bash create-private-repo.sh ../fastspeech2-english-telugu
```

The script checks the GitHub account is `ParichayMago` and that Git has an author
identity. It clones `ming024/FastSpeech2`, preserves its history, keeps its remote
as `upstream`, and creates a `main` branch at:

```
d4e79eb52e8b01d24703b2dfc0385544092958f3
```

It copies this setup into the project, creates the required empty deliverable
folders, commits those additions and creates
`ParichayMago/fastspeech2-english-telugu` with `--private`. It verifies server-side
visibility before pushing `main`. It preserves the original README and MIT
license. It does not create a public fork, rent a machine, or change repository
visibility later. A failed run leaves its local files for inspection; use a new
destination for another attempt rather than deleting work automatically.

The assignment eventually requires a public submission. Keep development private
now and review the final repository before an explicitly requested visibility
change. Checkpoints are ignored by ordinary Git and will need Git LFS or a
documented external download at submission.

## 2. Machine recommendation for later

No rental is authorized at present. Proposed starting configuration:

| Setting | Recommendation |
| --- | --- |
| Provider/workload | Runpod on-demand GPU Pod |
| GPU | 1 NVIDIA RTX 4090, 24 GB VRAM |
| System memory | At least 32 GB where available |
| CPU | Approximately 6-8 vCPUs |
| Container | Official Runpod PyTorch template, Linux x86_64, NVIDIA driver compatible with CUDA 11.8 |
| Container disk | 30 GB starting allocation |
| Persistent storage | 100 GB at `/workspace`, preferably a network volume |
| Access | SSH; optional JupyterLab through the official template |
| Training | English and Telugu separately; initially one run at a time |

The live pricing page checked on 1 November lists RTX 4090 at US$0.89/hour.
That is about US$21.36 for 24 active hours or US$42.72 for 48 active hours,
before storage/tax. These are budgeting examples, not training-time estimates.
Confirm the exact console quote and availability before any rental. The older
GPU-specific marketing page shows different rates, so use the live console.

Network volumes are independent of the Pod. A Pod's ordinary volume survives
stopping but is deleted when the Pod is deleted; the container disk is lost on
stop/restart. Keep checkpoints and data under `/workspace` and maintain a second
backup before deleting anything. Storage can continue to incur charges while
compute is stopped.

Sources:
- https://www.runpod.io/pricing
- https://docs.runpod.io/pods/storage/types
- https://docs.runpod.io/pods/connect-to-a-pod

## 3. Transfer the checkout to a future GPU Pod

Use the exact SSH host, port, user and already-configured key shown by the cloud
provider. Do not paste private keys or API tokens into chat or commit them.
The following are templates; substitute the connection details before running:

```bash
ssh -p PORT USER@HOST 'mkdir -p /workspace/fastspeech2-english-telugu'
rsync -av -e 'ssh -p PORT' \
  --exclude='.env*' --exclude='.venv*' --exclude='__pycache__' \
  ../fastspeech2-english-telugu/ \
  USER@HOST:/workspace/fastspeech2-english-telugu/
```

This copies the checkout and Git metadata without copying GitHub credentials.
No GitHub login is required on the GPU machine. SSH host identity checking is
left enabled. This chat's current terminal network restriction must be resolved
before the agent can use SSH; an authorized provider web terminal is an
alternative when available. Long jobs should run in `tmux`.

## 4. Bootstrap on the GPU machine

Inside the copied checkout on the cloud machine:

```bash
cd /workspace/fastspeech2-english-telugu
bash setup/bootstrap-cloud.sh --plan
tmux new -s fs2-setup
set -o pipefail
bash setup/bootstrap-cloud.sh 2>&1 | tee /workspace/fs2-bootstrap.log
```

The bootstrap checks the GPU, installs required container system packages when
run as root, and creates an isolated environment at
`/workspace/.venvs/fastspeech2`. It uses Python 3.10, official PyTorch 2.0.1 CUDA
11.8 wheels and proposed pinned speech dependencies. This avoids changing the
PyTorch installation supplied by the cloud template. The old upstream
`requirements.txt` is retained as a reference; do not install it over this
environment.

The pins are a compatibility starting point, not a complete transitive lock.
After the first successful installation the script saves the fully resolved
package list to `/workspace/logs/requirements-resolved.txt`. Preserve that list
with the final experiment configuration.

The environment check exercises actual upstream English G2P, model
forward/backward, WORLD pitch extraction and CUDA mel/energy extraction. It uses
temporary synthetic data only to check software and does not represent model
training or evaluation. It writes `/workspace/logs/environment-check.json` only
after all checks pass. Do not regard an install-only success as readiness.

Activate in subsequent SSH sessions:

```bash
source /workspace/.venvs/fastspeech2/bin/activate
export NLTK_DATA=/workspace/.cache/nltk_data
export HF_HOME=/workspace/.cache/huggingface
export MPLBACKEND=Agg
```

The dependency installation and GPU smoke check have NOT been run on a cloud
machine yet. No pretrained weights or datasets are downloaded by the bootstrap.

## 5. Next steps after environment verification

1. Audit and download Svarah and IndicTTS Telugu. Build reproducible, nonoverlapping
   5-hour training / 1-hour validation selections per language.
2. Set up a separate Conda environment for Montreal Forced Aligner. Follow its
   official installation guide; verify its non-root/database requirements for
   the chosen version. Do not use a pip-only install of MFA without its native
   dependencies. No aligner is installed by this kit.
3. Verify a small Telugu text/audio alignment. The upstream repository supplies
   English and Mandarin frontends; Telugu requires its own tested text/phoneme
   inventory and alignment strategy. Do not use English G2P for Telugu.
4. Configure speaker handling. Svarah has multiple speakers; preserve the
   available identities and explicitly document any speaker selection.
5. Extract real training statistics and features. Obtain a compatible pretrained
   vocoder separately and verify sample rate, mel bins and frequency settings.
6. Test a small training/inference run for each language before full training.

References:
- https://github.com/ming024/FastSpeech2
- https://huggingface.co/datasets/ai4bharat/Svarah
- https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu
- https://pytorch.org/get-started/previous-versions/
- https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html

## Current validation

Local checks: shell syntax, Python syntax, no-change plan mode, and mocked checks
that only a confirmed private repository is pushed. These do not verify GitHub
authentication, real repository creation, dependency resolution, GPU execution,
dataset preparation, Telugu support or trained audio quality.
