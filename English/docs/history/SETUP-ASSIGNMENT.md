> Historical setup notes from 1 November. For current preprocessing status and
> explicitly estimated English speaker labels, read `ASSIGNMENT-WORKSPACE.md`,
> `Report/Preprocessing_Methods.txt`, and `English/artifacts/Preprocessing/`.

# FastSpeech 2 assignment: private repository and cloud setup

Updated 1 November 2026. The private repository has been created and pushed:
https://github.com/ParichayMago/fastspeech2-english-telugu

This directory is the local checkout of that private repository.
No GPU has been rented and no cloud account has been changed. The cloud
environment is prepared but has not been installed on a remote machine. This
kit does not implement the English/Telugu assignment itself.

## 1. Create the private repository from your Mac

Requirements: Git, Python 3, GitHub CLI (`gh`), a valid GitHub login, and network
access to GitHub. GitHub network access and the `ParichayMago` CLI login have
now been verified. The repository creation and push are complete. The commands
below document the completed setup; do not rerun into the existing checkout.
The connected GitHub app does not expose repository creation, so the authenticated
GitHub CLI was used.

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

Runpod has been selected with a US$20 maximum total spend. Runpod OAuth access is verified. A no-volume Pod creation was rejected for unavailable capacity;
no cloud resources have been created. The budget plan in
`setup/runpod-budget.json` reserves US$15 for GPU compute and US$5 for storage,
tax and contingency. These are planning amounts, not an active billing cap.
Proposed starting configuration:

| Setting | Recommendation |
| --- | --- |
| Provider/workload | Runpod on-demand GPU Pod |
| GPU | 1 NVIDIA L4, 24 GB VRAM; verify live availability before creating |
| System memory | At least 24 GB; prefer more for preprocessing |
| CPU | Approximately 6-8 vCPUs |
| Container | Official Runpod PyTorch template, Linux x86_64, NVIDIA driver compatible with CUDA 11.8 |
| Container disk | 50 GB of temporary storage; erased on stop/restart |
| Persistent cloud storage | None; master data and completed results stay on the Mac |
| Access | SSH; optional JupyterLab through the official template |
| Training | English and Telugu separately; initially one run at a time |

The Runpod MCP GPU catalog checked on 1 November lists NVIDIA L4 at
US$0.59/hour in Secure Cloud. The last stock read showed LOW availability in
EU-RO-1. This is not a reservation; verify again before creating. A5000 and
RTX 3090 were unavailable in the catalog reads made during this attempt.

The current Runpod plugin documents that the old stop-after/terminate-after
flags were not enforced and were removed. Do not rely on them. The setup plan
is to verify the environment, then explicitly stop the Pod through the control
plane before handing it back. Future training runs need active cost monitoring.
This budget file is not an automatic account spending cap. GPU setup and idle
time count toward the total. Keep automatic top-up disabled.

The user selected local Mac storage instead of a network volume. No network
volume will be created. The Pod still needs 50 GB of temporary container disk,
billed only while it runs. The current proposed GPU rate is US$0.59/hour;
50 GB at US$0.10/GB/month adds approximately US$0.00694/hour (720-hour month),
so the planned running rate is approximately US$0.597/hour before tax. Actual
Pod pricing must be checked after creation. GPU credit is still required.
A no-volume Pod create returned a valid-request capacity error; no Pod exists.

There is no persistent cloud volume in the selected plan. `/workspace` is a
normal directory on the Pod's temporary container disk, not a persistent mount.
Keep the master project and datasets on the Mac. Copy the working files to the
Pod, then copy checkpoints, generated audio, plots and reports back before
stopping. A stop, restart, reset or termination erases temporary Pod data.

Run this on the Mac after checkpoints/results exist, using verified direct SSH
connection details:

```bash
bash setup/sync-results-to-local.sh USER@HOST PORT PRIVATE_KEY_PATH
```

It copies the English/Telugu folders, Results, Report, logs and setup metadata
to `../fastspeech2-backups` by default, without deleting local files. It does
not copy raw datasets or stop the Pod. Run it regularly during training and
inspect the required outputs before stopping. The Mac must be reachable for a
sync to succeed. Once the Pod is stopped, GPU and container-disk billing stop;
restarting requires transferring files and rebuilding the temporary environment.

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
left enabled. Network access from this chat's terminal is now available. A
future cloud host, SSH connection and accepted host identity are still needed
before operating a GPU machine. Long jobs should run in `tmux`.

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
`/workspace/.venvs/fastspeech2` on temporary disk in this plan. It uses Python 3.10, official PyTorch 2.0.1 CUDA
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

Activate in later SSH sessions while this same Pod remains running:

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
   vocoder and verify sample rate, mel bins and frequency settings. Upstream
   includes HiFi-GAN checkpoints; verify their compatibility before using them.
6. Test a small training/inference run for each language before full training.

References:
- https://github.com/ming024/FastSpeech2
- https://huggingface.co/datasets/ai4bharat/Svarah
- https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu
- https://pytorch.org/get-started/previous-versions/
- https://montreal-forced-aligner.readthedocs.io/en/latest/installation.html

## Current validation

Completed checks: shell syntax, Python syntax, no-change plan mode, mocked
private-push guards, real GitHub authentication, private repository creation and
initial push. Cloud dependency resolution, GPU execution, dataset preparation,
Telugu support and trained audio quality remain unverified.
