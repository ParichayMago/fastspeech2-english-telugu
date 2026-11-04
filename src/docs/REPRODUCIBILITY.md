# Reproducibility guide

Run all commands from the repository root. The [main README](../../README.md)
provides the short inference setup; this guide covers environment, dataset
restoration, preprocessing, independent training and verification.

## Environment

Use **Python 3.10** in a project-specific environment. CPU inference was executed
locally with PyTorch 2.8.0. The recorded CUDA training used PyTorch 2.8.0+cu128.

For inference only:

```bash
uv venv --python 3.10 .venvs/final-inference
uv pip install --python .venvs/final-inference/bin/python -r src/setup/requirements-inference.txt
python3 src/setup/download_checkpoints.py
.venvs/final-inference/bin/python src/setup/prepare_final_inference.py --nltk-only
```

For the full preprocessing/training stack, create a separate environment:

```bash
uv venv --python 3.10 .venvs/assignment
uv pip install --python .venvs/assignment/bin/python -r requirements.txt
```

On a compatible Linux NVIDIA machine, install matching CUDA wheels **before** the
full requirements command:

```bash
uv pip install --python .venvs/assignment/bin/python torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

Forced alignment uses MFA **3.3.9 in a separate conda-forge environment**. The
Linux alignment wrapper expects `/workspace/.venvs/mfa/bin/mfa` and the saved
English `english_us_arpa.zip` model. See `src/setup/bootstrap-preprocessing.sh` for
its environment setup and system eSpeak-NG dependency. Native feature-extraction
packages may require build tools and libsndfile. Report authoring dependencies
are separate in `Report/requirements-report.txt`; optional English ASR diagnostics
use `openai-whisper==20250625`.

## Frozen weights and inference

Required local files:

| File | SHA256 |
| --- | --- |
| `English/checkpoints/final.pth.tar` | `74ce8b28183ec37f98a001bfcd5b484cb660a2eea2d1a780ef6dc8ea7e6a38bc` |
| `Telugu/checkpoints/final.pth.tar` | `2268f94c05c0c90d9a4b85780971b40dd07297bc39a16271b375311d4b7fcfb8` |
| `src/hifigan/generator_universal.pth.tar` | `771eaf4876485a35e25577563d390c262e23c2421e4a8c929eacfde34a5b7a60` |

Inference verifies weights, resources, symbol inventory and vocoder hashes.
It does not require raw datasets or training feature directories. Run `python3 src/setup/download_checkpoints.py` to download the submission archive
and restore only these three hash-verified files. Use `--archive /path/to/fastspeech2-assignment-v5.zip`
to reuse a download, or `--check-only` to verify existing weights without network
access. `src/setup/prepare_final_inference.py --nltk-only` downloads text resources
while leaving frozen weights/configs intact.

Run commands from the repository root:

```bash
.venvs/final-inference/bin/python English/inference.py --test-id E1
.venvs/final-inference/bin/python Telugu/inference.py --test-id T1
.venvs/final-inference/bin/python English/inference.py --text "I like Telugu."
.venvs/final-inference/bin/python Telugu/inference.py --text "నాకు తెలుగు అంటే ఇష్టం."
.venvs/final-inference/bin/python src/setup/generate_assignment_outputs.py
.venvs/final-inference/bin/python src/setup/verify_assignment_outputs.py
```

The ten exact source strings are in `src/setup/assignment_test_sentences.json`.
The supplied split words `rhyth m` and `అడ విలో` are preserved. Spoken number/date
normalization is stored separately from source text. Unsupported Telugu phones
or numeric ordinals fail explicitly rather than being silently removed.

Open `Results/Assignment/index.html` to review all audios and plots together.
The output ZIP there contains test results, not the full source or model weights.

## Data preparation

Source selections target approximately 5 hours training and 1 hour validation per
language. Seed 20261102 selects atomic groups of repeated transcript/audio and
same-recording English chunks; no group crosses splits. This is an utterance-held-
out split, not a speaker-disjoint evaluation. Exact dataset revisions, raw hours
and split hashes are recorded in `Results/DataSplits/summary.json`.

| Language | Usable train / validation clips | Aligned train / validation hours |
| --- | --- | --- |
| English | 3,037 / 617 | 4.6536 / 0.9346 |
| Telugu | 2,153 / 428 | 4.3127 / 0.8616 |

Aligned duration is lower than selected raw duration because edge silence is
trimmed; two English training clips were rejected. Estimated English labels are
fit on training audio and validation assigns to frozen centroids.

Acquire dataset access under each source's conditions. Raw data, embeddings,
TextGrids and features are excluded from Git. Audit/download, split selection,
raw export and speaker estimation are implemented in `src/setup/audit_datasets.py`,
`src/setup/prepare_splits.py`, `src/setup/export_split_audio.py` and
`src/setup/estimate_english_speakers.py`. Source membership, partitions and frozen conditioning labels are distributed in
`Results/DataSplits/selection-membership.jsonl`, without raw text or audio.
After obtaining gated Svarah access, the following commands reconstruct the same
source selections from the pinned datasets in a clean checkout:

```bash
python src/setup/audit_datasets.py --dataset english --download --revision ebbf7777fe771490696a3f7b007097606fa8c924
python src/setup/audit_datasets.py --dataset telugu --download --revision 7ab3c18491bef3f5d4e820c2701b5f78f54031d6
python src/setup/restore_assignment_splits.py
python src/setup/export_split_audio.py
```

The restoration command verifies dataset revisions and audio hashes and preserves
the estimated speaker labels; it refuses to overwrite differing existing splits.
Restored metadata is not claimed byte-identical to historical metadata.
The source recordings, split membership and conditioning labels are fixed.

On the prepared Linux environment, with those manifests/audio restored:

```bash
python English/preprocess.py --stage all --device cpu --feature-version gap-fixed-v2
python Telugu/preprocess.py --stage all --device cpu --feature-version gap-fixed-v2
python src/setup/verify_preprocessed_features.py --language english --feature-version gap-fixed-v2
python src/setup/verify_preprocessed_features.py --language telugu --feature-version gap-fixed-v2
```

Stages are `corpus`, `align`, `features`, and `check`. The `check` stage performs a
throwaway batch update to verify gradients, without producing a trained model.
The pause feature builder derives all three training YAMLs from measured
training-only pause statistics. It requires an unbuilt feature destination and
refuses to overwrite completed features:

```bash
python src/setup/build_learned_pause_features.py
python src/setup/test_learned_pause.py
```

If features already exist, recreate only their configurations with
`python src/setup/build_learned_pause_features.py --configs-only`. The published
`src/config/learned-pauses-v1/English/` also records the original experiment
configuration; newly measured data must use its regenerated configuration.

See `Report/Alignment_Gap_Fix.txt` and `Report/Learned_Pause_Pilot.txt` for target
verification. Older preprocessing/training notes are historical; the report and
`gap-fixed-v2` rebuild summary provide the final feature counts/hours.

## Training

Training defaults to CUDA and requires prepared features; the scripts do not
rent machines. Explicit `--device cpu` supports local synthetic smoke tests.
Each language entry point fixes its language and requires an isolated run ID.
Use a new ID with `--fresh` to initialize new weights. Use `--resume` without
initialization flags to restore the existing run's saved
model type, exact configurations, optimizer groups, rates, best checkpoint and
early-stopping counter. Original parent files and generated YAMLs are not needed
for resume; the checkpoint contains the configurations. Missing or inconsistent
best files/metadata cause an error instead of silently replacing the best.

```bash
python English/train.py --run-id existing-english-run --resume --minutes 30 --max-steps 35000
python Telugu/train.py --run-id existing-telugu-run --resume --minutes 30 --max-steps 35000
```

Resume preserves the original run-config.json and appends resume-history.jsonl.
Full latest checkpoints with Adam state are required; released model-only best
weights alone cannot resume an optimizer. Omitting both --fresh and --resume also
resumes an existing checkpoint, but never starts a new run implicitly.
Resume reshuffles data and is not a bit-exact mid-epoch continuation.

```bash
python English/train.py --run-id english-new --fresh --minutes 120 --max-steps 30000 --batch-size 8
python Telugu/train.py --run-id telugu-new --fresh --minutes 120 --max-steps 30000 --batch-size 8
```

These are alternative independent runs. To reproduce the paired scheduling used
in the corrected experiment, the shared runner alternates separate optimizers:

```bash
python src/setup/train_assignment.py --run-id bilingual-new --fresh --language both --minutes 120 --max-steps 30000 --batch-size 8
```

Both bounds apply: elapsed minutes or maximum steps, whichever is reached first.
Checkpoints are saved every 100 steps and complete validation every 500. Run
arguments, configurations, data/source fingerprints and metrics are saved under
`output/runs/<run-id>/`. A saved `session/STOP` requests a graceful stop. The
runner rejects a retained marker before writing any session/checkpoint files,
including on implicit resume. To continue a deliberately stopped run, first
confirm the run ID and remove only its `output/runs/<run-id>/session/STOP` file,
then rerun with `--resume`. The runner never removes a stop request automatically.
The historical local supervisors supplied credit checks, verified backups and Pod
shutdown; a trainer command alone is not a provider billing cap or shutdown guard.

English fine-tuning accepts `--initialize-from` (a checkpoint under `output/`),
its exact `--initialize-sha256`, `--learning-rate 2e-5` and
`--early-stop-patience 5`. The learned-pause pilot starts from the fine-tune best
and adds `--learned-pauses`,
`--config-root data/rebuild-config/learned-pauses-v1`,
`--new-head-learning-rate 2e-4` and patience 8. Both use `--fresh` and new run IDs.
Exact historical arguments are in each run's language-specific `run-config.json`.

## Evaluation and checks

```bash
.venvs/final-inference/bin/python -m unittest discover -s src/setup -p test_final_inference.py -v
.venvs/final-inference/bin/python src/setup/verify_assignment_outputs.py
```

Training curves in `Results/English/` describe the selected learned-pause run;
Telugu curves describe its corrected 30k run. Earlier stages remain under
`Results/Training/`. The technical report compares all stages separately. Portable raw metrics and
run configurations are included under `Results/Training/<run-id>/<language>/`;
analysis falls back to those copies when local `output/runs/` backups are absent.
Duration plots are in seconds, F0 plots are denormalized **model conditioning**
in Hz, and energy is linear spectral L2, not dB. These are not measured waveform
pitch or subjective quality scores. English ASR is diagnostic; Telugu recognition
was excluded after its natural-audio control failed. Native listening is pending.


## Additional verification and listening review

Recovery tests require the full `requirements.txt` stack, including TensorBoard.
These checks do not establish speech quality. Private-data integration tests
explicitly skip when their fixtures are absent; a skip is not a pass.

```bash
python -m unittest discover -s src/setup -p test_review_fixes.py -v
python -m unittest discover -s src/audio_review -p test_server.py -v
python src/setup/verify_frozen_regeneration.py
python src/setup/verify_submission_layout.py
```

To collect pronunciation and quality observations for the required recordings:

```bash
python3 src/audio_review/server.py --port 8770 --store data/audio-review/listening-review-v1.json
```

Open `http://127.0.0.1:8770/?view=final`. Reviews stay local; export observations
and add the actual findings to the report. Telugu pronunciation needs a
Telugu-speaking reviewer. No formal MOS panel or A/B test is required by the
assignment. The current report documents the remaining evaluation gap.
