# FastSpeech 2 for English and Telugu

Independent English and Telugu text-to-speech models adapted from
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), using Svarah and
IndicTTS Telugu. The project includes preprocessing, training, frozen inference,
10 assignment test audios, 30 variance plots, loss curves and a technical report.

The submission release contains the complete project, English and Telugu weights,
and the required vocoder. Large model files are delivered through GitHub Releases
rather than ordinary Git. The repository URL is intended for assignment evaluation.

## Implementation and results

- Separate models, optimizers, configurations, features and checkpoints for each language.
- English uses 254 **estimated acoustic speaker labels**, not verified identities.
  Final samples use `en_est_0155`. Of 254 labels, 77 have only 1–2 selected training clips;
  median source coverage is 38.3 seconds. The final label has 90 selected / 89
  accepted clips (7.500 / 7.482 source minutes). These are source WAV durations,
  not cropped feature durations. Telugu uses the source `te_male` label.
- `gap-fixed-v2` preserves internal alignment gaps and complete duration targets.
  Both models trained afresh on these corrected features to 30,000 updates each.
- English base best: step 12,000, validation 2.6051. Telugu best: step 18,000,
  validation 2.1065. Later updates did not improve their held-out minima.
- English fixed-rate fine-tuning selected +2,500 updates (2.5788). The final English
  model adds word-boundary/punctuation conditioning and learned pause heads,
  selecting +1,000 pause-pilot updates (3.0544 under its new objective).
- The pause objective adds two weighted losses and changes masks. Its total is
  not comparable with the base-model total. Loss improvements do not establish
  clearer or more natural speech; final audio listening review is pending.
- Both models use a frozen universal HiFi-GAN. Final outputs use unity controls
  and no denoising, external silence, sentence concatenation or duration scaling.

## Layout

```text
English/ and Telugu/
  train.py, preprocess.py, inference.py, config.yaml
  checkpoints/final.pth.tar, checkpoints/checkpoint.json
  resources/                    # frozen lexicon, stats and speaker mappings
Results/English/ and Results/Telugu/
  training_loss.png, validation_loss.png, loss_history.csv
  test_generated_audios/        # WAV, exact text, JSON, NPZ and CSV
  variance_adaptor_plots/{duration,pitch,energy}/
Results/Assignment/              # review page, manifests and verification
Results/Training/               # saved run observations and backup records
Report/Technical_Report.pdf
setup/                          # shared implementations and dependency profiles
config/English/ and config/Telugu/  # preprocessing/model/training YAMLs
```

`English/config.yaml` and `Telugu/config.yaml` configure **frozen inference**.
Training uses the three YAML files in each language's `config/` directory; the
learned-pause experiment uses `data/rebuild-config/learned-pauses-v1/English/`.

## Complete submission download

Download [fastspeech2-assignment-v3.zip](https://github.com/ParichayMago/fastspeech2-english-telugu/releases/download/assignment-v3/fastspeech2-assignment-v3.zip)
and [SHA256SUMS](https://github.com/ParichayMago/fastspeech2-english-telugu/releases/download/assignment-v3/SHA256SUMS)
from the [assignment release](https://github.com/ParichayMago/fastspeech2-english-telugu/releases/tag/assignment-v3).
The ZIP contains a `project/` folder matching the required layout, including all
three model files, the report, code, 10 audios and 30 variance plots. Verify it
with `shasum -a 256 -c SHA256SUMS` before extracting.

## Environment

Use **Python 3.10** in a project-specific environment. CPU inference was executed
locally with PyTorch 2.8.0. The recorded CUDA training used PyTorch 2.8.0+cu128.

For inference only:

```bash
uv venv --python 3.10 .venvs/final-inference
uv pip install --python .venvs/final-inference/bin/python -r setup/requirements-inference.txt
python3 setup/download_checkpoints.py
.venvs/final-inference/bin/python setup/prepare_final_inference.py --nltk-only
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
English `english_us_arpa.zip` model. See `setup/bootstrap-preprocessing.sh` for
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
| `hifigan/generator_universal.pth.tar` | `771eaf4876485a35e25577563d390c262e23c2421e4a8c929eacfde34a5b7a60` |

Inference verifies weights, resources, symbol inventory and vocoder hashes.
It does not require raw datasets or training feature directories. Run `python3 setup/download_checkpoints.py` to download the submission archive
and restore only these three hash-verified files. Use `--archive /path/to/fastspeech2-assignment-v3.zip`
to reuse a download, or `--check-only` to verify existing weights without network
access. `setup/prepare_final_inference.py --nltk-only` downloads text resources
while leaving frozen weights/configs intact.

Run commands from the repository root:

```bash
.venvs/final-inference/bin/python English/inference.py --test-id E1
.venvs/final-inference/bin/python Telugu/inference.py --test-id T1
.venvs/final-inference/bin/python English/inference.py --text "I like Telugu."
.venvs/final-inference/bin/python Telugu/inference.py --text "నాకు తెలుగు అంటే ఇష్టం."
.venvs/final-inference/bin/python setup/generate_assignment_outputs.py
.venvs/final-inference/bin/python setup/verify_assignment_outputs.py
```

The ten exact source strings are in `setup/assignment_test_sentences.json`.
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
raw export and speaker estimation are implemented in `setup/audit_datasets.py`,
`setup/prepare_splits.py`, `setup/export_split_audio.py` and
`setup/estimate_english_speakers.py`. Source membership, partitions and frozen conditioning labels are distributed in
`Results/DataSplits/selection-membership.jsonl`, without raw text or audio.
After obtaining gated Svarah access, the following commands reconstruct the same
source selections from the pinned datasets in a clean checkout:

```bash
python setup/audit_datasets.py --dataset english --download --revision ebbf7777fe771490696a3f7b007097606fa8c924
python setup/audit_datasets.py --dataset telugu --download --revision 7ab3c18491bef3f5d4e820c2701b5f78f54031d6
python setup/restore_assignment_splits.py
python setup/export_split_audio.py
```

The restoration command verifies dataset revisions and audio hashes and preserves
the estimated speaker labels; it refuses to overwrite differing existing splits.
Restored metadata is not claimed byte-identical to historical metadata.
The source recordings, split membership and conditioning labels are fixed.

On the prepared Linux environment, with those manifests/audio restored:

```bash
python English/preprocess.py --stage all --device cpu --feature-version gap-fixed-v2
python Telugu/preprocess.py --stage all --device cpu --feature-version gap-fixed-v2
python setup/verify_preprocessed_features.py --language english --feature-version gap-fixed-v2
python setup/verify_preprocessed_features.py --language telugu --feature-version gap-fixed-v2
```

Stages are `corpus`, `align`, `features`, and `check`. The `check` stage performs a
throwaway batch update to verify gradients, without producing a trained model.
The pause feature builder derives all three training YAMLs from measured
training-only pause statistics. It requires an unbuilt feature destination and
refuses to overwrite completed features:

```bash
python setup/build_learned_pause_features.py
python setup/test_learned_pause.py
```

If features already exist, recreate only their configurations with
`python setup/build_learned_pause_features.py --configs-only`. The published
`config/learned-pauses-v1/English/` also records the original experiment
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
python English/train.py --run-id english-new --resume --minutes 30 --max-steps 35000
python Telugu/train.py --run-id telugu-new --resume --minutes 30 --max-steps 35000
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
python setup/train_assignment.py --run-id bilingual-new --fresh --language both --minutes 120 --max-steps 30000 --batch-size 8
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
.venvs/final-inference/bin/python -m unittest discover -s setup -p test_final_inference.py -v
.venvs/final-inference/bin/python setup/verify_assignment_outputs.py
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

## Corrected release: assignment-v3

This release fixes best-checkpoint preservation, base/fine-tune/pause resume,
pause-configuration reconstruction, tensor device allocation, energy-control
wiring and review-UI metadata compatibility. Frozen weights are unchanged; all
ten WAVs and every prediction array regenerate identically under unity controls.
This version also rejects retained STOP markers before session writes and
discovers all ten final recordings in the annotation server. It documents speaker
coverage and remaining speech-quality/punctuation limitations. The current local
suites pass 36 setup and nine server tests; browser playback/save/export checks
also pass. No retraining was performed or required for these software fixes.

Install the full `requirements.txt` stack, including TensorBoard, before running
the training-recovery tests below. The minimal inference profile is sufficient
for frozen regeneration and final-frontend checks; HTTP server tests use only
Python's standard library.

```bash
python -m unittest discover -s setup -p test_review_fixes.py -v
python -m unittest discover -s audio_review -p test_server.py -v
python setup/verify_frozen_regeneration.py
```

The first suite uses portable synthetic fixtures and tests real CPU interrupted
runs for base, fixed-rate and learned-pause models, including legacy checkpoints.
The full setup suite also includes local real-data integration tests; those tests
explicitly skip when their private fixtures are absent. A skip is not a pass.
CUDA host execution was not rerun for this correction.

Pause occurrence baselines are saved in
`Results/Assignment/analysis/pause-baselines.json` and the updated report. Model
F1 is 45.6%, versus 38.5% for every-boundary and 21.0% for punctuation-only
prediction. This does not establish audible improvement. Per-sentence native
listening observations are still required; no MOS ratings are invented.

## Local listening review and optional English checkpoint comparison

The review catalog includes all ten frozen recordings under
`Results/{English,Telugu}/test_generated_audios/`, with original transcripts and
current WAV hashes. Choose **Final assignment audio** to collect per-sentence
pronunciation and quality observations. Telugu needs a Telugu-speaking reviewer.
Existing dataset and historical sample reviews remain separate choices.

```bash
.venvs/inference/bin/python setup/prepare_listening_comparison.py
.venvs/inference/bin/python audio_review/server.py --port 8770 --store data/audio-review/listening-review-v1.json
```

Open `http://127.0.0.1:8770/?view=comparison`. Six A/B pairs cover E1, E2a,
E2b, E3, E4 and the user's MIT passage. Both systems use the same text, lexical
pronunciations, voice `en_est_0155`, statistics and universal HiFi-GAN, with unity
controls and no denoising, injected silence, sentence splitting or waveform edits.
This optional comparison requires the retained local parent checkpoint; it is
not required for final-recording review and that historical parent is not bundled.
The retained plain parent is fine-tune step 2500 (initialized from corrected
step 12000); the child is learned-pause step 1000. The parent uses lexical phones
without the child's new boundary slots. This compares the two systems, not just
the pause head in isolation. The five learned assignment WAVs are checked against
the released WAV hashes; canonical outputs are never overwritten.

Listen to both versions at normal speed, note exact errors separately for A and B,
then record preferences for clarity, pacing, intonation and overall quality.
“About the same”, “Neither is acceptable” and “Unsure” are valid responses.
A/B ordering is balanced and fixed before listening; the UI hides system labels.
The auditable key is in `Results/Listening/english-parent-vs-learned-pauses/comparison.json`;
avoid opening that key until judgments are saved. This is a label-hidden review,
not a secured clinical blind study. No listening judgments are pre-populated.

Individual reviews persist in `data/audio-review/listening-review-v1.json`;
paired judgments persist in `data/audio-review/listening-review-v1-pairs.json`.
Only judgments matching both current audio hashes are shown/exported. The A/B
page exports all saved comparisons as JSON; the individual review view offers
**Export all observations** for every reviewed status, or good selections as CSV.
Review data remain local and excluded from Git. Preparation
uses CPU inference only; it does not resume training or start cloud resources.

Speaker coverage can be recomputed without raw datasets:

```bash
python setup/analyze_speaker_coverage.py
```

The saved analysis distinguishes 3,039 selected from 3,037 accepted English
training clips, and source WAV coverage from aligned feature hours. The clustering
threshold is heuristic (cosine distance 0.30). A count of 254 estimated labels
versus the publisher's 117 speakers warrants an audit, not an assertion of exact
identity errors or a proven cause of noise. Final English pause benefits remain
unproven by listening. Telugu punctuation is removed before acoustic inference,
so the current model cannot respond specifically to T4's comma. Per-sentence
pronunciation and quality observations for all ten final recordings remain pending.

## References and delivery status

- [FastSpeech 2 paper](https://arxiv.org/abs/2006.04558)
- [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)
- [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu)
- [HiFi-GAN](https://arxiv.org/abs/2010.05646)
- [Technical report](Report/Technical_Report.pdf)

The original upstream README is preserved in `docs/UPSTREAM_README.md`; the
upstream license remains in `LICENSE`. Respect dataset/model licenses when
redistributing artifacts. The GitHub URL must still be submitted through the assignment form.
No form submission is performed by the download or training scripts.

The source ZIP intentionally omits raw datasets, credentials, account/runtime
state, temporary environments and unrelated upstream demonstration assets.
Checkpoint parents/optimizer states remain local historical backups; the released
model-only best checkpoints are the final inference models.
