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
  Final samples use `en_est_0155`. Telugu uses the source `te_male` label.
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

Download [fastspeech2-assignment-v1.zip](https://github.com/ParichayMago/fastspeech2-english-telugu/releases/download/assignment-v1/fastspeech2-assignment-v1.zip)
and [SHA256SUMS](https://github.com/ParichayMago/fastspeech2-english-telugu/releases/download/assignment-v1/SHA256SUMS)
from the [assignment release](https://github.com/ParichayMago/fastspeech2-english-telugu/releases/tag/assignment-v1).
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
and restore only these three hash-verified files. Use `--archive /path/to/fastspeech2-assignment-v1.zip`
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
The pause feature builder below requires an unbuilt destination and refuses to
overwrite completed features:

```bash
python setup/build_learned_pause_features.py
python setup/test_learned_pause.py
```

See `Report/Alignment_Gap_Fix.txt` and `Report/Learned_Pause_Pilot.txt` for target
verification. Older preprocessing/training notes are historical; the report and
`gap-fixed-v2` rebuild summary provide the final feature counts/hours.

## Training

Training requires CUDA and prepared features; the scripts do not rent machines.
Each language entry point fixes its language and requires an isolated run ID.
Use a new ID with `--fresh` to initialize new weights. Without `--fresh`, the
runner can resume that run's latest checkpoint, including optimizer state.
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
historical local supervisors supplied credit checks, verified backups and Pod
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
