# FastSpeech 2: English and Telugu

Separate text-to-speech models based on [FastSpeech 2](https://arxiv.org/abs/2006.04558)
and the [reference implementation](https://github.com/ming024/FastSpeech2).
Datasets: [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) and
[IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu), with
approximately 5 hours selected for training and 1 hour for validation per language.

## Structure

The repository root is `project/` in the assignment instructions.

```text
English/    # preprocess.py, train.py, inference.py, config.yaml, checkpoints/
Telugu/     # preprocess.py, train.py, inference.py, config.yaml, checkpoints/
Results/    # loss curves, 10 test audios and 30 duration/pitch/energy plots
Report/Technical_Report.pdf
requirements.txt
```

## Setup and inference

Use Python 3.10; run from the repository root. These commands install CPU
inference dependencies and restore the checkpoints and language-local vocoders.

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r English/setup/requirements-inference.txt
python3 English/setup/download_checkpoints.py
.venv/bin/python English/setup/prepare_final_inference.py --nltk-only
.venv/bin/python English/inference.py --test-id E1
.venv/bin/python Telugu/inference.py --test-id T1
```

Large checkpoints are available in the
[complete submission ZIP](https://github.com/ParichayMago/fastspeech2-english-telugu/releases/download/assignment-v9/fastspeech2-assignment-v9.zip)
with [SHA256SUMS](https://github.com/ParichayMago/fastspeech2-english-telugu/releases/download/assignment-v9/SHA256SUMS).
Model hashes are recorded in [checkpoint_manifest.json](English/setup/checkpoint_manifest.json).

## Results and reproduction

- [Technical report](Report/Technical_Report.pdf)
- [English results](Results/English/) and [Telugu results](Results/Telugu/)
- [Preprocessing, training, dependencies and verification commands](English/docs/REPRODUCIBILITY.md)
- [License](LICENSE)
