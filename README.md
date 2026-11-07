# FastSpeech 2 for English (Svarah) and Telugu (IndicTTS)

This repository adapts the FastSpeech 2 implementation below to two separately trained models: Indian-accented English from [AI4Bharat Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) and Telugu from [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu). Each uses about 5 h of training audio and 1 h of validation audio, with no recording overlap.

The technical report is [Report/Technical_Report.pdf](Report/Technical_Report.pdf).

## Layout

```
English/   train.py  preprocess.py  inference.py  config.yaml  checkpoints/
Telugu/    train.py  preprocess.py  inference.py  config.yaml  checkpoints/
Results/   English/ and Telugu/: training_loss.png, validation_loss.png,
           test_generated_audios/, variance_adaptor_plots/{duration,pitch,energy}/
Report/    Technical_Report.pdf
README.md  requirements.txt
```

**Language folders.** Besides the required files, each language folder holds:
- the modules its scripts import (English: `common.py`, `text_frontend.py`; Telugu: `telugu_frontend.py`, `test_sentences.py`);
- its data-preparation steps (English: `select_clean_split.py`, `make_group_variant.py`; Telugu: `inspect_dataset.py`);
- `plot_losses.py` and `export_checkpoint.py` (English only; Telugu uses them too);
- unit tests in `tests/`;
- `data/`: split manifests, recording mappings and the feature index; audio and features are regenerated;
- `logs/`: loss CSVs, plus each sentence's plotted values in `logs/inference/`.

**Repository root.** The root keeps the original implementation (`model/`, `audio/`, `text/`, `transformer/`, `utils/`, `preprocessor/`, `hifigan/`, `lexicon/`), because the language folders import it.

| | English | Telugu |
|---|---|---|
| Data | Cleanest 6 h of 1-20 s Svarah clips, random 5 h / 1 h split | Random 5 h / 1 h of the female voice |
| Conditioning | 109 demographic groups (not verified speakers) | None (single voice) |
| Inference voice | `Male-30-45-Maithili-Bihar-Darbhanga` (chosen by speech-recognition error), pitch range ×1.3 | Single voice |
| Steps | 20,000 | 25,000 |
| Final validation loss | 2.269 | 1.792 |
| Checkpoint (weights only, Git LFS) | `English/checkpoints/model_only_20000.pth.tar` | `Telugu/checkpoints/model_only_25000.pth.tar` |

## Setup

Use Python 3.10. Install PyTorch for your CUDA version, then the pinned packages:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The HiFi-GAN vocoder (`hifigan/generator_universal.pth.tar.zip`) is unzipped automatically on first use.

The checkpoints are stored with Git LFS. Run `git lfs install` before cloning, or `git lfs pull` after it. They are also attached to the [v1.0 release](https://github.com/theusscoder/FastSpeech2-English-Telugu-Svarah-IndicTTS/releases/tag/v1.0), in case the LFS download quota runs out:

```bash
curl -L -o English/checkpoints/model_only_20000.pth.tar https://github.com/theusscoder/FastSpeech2-English-Telugu-Svarah-IndicTTS/releases/download/v1.0/English_model_only_20000.pth.tar
curl -L -o Telugu/checkpoints/model_only_25000.pth.tar https://github.com/theusscoder/FastSpeech2-English-Telugu-Svarah-IndicTTS/releases/download/v1.0/Telugu_model_only_25000.pth.tar
```

## Synthesize

```bash
python English/inference.py --checkpoint English/checkpoints/model_only_20000.pth.tar --test_sentences
python Telugu/inference.py --checkpoint Telugu/checkpoints/model_only_25000.pth.tar --test_sentences
python English/inference.py --checkpoint English/checkpoints/model_only_20000.pth.tar --text "Your sentence." --id mine
python Telugu/inference.py --checkpoint Telugu/checkpoints/model_only_25000.pth.tar --text "మీ వాక్యం." --id mine
```

**Outputs:**
- audio goes to `Results/<language>/test_generated_audios/`;
- duration, pitch and energy plots go to `Results/<language>/variance_adaptor_plots/`;
- the plotted values go to `<language>/logs/inference/`.

**Voices and pitch.** The English voice is a demographic group, set with `--speaker`. The list is in `English/data/clean/preprocessed_groups/speakers.json`. `--pitch_range` widens or narrows the predicted pitch around each sentence's mean. The configs set the defaults: English ×1.3, Telugu ×1.0.

## Reproduce preprocessing and training

**Aligner.** Montreal Forced Aligner 3.4.3 needs its own environment. The scripts expect it in the git-ignored `montreal-forced-aligner/` folder, created with [micromamba](https://github.com/mamba-org/micromamba-releases/releases) saved as `montreal-forced-aligner/bin/micromamba.exe`:

```bash
montreal-forced-aligner/bin/micromamba.exe create -y -p montreal-forced-aligner/env -c conda-forge montreal-forced-aligner
MFA_ROOT_DIR=montreal-forced-aligner/mfa_root montreal-forced-aligner/bin/micromamba.exe run -p montreal-forced-aligner/env mfa model download acoustic english_us_arpa
```

Telugu has no pretrained acoustic model, so `Telugu/preprocess.py align` trains one. The trained model is included as `Telugu/data/mfa/telugu_acoustic_model.zip`.

**Data access.** Svarah requires Hugging Face access (`huggingface-cli login`).

**English**, all with `English/config.yaml`:

```bash
python English/select_clean_split.py candidates
python English/preprocess.py export
python English/preprocess.py dict
python English/preprocess.py align --jobs 4
python English/select_clean_split.py select
python English/preprocess.py features
python English/preprocess.py pronunciations
python English/make_group_variant.py
python English/train.py --require_gpu
python English/plot_losses.py
python English/export_checkpoint.py --step 20000
python English/rank_voices.py --candidates 10 --sentences 30
python English/inference.py --restore_step 20000 --test_sentences
```

`preprocess.py pronunciations` records which dictionary pronunciation the aligner chose for each word in training. Inference then asks for the most frequent one. `rank_voices.py` ranks the best-represented voices by Whisper's error rate on validation sentences.

**Telugu**, all with `Telugu/config.yaml`:

```bash
python Telugu/inspect_dataset.py
python Telugu/preprocess.py split
python Telugu/preprocess.py export
python Telugu/preprocess.py dict
python Telugu/preprocess.py align --jobs 4
python Telugu/preprocess.py features
python Telugu/train.py --require_gpu --total_step 25000
python English/plot_losses.py --config Telugu/config.yaml
python Telugu/inference.py --restore_step 25000 --test_sentences
python English/export_checkpoint.py --config Telugu/config.yaml --step 25000
```

**Evaluation** (Whisper speech recognition and mel-cepstral distortion; results go to `<language>/logs/evaluation/`; ASR needs `pip install openai-whisper`):

```bash
python English/evaluate_asr.py
python English/evaluate_mcd.py
python English/evaluate_asr.py --config Telugu/config.yaml
python English/evaluate_mcd.py --config Telugu/config.yaml
```

**Tests:**

```bash
python -m unittest discover -s English/tests
python -m unittest discover -s Telugu/tests
```

**Training options:**
- `--restore_step <step>`: resume from a checkpoint.
- `--max_minutes <minutes>`: limit a run's length.
- `--smoke_test`: a short run that also checks the checkpoint round trip.

## Changes to the original implementation

| File | Change |
|---|---|
| `model/modules.py` | Energy prediction used the pitch control value; it now uses the energy control |
| `audio/stft.py`, `audio/audio_processing.py` | Removed hardcoded `.cuda()` calls; updated `librosa` calls |
| `preprocessor/preprocessor.py` | Keeps MFA 3 silences; stops statistics reusing the previous clip's values; handles duration and frame rounding; passes the sample rate explicitly |
| `utils/model.py` | Loads checkpoints on any device; makes vocoder paths independent of the working directory |
| `model/optimizer.py` | Stores the learning rate as a plain float, so checkpoints reload on PyTorch 2.x |
| `text/symbols.py`, `text/telugu.py` | Separate Telugu symbol table, selected with `FASTSPEECH2_SYMBOLS=telugu`; the English table is unchanged |

## Credits

- **Model code:** [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (MIT licence, see `LICENSE`). This repository keeps only the parts of that implementation that the English and Telugu pipelines use; the modified files are listed above.
- **Paper:** Ren et al., *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech*, ICLR 2021.
- **Vocoder:** [HiFi-GAN](https://github.com/jik876/hifi-gan) (Kong et al., 2020), the universal generator from the reference implementation.
- **Data:** [AI4Bharat Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) and [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu).
