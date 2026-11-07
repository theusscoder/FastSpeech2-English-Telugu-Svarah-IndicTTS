# FastSpeech 2 Text-to-Speech for English and Telugu

This project teaches a computer to **read text aloud** in two languages, using the FastSpeech 2 model:

- **English** with an Indian accent, trained on [AI4Bharat Svarah](https://huggingface.co/datasets/ai4bharat/Svarah)
- **Telugu**, trained on [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu)

The two languages are separate models. Each was trained on about **5 hours** of speech and checked on another **1 hour** that it never trained on.

**The full write-up is in [Report/Technical_Report.pdf](Report/Technical_Report.pdf).**

## Results at a glance

| | English | Telugu |
|---|---|---|
| Training data | The cleanest 6 hours of Svarah (5 h training, 1 h validation) | 5 h training and 1 h validation from one female voice |
| Training | 20,000 steps (1.3 hours on a laptop GPU) | 25,000 steps (1.8 hours) |
| Final validation loss | 2.269 | 1.792 |
| How clear it sounds | A speech recognizer understood about 80% of the words (real recordings: 93%) | Easy to understand; dates and times read correctly in Telugu |

## Listen to the results

| Language | Audio | Duration, pitch and energy plots |
|---|---|---|
| English | [Results/English/test_generated_audios](Results/English/test_generated_audios) | [Results/English/variance_adaptor_plots](Results/English/variance_adaptor_plots) |
| Telugu | [Results/Telugu/test_generated_audios](Results/Telugu/test_generated_audios) | [Results/Telugu/variance_adaptor_plots](Results/Telugu/variance_adaptor_plots) |

The loss curves are `training_loss.png` and `validation_loss.png` in each `Results/<language>/` folder.

## What is in this repository

```
English/            English model: train.py, preprocess.py, inference.py, config.yaml
  checkpoints/      the trained English model (model_only_20000.pth.tar)
Telugu/             Telugu model: train.py, preprocess.py, inference.py, config.yaml
  checkpoints/      the trained Telugu model (model_only_25000.pth.tar)
Results/            generated audio, loss curves and duration/pitch/energy plots
Report/             Technical_Report.pdf
model/ transformer/ audio/ text/ utils/ preprocessor/ hifigan/ lexicon/ dataset.py
                    the FastSpeech 2 code from ming024/FastSpeech2, which both languages use
                    (some files fixed or extended, see below)
requirements.txt    Python packages
```

Each language folder also contains:
- its helper code;
- `data/`: the list of which recordings were used, and the settings the model needs;
- `logs/`: training logs and evaluation results;
- `tests/`.

## Quick start: make the models speak

**1. Get the code.** The trained models are large files stored with [Git LFS](https://git-lfs.com), so install Git LFS before cloning:

```bash
git lfs install
```
```bash
git clone https://github.com/theusscoder/FastSpeech2-English-Telugu-Svarah-IndicTTS.git
```
```bash
cd FastSpeech2-English-Telugu-Svarah-IndicTTS
```

**2. Install Python packages** (Python 3.10). First install [PyTorch](https://pytorch.org/get-started/locally/) for your computer, then:

```bash
pip install -r requirements.txt
```

**3. Generate the test sentences.** A GPU is used if available; a CPU also works.

```bash
python English/inference.py --checkpoint English/checkpoints/model_only_20000.pth.tar --test_sentences
```
```bash
python Telugu/inference.py --checkpoint Telugu/checkpoints/model_only_25000.pth.tar --test_sentences
```

The audio appears in `Results/<language>/test_generated_audios/`, and the plots in `Results/<language>/variance_adaptor_plots/`.

**Say your own sentence:**

```bash
python English/inference.py --checkpoint English/checkpoints/model_only_20000.pth.tar --text "Hello, how are you?" --id hello
```
```bash
python Telugu/inference.py --checkpoint Telugu/checkpoints/model_only_25000.pth.tar --text "మీరు ఎలా ఉన్నారు?" --id hello
```

**Optional controls:**
- `--pitch_range` makes the melody livelier (above 1) or flatter (below 1).
- `--speaker` picks another English voice. The list is in `English/data/clean/preprocessed_groups/speakers.json`.

## How it works

1. **Text to sounds.** The text is turned into phonemes, the individual speech sounds.
   - **English** uses a pronunciation dictionary. It reads numbers, dates and times as words, and picks the right stress for words like "conduct" (noun vs verb).
   - **Telugu** uses rules we wrote for Telugu script, including Telugu number words.
2. **Sounds to a spectrogram.** FastSpeech 2 predicts how long, how high (pitch) and how loud (energy) each sound should be, then draws a spectrogram, a picture of the sound.
3. **Spectrogram to audio.** The pretrained HiFi-GAN vocoder turns the picture into a waveform.

To learn sound durations, each recording was lined up with its text by the Montreal Forced Aligner. No Telugu aligner existed, so we trained one on the Telugu data.

## What we added or fixed

- **Telugu support:** a Telugu phonemizer, Telugu number reading, a separate Telugu symbol set (`text/telugu.py`), and a Telugu aligner model.
- **Better English data:** the 6 cleanest hours of Svarah, chosen by measuring how much louder speech is than the background.
- **Clearer English speech at inference:**
  - each word uses the pronunciation the speakers used most often in training;
  - the voice was chosen by how well a speech recognizer understood it;
  - the melody is widened to sound less monotonous.
- **Fixes to the original code:**

| File | Fix |
|---|---|
| `model/modules.py` | The loudness (energy) control was wrongly using the pitch setting |
| `audio/stft.py`, `audio/audio_processing.py` | Runs without a GPU; works with current `librosa` |
| `preprocessor/preprocessor.py` | Keeps the pauses that the new aligner writes; fixes a statistics bug and rounding errors |
| `utils/model.py` | Loads models on any computer; unzips the vocoder automatically |
| `model/optimizer.py` | Saved models reload in PyTorch 2.x |

## Checking the results

- **Speech recognition (Whisper)** measures how many words are understood.
- **Mel-cepstral distortion** measures how close the generated speech is to the real recordings.
- **Tests:** 20 automated tests check the main pieces.

The evaluation results are in `English/logs/evaluation/` and `Telugu/logs/evaluation/`.

```bash
python -m unittest discover -s English/tests
```
```bash
python -m unittest discover -s Telugu/tests
```

<details>
<summary><b>Rebuild everything from scratch (data preparation and training)</b></summary>

You need:
- **Hugging Face access to Svarah:** accept its terms on the dataset page, then run `huggingface-cli login`.
- **Montreal Forced Aligner 3.4.3**, in its own environment. The scripts expect it in a `montreal-forced-aligner/` folder, created with [micromamba](https://github.com/mamba-org/micromamba-releases) saved as `montreal-forced-aligner/bin/micromamba.exe`:

```bash
montreal-forced-aligner/bin/micromamba.exe create -y -p montreal-forced-aligner/env -c conda-forge montreal-forced-aligner
```
```bash
MFA_ROOT_DIR=montreal-forced-aligner/mfa_root montreal-forced-aligner/bin/micromamba.exe run -p montreal-forced-aligner/env mfa model download acoustic english_us_arpa
```

**English** (settings in `English/config.yaml`):

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
```

**Telugu** (settings in `Telugu/config.yaml`):

```bash
python Telugu/inspect_dataset.py
python Telugu/preprocess.py split
python Telugu/preprocess.py export
python Telugu/preprocess.py dict
python Telugu/preprocess.py align --jobs 4
python Telugu/preprocess.py features
python Telugu/train.py --require_gpu --total_step 25000
python English/plot_losses.py --config Telugu/config.yaml
python English/export_checkpoint.py --config Telugu/config.yaml --step 25000
```

**Evaluation** (speech recognition needs `pip install openai-whisper`):

```bash
python English/evaluate_asr.py
python English/evaluate_mcd.py
python English/evaluate_asr.py --config Telugu/config.yaml
python English/evaluate_mcd.py --config Telugu/config.yaml
```

**Training options:**
- `--restore_step <step>` resumes a run.
- `--max_minutes <minutes>` limits its length.
- `--smoke_test` runs a quick check.

</details>

## Credits

- **Model code:** [ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), MIT licence (see `LICENSE`). Only the parts this project uses are kept.
- **Paper:** Ren et al., *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech*, ICLR 2021.
- **Vocoder:** [HiFi-GAN](https://github.com/jik876/hifi-gan), Kong et al., 2020.
- **Data:** [AI4Bharat Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) and [IndicTTS Telugu](https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu).
