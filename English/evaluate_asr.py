"""Intelligibility check with automatic speech recognition (OpenAI Whisper).

Whisper transcribes:
- the test sentences in Results/<language>/test_generated_audios/;
- a fixed random sample of validation clips, both the real recordings and the same
  text synthesized by the model (its own durations, pitch and energy).

Scoring the real recordings shows how well Whisper itself copes with the speakers,
accent and language, so the synthetic scores can be read against that reference.
Scores: word error rate (WER) and character error rate (CER, spaces ignored), summed
over all clips. English text is normalized with Whisper's English normalizer; Telugu
only loses punctuation, since Whisper's basic normalizer strips Telugu vowel signs.

Optional dependency, not needed for training or inference:
    pip install openai-whisper

    python English/evaluate_asr.py
    python English/evaluate_asr.py --config Telugu/config.yaml
"""
import argparse
import csv
import json
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import yaml

ENGLISH_DIR = Path(__file__).resolve().parent
ROOT = ENGLISH_DIR.parent
sys.path.insert(0, str(ENGLISH_DIR))


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, r in enumerate(reference, 1):
        current = [i]
        for j, h in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def telugu_normalize(text):
    text = unicodedata.normalize("NFC", text)
    text = "".join(" " if unicodedata.category(c).startswith(("P", "S")) else c for c in text)
    return re.sub(r"\s+", " ", text).strip().lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ENGLISH_DIR / "config.yaml"))
    parser.add_argument("--checkpoint", help="weights to evaluate; default: the latest saved step")
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--samples", type=int, default=100, help="validation clips to transcribe")
    parser.add_argument("--whisper_model", default="small")
    parser.add_argument("--whisper_root", help="where Whisper models are stored or downloaded")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        language = yaml.safe_load(f)["preprocess"]["preprocessing"]["text"]["language"]
    if language == "te":
        os.environ["FASTSPEECH2_SYMBOLS"] = "telugu"

    import soundfile as sf
    import soxr
    import torch
    import whisper
    from tqdm import tqdm
    from whisper.normalizers import EnglishTextNormalizer

    import common
    from common import load_configs
    from inference import load_model, synthesize
    from utils.model import get_vocoder, vocoder_infer
    from utils.tools import device

    if language == "te":
        sys.path.insert(0, str(ROOT / "Telugu"))
        from test_sentences import TEST_SENTENCES
        normalize = telugu_normalize
        sentence_ids = list(TEST_SENTENCES)
    else:
        from inference import TEST_SENTENCES
        normalize = EnglishTextNormalizer()
        sentence_ids = list(TEST_SENTENCES)

    preprocess_config, model_config, train_config, inference_config = load_configs(args.config)
    data_dir = Path(preprocess_config["path"]["preprocessed_path"])
    raw_dir = Path(preprocess_config["path"]["raw_path"])
    settings = preprocess_config["preprocessing"]
    rate, hop = settings["audio"]["sampling_rate"], settings["stft"]["hop_length"]

    asr = whisper.load_model(args.whisper_model, device=device, download_root=args.whisper_root)

    def transcribe(wav, wav_rate):
        audio = soxr.resample(wav.astype(np.float32), wav_rate, 16000)
        result = asr.transcribe(audio, language=language, task="transcribe", temperature=0.0,
                                beam_size=5, condition_on_previous_text=False,
                                fp16=device.type == "cuda")
        return result["text"].strip()

    def score(reference, hypothesis):
        ref, hyp = normalize(reference), normalize(hypothesis)
        return {
            "word_errors": edit_distance(ref.split(), hyp.split()), "words": len(ref.split()),
            "char_errors": edit_distance(ref.replace(" ", ""), hyp.replace(" ", "")),
            "chars": len(ref.replace(" ", "")),
        }

    def rates(rows):
        words = sum(r["words"] for r in rows)
        chars = sum(r["chars"] for r in rows)
        return {"wer": round(sum(r["word_errors"] for r in rows) / words, 4),
                "cer": round(sum(r["char_errors"] for r in rows) / chars, 4),
                "clips": len(rows), "reference_words": words}

    out_dir = Path(train_config["path"]["log_path"]) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Test sentences: the generated files in Results/.
    audio_dir = Path(train_config["path"]["result_path"]) / "test_generated_audios"
    speaker = inference_config["default_speaker"]
    test_rows = []
    for identifier in sentence_ids:
        wav, wav_rate = sf.read(audio_dir / f"{identifier}_{speaker}.wav")
        text = transcribe(wav, wav_rate)
        test_rows.append({"id": identifier, "reference": TEST_SENTENCES[identifier], "transcript": text,
                          **score(TEST_SENTENCES[identifier], text)})
        print(identifier, "|", text)

    # Validation sample: real recording and synthesized speech for the same text.
    if not args.checkpoint and not args.restore_step:
        args.restore_step = max(int(p.name.split(".")[0]) for p in Path(train_config["path"]["ckpt_path"]).glob("*.pth.tar")
                                if p.name.split(".")[0].isdigit())
    model, step, path = load_model(preprocess_config, model_config, train_config, args.restore_step, args.checkpoint)
    vocoder = get_vocoder(model_config, device)
    speakers = json.loads((data_dir / "speakers.json").read_text(encoding="utf-8"))
    lines = (data_dir / "val.txt").read_text(encoding="utf-8").splitlines()
    sample = random.Random(42).sample(lines, min(args.samples, len(lines)))

    real_rows, synthetic_rows, val_rows = [], [], []
    for line in tqdm(sample, desc="Validation clips"):
        basename, voice, phones, reference = line.split("|")
        gender = "Male" if voice.startswith("Male") else "Female"
        recording, recording_rate = sf.read(raw_dir / gender / f"{basename}.wav")
        real_text = transcribe(recording, recording_rate)

        result = synthesize(model, phones.strip("{}").split(), speakers[voice], settings["text"]["text_cleaners"],
                            inference_config.get("pitch_range", 1.0))
        synthetic = vocoder_infer(result["postnet_mel"], vocoder, model_config, preprocess_config,
                                  lengths=[result["mel_length"] * hop])[0] / 32768.0
        synthetic_text = transcribe(synthetic, rate)

        real, synth = score(reference, real_text), score(reference, synthetic_text)
        real_rows.append(real)
        synthetic_rows.append(synth)
        val_rows.append({"basename": basename, "reference": reference, "real_transcript": real_text,
                         "synthetic_transcript": synthetic_text,
                         "real_word_errors": real["word_errors"], "synthetic_word_errors": synth["word_errors"],
                         "words": real["words"]})

    for name, rows in (("asr_test_sentences.csv", test_rows), ("asr_validation_sample.csv", val_rows)):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "asr": f"OpenAI Whisper {args.whisper_model}, language {language}, beam 5, temperature 0",
        "checkpoint": Path(os.path.relpath(path, common.ROOT)).as_posix(),
        "pitch_range": inference_config.get("pitch_range", 1.0),
        "test_sentences": {
            "overall": rates(test_rows),
            "per_sentence": {r["id"]: {"wer": round(r["word_errors"] / r["words"], 3),
                                       "cer": round(r["char_errors"] / r["chars"], 3),
                                       "transcript": r["transcript"]} for r in test_rows},
        },
        "validation_sample": {"seed": 42, "real_recordings": rates(real_rows),
                              "synthesized": rates(synthetic_rows)},
    }
    (out_dir / "asr_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
