"""Mel-cepstral distortion (MCD) of a trained model on its validation set.

For every validation clip, three signals are compared with the original recording
(trimmed to the aligned speech, as in preprocessing):

- full system: phonemes -> model with its own durations, pitch and energy -> HiFi-GAN
- vocoder only: the clip's real mel-spectrogram -> HiFi-GAN (the lower bound the
  vocoder allows; it isolates the vocoder's share of the distortion)

Mel-cepstra: WORLD spectral envelope (pyworld DIO + StoneMask + CheapTrick) coded to
25 mel-cepstral coefficients; c0 (energy) is excluded. Frames are matched with DTW,
because the free-running model's timing differs from the recording.
MCD = 10 / ln(10) * sqrt(2) * mean over the DTW path of the Euclidean distance (dB).
Values are comparable within this project; other papers use other analysis settings.

    python English/evaluate_mcd.py
    python English/evaluate_mcd.py --config Telugu/config.yaml
"""
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ENGLISH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGLISH_DIR))

SILENCE = {"", "sil", "sp", "spn"}
ORDER = 25
K = 10 / math.log(10) * math.sqrt(2)


def mel_cepstrum(wav, rate):
    import pyworld

    wav = wav.astype(np.float64)
    f0, times = pyworld.dio(wav, rate, frame_period=5.0)
    f0 = pyworld.stonemask(wav, f0, times, rate)
    envelope = pyworld.cheaptrick(wav, f0, times, rate)
    return pyworld.code_spectral_envelope(envelope, rate, ORDER)[:, 1:]  # drop c0


def mcd(reference, test):
    import librosa

    _, path = librosa.sequence.dtw(reference.T, test.T, metric="euclidean")
    diff = reference[path[:, 0]] - test[path[:, 1]]
    return float(K * np.mean(np.sqrt(np.sum(diff ** 2, axis=1))))


def speech_span(textgrid_path):
    import tgt

    tier = tgt.io.read_textgrid(str(textgrid_path), include_empty_intervals=True).get_tier_by_name("phones")
    spoken = [i for i in tier._objects if i.text.strip() not in SILENCE]
    return spoken[0].start_time, spoken[-1].end_time


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ENGLISH_DIR / "config.yaml"))
    parser.add_argument("--checkpoint", help="weights to evaluate; default: the latest saved step")
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="only the first N validation clips")
    args = parser.parse_args()

    # The Telugu symbol table must be chosen before the model and text modules load.
    with open(args.config, encoding="utf-8") as f:
        if yaml.safe_load(f)["preprocess"]["preprocessing"]["text"]["language"] == "te":
            os.environ["FASTSPEECH2_SYMBOLS"] = "telugu"

    import soundfile as sf
    import torch
    from tqdm import tqdm

    import common  # noqa: F401
    from common import feature_stage_config, load_configs
    from inference import load_model, synthesize
    from utils.model import get_vocoder, vocoder_infer
    from utils.tools import device

    preprocess_config, model_config, train_config, inference_config = load_configs(args.config)
    pitch_range = inference_config.get("pitch_range", 1.0)
    data_dir = Path(preprocess_config["path"]["preprocessed_path"])
    features_dir = Path(feature_stage_config(preprocess_config)["path"]["preprocessed_path"])
    raw_dir = Path(preprocess_config["path"]["raw_path"])
    settings = preprocess_config["preprocessing"]
    rate, hop = settings["audio"]["sampling_rate"], settings["stft"]["hop_length"]
    cleaners = settings["text"]["text_cleaners"]

    if not args.checkpoint and not args.restore_step:
        steps = [int(p.name.split(".")[0]) for p in Path(train_config["path"]["ckpt_path"]).glob("*.pth.tar")
                 if p.name.split(".")[0].isdigit()]
        args.restore_step = max(steps)
    model, step, path = load_model(preprocess_config, model_config, train_config,
                                   args.restore_step, args.checkpoint)
    vocoder = get_vocoder(model_config, device)
    speakers = json.loads((data_dir / "speakers.json").read_text(encoding="utf-8"))
    print(f"Evaluating {path} (step {step}) on {device}")

    lines = (data_dir / "val.txt").read_text(encoding="utf-8").splitlines()
    if args.limit:
        lines = lines[: args.limit]

    rows = []
    for line in tqdm(lines, desc="MCD"):
        basename, speaker, phones, _ = line.split("|")
        phones = phones.strip("{}").split()
        gender = "Male" if speaker.startswith("Male") else "Female"

        start, end = speech_span(features_dir / "TextGrid" / gender / f"{basename}.TextGrid")
        recording, recording_rate = sf.read(raw_dir / gender / f"{basename}.wav")
        assert recording_rate == rate
        recording = recording[int(rate * start):int(rate * end)]

        real_mel = np.load(data_dir / "mel" / f"{speaker}-mel-{basename}.npy").T  # (80, frames)
        real_mel = torch.from_numpy(real_mel).float().unsqueeze(0).to(device)
        copy = vocoder_infer(real_mel, vocoder, model_config, preprocess_config,
                             lengths=[real_mel.shape[2] * hop])[0] / 32768.0

        result = synthesize(model, phones, speakers[speaker], cleaners, pitch_range)
        synthetic = vocoder_infer(result["postnet_mel"], vocoder, model_config, preprocess_config,
                                  lengths=[result["mel_length"] * hop])[0] / 32768.0

        reference = mel_cepstrum(recording, rate)
        rows.append({
            "basename": basename,
            "speaker": speaker,
            "recording_seconds": round(len(recording) / rate, 3),
            "synthesized_seconds": round(len(synthetic) / rate, 3),
            "mcd_vocoder_only": round(mcd(reference, mel_cepstrum(copy, rate)), 3),
            "mcd_full_system": round(mcd(reference, mel_cepstrum(synthetic, rate)), 3),
        })

    out_dir = Path(train_config["path"]["log_path"]) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "mcd_per_utterance.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def describe(key):
        values = np.array([r[key] for r in rows])
        return {"mean": round(float(values.mean()), 3), "std": round(float(values.std()), 3),
                "median": round(float(np.median(values)), 3)}

    ratio = np.array([r["synthesized_seconds"] / r["recording_seconds"] for r in rows])
    summary = {
        "checkpoint": Path(os.path.relpath(path, common.ROOT)).as_posix(),
        "step": step,
        "pitch_range": pitch_range,
        "validation_clips": len(rows),
        "mcd_db_vocoder_only": describe("mcd_vocoder_only"),
        "mcd_db_full_system": describe("mcd_full_system"),
        "synthesized_to_recording_length_ratio": {
            "median": round(float(np.median(ratio)), 3),
            "p10": round(float(np.percentile(ratio, 10)), 3),
            "p90": round(float(np.percentile(ratio, 90)), 3),
        },
        "method": "WORLD (DIO, StoneMask, CheapTrick), 25 coded mel-cepstra without c0, DTW, "
                  "MCD = 10/ln10 * sqrt(2) * mean Euclidean distance",
    }
    (out_dir / "mcd_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
