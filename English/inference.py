"""Synthesize speech and record what the variance adaptor predicted.

Duration, pitch and energy are taken from the model's inference pass; no training
target is involved. Pitch and energy are also converted back to Hz and raw energy.

Examples:
    python English/inference.py --restore_step 20000 --test_sentences
    python English/inference.py --restore_step 20000 --text "Hello there." --id hello
"""
import argparse
import csv
import json
import os
from pathlib import Path

import common  # noqa: F401  (adds the repository root to sys.path)
import matplotlib
import numpy as np
import soundfile as sf
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from common import ROOT, load_configs  # noqa: E402
from model import FastSpeech2  # noqa: E402
from text import text_to_sequence  # noqa: E402
from text_frontend import ALIGNED_PRONUNCIATIONS, EnglishFrontend  # noqa: E402
from utils.model import get_vocoder, vocoder_infer  # noqa: E402
from utils.tools import device  # noqa: E402

# Text from the assignment PDF. The PDF prints E3 with a gap inside "rhythm"
# ("rhyth  m"); it is read as the word "rhythm".
TEST_SENTENCES = {
    "E1": "The antidisestablishmentarianism movement gained considerable momentum "
    "during the parliamentary proceedings.",
    "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3": "The rhythm of the orchestra changed unexpectedly.",
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, "
    "so andaroo should arrive on time.",
}


def load_model(preprocess_config, model_config, train_config, restore_step, checkpoint_path=None):
    path = (
        Path(checkpoint_path)
        if checkpoint_path
        else Path(train_config["path"]["ckpt_path"]) / f"{restore_step}.pth.tar"
    )
    checkpoint = torch.load(path, map_location=device)
    model = FastSpeech2(preprocess_config, model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint.get("step", restore_step), path


@torch.no_grad()
def synthesize(model, phones, speaker_id, cleaners, pitch_range=1.0):
    """Synthesize phones; pitch_range widens the predicted pitch around its sentence mean.

    pitch_range 1.0 is the plain model. Above 1.0, a second pass feeds the widened
    contour back as phoneme-level pitch targets with the first pass's durations, so
    timing is unchanged and only the melody is less flat.
    """
    sequence = np.array(
        text_to_sequence("{" + " ".join(phones) + "}", cleaners), dtype=np.int64
    )
    if len(sequence) != len(phones):
        raise ValueError(f"Unknown phone symbol in {phones}")

    speakers = torch.tensor([speaker_id], dtype=torch.long, device=device)
    texts = torch.from_numpy(sequence).unsqueeze(0).to(device)
    lengths = torch.tensor([len(sequence)], device=device)
    output = model(speakers, texts, lengths, len(sequence))
    pitch = output[2]

    if pitch_range != 1.0:
        durations = output[5]
        pitch = pitch.mean() + pitch_range * (pitch - pitch.mean())
        mel_lengths = durations.sum(dim=1)
        output = model(
            speakers, texts, lengths, len(sequence),
            mel_lens=mel_lengths, max_mel_len=int(mel_lengths.max()),
            p_targets=pitch, d_targets=durations,
        )

    mel_length = output[9][0].item()
    return {
        "postnet_mel": output[1][:, :mel_length].transpose(1, 2),
        "pitch_normalized": pitch[0].cpu().numpy(),
        "energy_normalized": output[3][0].cpu().numpy(),
        "log_duration": output[4][0].cpu().numpy(),
        "duration_frames": output[5][0].cpu().numpy().astype(int),
        "mel_length": mel_length,
    }


def word_labels(details, phones):
    """Word name for each phone position, for plot annotation."""
    words, cursor = [""] * len(phones), 0
    for detail in details:
        while cursor < len(phones) and phones[cursor] == "sp":
            cursor += 1
        for _ in detail["phones"]:
            words[cursor] = detail["word"]
            cursor += 1
    return words


def plot_series(values, phones, words, title, ylabel, path, kind="bar", color="#2a78b8"):
    figure, axis = plt.subplots(figsize=(max(9, 0.27 * len(phones)), 4))
    positions = np.arange(len(phones))
    if kind == "bar":
        axis.bar(positions, values, color=color, width=0.8)
    else:
        axis.plot(positions, values, color=color, marker="o", markersize=3.5, linewidth=1.2)
    axis.set_xticks(positions)
    axis.set_xticklabels(phones, rotation=90, fontsize=8)
    axis.set_xlim(-0.8, len(phones) - 0.2)
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=10, pad=58)
    axis.grid(axis="y", alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)

    # Light bands and labels mark the word each group of phones belongs to.
    start, shade = 0, False
    top = axis.get_ylim()[1]
    for index in range(1, len(words) + 1):
        if index == len(words) or words[index] != words[start]:
            if shade:
                axis.axvspan(start - 0.5, index - 0.5, color="#000000", alpha=0.04, linewidth=0)
            if words[start]:
                axis.text(
                    (start + index - 1) / 2, top, words[start], ha="center", va="bottom",
                    fontsize=7, rotation=45, color="#444444",
                )
            shade = not shade
            start = index
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_analysis(identifier, text, phones, details, result, stats, frame_ms, results_dir, data_dir):
    pitch_min, pitch_max, pitch_mean, pitch_std = stats["pitch"]
    energy_min, energy_max, energy_mean, energy_std = stats["energy"]
    pitch_hz = result["pitch_normalized"] * pitch_std + pitch_mean
    energy = result["energy_normalized"] * energy_std + energy_mean
    frames = result["duration_frames"]
    words = word_labels(details, phones)

    plots = results_dir / "variance_adaptor_plots"
    for name in ("duration", "pitch", "energy"):
        (plots / name).mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / f"{identifier}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index", "phone", "word", "duration_frames", "duration_ms",
            "log_duration_raw", "pitch_hz", "pitch_normalized", "energy", "energy_normalized",
        ])
        for i, phone in enumerate(phones):
            writer.writerow([
                i, phone, words[i], int(frames[i]), round(frames[i] * frame_ms, 2),
                float(result["log_duration"][i]), float(pitch_hz[i]),
                float(result["pitch_normalized"][i]), float(energy[i]),
                float(result["energy_normalized"][i]),
            ])
    (data_dir / f"{identifier}.json").write_text(
        json.dumps({
            "id": identifier, "text": text, "phones": phones, "words": details,
            "total_frames": int(result["mel_length"]),
            "total_seconds": round(result["mel_length"] * frame_ms / 1000, 3),
        }, indent=2),
        encoding="utf-8",
    )

    short = f"{identifier}: {text}"
    plot_series(frames * frame_ms, phones, words, f"{short}\nPredicted duration",
                "Duration (ms)", plots / "duration" / f"{identifier}_duration.png")
    plot_series(pitch_hz, phones, words, f"{short}\nPredicted pitch (F0)", "F0 (Hz)",
                plots / "pitch" / f"{identifier}_pitch.png", kind="line", color="#c8452c")
    plot_series(energy, phones, words, f"{short}\nPredicted energy", "Energy (mel frame L2)",
                plots / "energy" / f"{identifier}_energy.png", kind="line", color="#6b46a8")


def parse_args(default_config, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--checkpoint", help="explicit path, e.g. a model-only checkpoint")
    parser.add_argument("--test_sentences", action="store_true")
    parser.add_argument("--text")
    parser.add_argument("--id", default="custom")
    parser.add_argument("--speaker", help="voice; defaults to the config value")
    parser.add_argument("--pitch_range", type=float,
                        help="widen pitch around the sentence mean; defaults to the config value")
    args = parser.parse_args()
    if not args.test_sentences and not args.text:
        parser.error("give --test_sentences or --text")
    if not args.restore_step and not args.checkpoint:
        parser.error("give --restore_step or --checkpoint")
    return args


def run(args, frontend_factory, test_sentences):
    """Synthesize sentences with any frontend that provides text_to_phones."""
    preprocess_config, model_config, train_config, inference_config = load_configs(args.config)
    sentences = dict(test_sentences) if args.test_sentences else {args.id: args.text}

    out_dir = Path(preprocess_config["path"]["preprocessed_path"])
    speakers = json.loads((out_dir / "speakers.json").read_text(encoding="utf-8"))
    stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
    speaker = args.speaker or inference_config["default_speaker"]
    pitch_range = args.pitch_range or inference_config.get("pitch_range", 1.0)

    settings = preprocess_config["preprocessing"]
    rate = settings["audio"]["sampling_rate"]
    hop = settings["stft"]["hop_length"]
    frame_ms = hop / rate * 1000

    frontend = frontend_factory(preprocess_config)
    model, step, path = load_model(
        preprocess_config, model_config, train_config, args.restore_step, args.checkpoint
    )
    vocoder = get_vocoder(model_config, device)
    print(f"Loaded {path} (step {step}); voice {speaker}; device {device}")

    results_dir = Path(train_config["path"]["result_path"])
    # The plotted values and a run summary go next to the logs, so Results/ holds only
    # the audio and plots the assignment lists.
    data_dir = Path(train_config["path"]["log_path"]) / "inference"
    audio_dir = results_dir / "test_generated_audios"
    audio_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for identifier, text in sentences.items():
        phones, details = frontend.text_to_phones(text)
        result = synthesize(
            model, phones, speakers[speaker], settings["text"]["text_cleaners"], pitch_range
        )
        wav = vocoder_infer(
            result["postnet_mel"], vocoder, model_config, preprocess_config,
            lengths=[result["mel_length"] * hop],
        )[0]
        wav_path = audio_dir / f"{identifier}_{speaker}.wav"
        sf.write(str(wav_path), wav, rate, subtype="PCM_16")
        # Plot titles use the same romanized words as the labels when a script needs it.
        label = " ".join(d["word"] for d in details) if details[0].get("telugu") else text
        save_analysis(identifier, label, phones, details, result, stats, frame_ms, results_dir, data_dir)
        seconds = len(wav) / rate
        summary.append({
            "id": identifier, "text": text, "speaker": speaker, "pitch_range": pitch_range,
            "checkpoint_step": step, "seconds": round(seconds, 3),
            "wav": Path(os.path.relpath(wav_path, ROOT)).as_posix(),
            "phones": " ".join(phones),
            "words_needing_review": [
                d for d in details if d["source"] not in ("lexicon", "rule")
            ],
        })
        print(f"{identifier}: {seconds:.2f} s -> {wav_path}")
        print(f"  {' '.join(phones)}")

    # A --text run must not replace the test-sentence summary.
    name = "inference_summary.json" if args.test_sentences else f"inference_summary_{args.id}.json"
    (data_dir / name).write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    run(
        parse_args(common.DEFAULT_CONFIG),
        lambda config: EnglishFrontend(
            config["path"]["lexicon_path"],
            Path(config["path"]["raw_path"]).parent / ALIGNED_PRONUNCIATIONS,
        ),
        TEST_SENTENCES,
    )


if __name__ == "__main__":
    main()
