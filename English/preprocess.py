"""Svarah data preparation for FastSpeech 2.

Stages (run in order, or all of them with ``all``):
    export    write source and 22.05 kHz WAVs from the existing manifests
    dict      build the MFA pronunciation dictionary
    align     run Montreal Forced Aligner and collect the TextGrids
    features  extract mel, pitch, energy and duration, and write metadata

The train/validation split is read from English/data/splits and never recomputed.

Example:
    python English/preprocess.py export
    python English/preprocess.py align --limit 50
"""
import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import common  # noqa: F401  (adds the repository root to sys.path)
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from common import ROOT, feature_stage_config, load_configs
from text_frontend import (
    EnglishFrontend,
    acronyms_in,
    alignment_words,
    normalize_text,
    read_lexicon_variants,
    spelled_out,
)

SPLITS = {"train": "train_manifest.csv", "val": "val_manifest.csv"}
SOURCE_SAMPLE_RATE = 16000
MAPPING_NAME = "recording_mapping.csv"
MAPPING_FIELDS = [
    "split", "dataset_index", "filename", "basename", "gender", "align_group",
    "duration", "original_text", "normalized_text", "original_wav_path", "wav_path",
]
MFA_DIR = ROOT / "montreal-forced-aligner"


def read_manifests(config):
    manifest_dir = Path(config["path"]["manifest_dir"])
    rows = []
    for split, name in SPLITS.items():
        with open(manifest_dir / name, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row["split"] = split
                rows.append(row)
    return rows


def check_split(rows):
    """Stop if the manifests are not disjoint and fully usable."""
    names = Counter(r["filename"] for r in rows)
    indices = Counter(r["dataset_index"] for r in rows)
    duplicated = [n for n, c in names.items() if c > 1]
    if duplicated or max(indices.values()) > 1:
        raise ValueError(f"Duplicate recordings across manifests: {duplicated[:5]}")
    empty = [r["filename"] for r in rows if not r["text"].strip()]
    if empty:
        raise ValueError(f"Empty transcripts: {empty[:5]}")
    print(f"Split check passed: {len(rows)} unique recordings, no overlap.")


def align_group(row):
    """Folder for MFA speaker adaptation. A demographic group, not a speaker ID."""
    parts = [
        row["gender"], row["age_group"], row["primary_language"],
        row["native_place_state"], row["native_place_district"],
    ]
    return re.sub(r"[^A-Za-z0-9]+", "-", "_".join(parts)).strip("-")


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def stage_export(config, _args):
    """Write byte-exact source audio and properly resampled training audio."""
    from datasets import Audio, load_dataset

    rows = read_manifests(config)
    check_split(rows)

    dataset = load_dataset("ai4bharat/Svarah", split="test", token=True)
    dataset = dataset.cast_column("audio_filepath", Audio(decode=False))

    target_rate = config["preprocessing"]["audio"]["sampling_rate"]
    original_root = Path(config["path"]["original_wav_path"])
    raw_root = Path(config["path"]["raw_path"])
    mapping, problems = [], []

    for row in tqdm(rows, desc="Exporting"):
        filename = row["filename"]
        basename = Path(filename).stem
        record = dataset[int(row["dataset_index"])]
        audio = record["audio_filepath"]

        if Path(audio["path"]).name != filename:
            problems.append((filename, f"index points to {audio['path']}"))
            continue
        if record["text"] != row["text"]:
            problems.append((filename, "manifest text differs from the dataset"))
            continue

        group = align_group(row)
        original_path = original_root / group / filename
        original_path.parent.mkdir(parents=True, exist_ok=True)
        original_path.write_bytes(audio["bytes"])

        try:
            info = sf.info(str(original_path))
            samples, rate = sf.read(str(original_path), dtype="float32")
        except Exception as error:  # a corrupt file must be reported, not fatal
            problems.append((filename, f"unreadable audio: {error}"))
            continue
        if rate != SOURCE_SAMPLE_RATE or info.channels != 1 or len(samples) == 0:
            problems.append((filename, f"unexpected format {rate} Hz {info.channels} ch"))
            continue

        normalized = normalize_text(row["text"])
        words = alignment_words(row["text"])
        if not words:
            problems.append((filename, "no words after normalization"))
            continue

        original_path.with_suffix(".lab").write_text(" ".join(words), encoding="utf-8")

        resampled = librosa.resample(
            samples, orig_sr=rate, target_sr=target_rate, res_type="soxr_hq"
        )
        wav_path = raw_root / row["gender"] / filename
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(wav_path), np.clip(resampled, -1.0, 1.0), target_rate, subtype="PCM_16")
        # The upstream preprocessor reads raw_text from this file, so it holds the
        # untouched transcript. '|' is the metadata separator and must not appear.
        wav_path.with_suffix(".lab").write_text(
            row["text"].replace("|", " ").strip(), encoding="utf-8"
        )

        mapping.append({
            "split": row["split"], "dataset_index": row["dataset_index"],
            "filename": filename, "basename": basename, "gender": row["gender"],
            "align_group": group, "duration": row["duration"],
            "original_text": row["text"], "normalized_text": normalized,
            "original_wav_path": relative(original_path), "wav_path": relative(wav_path),
        })

    data_dir = raw_root.parent
    with open(data_dir / MAPPING_NAME, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, MAPPING_FIELDS)
        writer.writeheader()
        writer.writerows(mapping)
    with open(data_dir / "export_problems.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "problem"])
        writer.writerows(problems)

    hours = Counter()
    for item in mapping:
        hours[item["split"]] += float(item["duration"]) / 3600
    print(f"Exported {len(mapping)} of {len(rows)} recordings; {len(problems)} problems.")
    print({split: round(value, 4) for split, value in hours.items()})


def read_mapping(config):
    path = Path(config["path"]["raw_path"]).parent / MAPPING_NAME
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def stage_dict(config, _args):
    """Dictionary for the corpus: all lexicon variants, g2p for unknown words."""
    mapping = read_mapping(config)
    lexicon = read_lexicon_variants(config["path"]["lexicon_path"])
    frontend = EnglishFrontend(config["path"]["lexicon_path"])

    words, acronyms = Counter(), set()
    for item in mapping:
        words.update(alignment_words(item["original_text"]))
        acronyms |= acronyms_in(item["original_text"])

    mfa_dir = Path(config["path"]["mfa_path"])
    mfa_dir.mkdir(parents=True, exist_ok=True)
    oov_rows = []
    with open(mfa_dir / "dictionary.txt", "w", encoding="utf-8") as f:
        for word in sorted(words):
            variants = list(lexicon.get(word, []))
            if not variants:
                g2p_phones = frontend.oov_pronunciation(word)
                variants.append(g2p_phones)
                oov_rows.append((word, words[word], " ".join(g2p_phones)))
            if word in acronyms and word.isalpha():
                spelled = spelled_out(word)
                if spelled not in variants:
                    variants.append(spelled)
            for phones in variants:
                f.write(f"{word}\t{' '.join(phones)}\n")
    with open(mfa_dir / "oov_words.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "count", "g2p_pronunciation"])
        writer.writerows(oov_rows)
    print(f"{len(words)} word types; {len(oov_rows)} not in the lexicon (used g2p).")


def build_subset_corpus(config, mapping, limit):
    """Copy the first ``limit`` recordings per the seeded manifest order."""
    subset_root = Path(config["path"]["mfa_path"]) / f"subset_corpus_{limit}"
    if subset_root.exists():
        shutil.rmtree(subset_root)
    for item in mapping[:limit]:
        source = ROOT / item["original_wav_path"]
        target = subset_root / item["align_group"] / item["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        shutil.copy2(source.with_suffix(".lab"), target.with_suffix(".lab"))
    return subset_root


def stage_align(config, args):
    mapping = read_mapping(config)
    mfa_dir = Path(config["path"]["mfa_path"])
    corpus = Path(config["path"]["original_wav_path"])
    output_name = "textgrids"
    if args.limit:
        corpus = build_subset_corpus(config, mapping, args.limit)
        output_name = f"textgrids_subset_{args.limit}"
    output = mfa_dir / output_name
    output.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["MFA_ROOT_DIR"] = str(MFA_DIR / "mfa_root")
    env["MAMBA_ROOT_PREFIX"] = str(MFA_DIR / "mamba_root")
    command = [
        str(MFA_DIR / "bin" / "micromamba.exe"), "run", "-p", str(MFA_DIR / "env"),
        "mfa", "align", str(corpus), str(mfa_dir / "dictionary.txt"), args.acoustic_model,
        str(output), "--clean", "--overwrite", "--beam", "100", "--retry_beam", "400",
        "--num_jobs", str(args.jobs), "--output_format", "long_textgrid",
    ]
    print(" ".join(command))
    subprocess.run(command, check=True, env=env)

    collect_textgrids(config, mapping, output)


def collect_textgrids(config, mapping, output):
    """Copy TextGrids to where the upstream preprocessor reads them."""
    target_root = Path(config["path"]["preprocessed_path"]) / "TextGrid"
    found = 0
    for item in mapping:
        source = output / item["align_group"] / f"{item['basename']}.TextGrid"
        if source.exists():
            target = target_root / item["gender"] / f"{item['basename']}.TextGrid"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            found += 1
    print(f"Collected {found} TextGrids of {len(mapping)} recordings.")


def stage_features(config, args):
    from preprocessor.preprocessor import Preprocessor
    from sklearn.preprocessing import StandardScaler
    from text import text_to_sequence

    mapping = read_mapping(config)
    if args.limit:
        mapping = mapping[: args.limit]

    # Paths must exist before Preprocessor writes into them.
    out_dir = Path(config["path"]["preprocessed_path"])
    for name in ("mel", "pitch", "energy", "duration"):
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    preprocessor = Preprocessor(config)
    cleaners = config["preprocessing"]["text"]["text_cleaners"]
    hop = config["preprocessing"]["stft"]["hop_length"]
    rate = config["preprocessing"]["audio"]["sampling_rate"]
    n_mels = config["preprocessing"]["mel"]["n_mel_channels"]

    metadata = {"train": [], "val": []}
    kept, excluded = [], []
    pitch_scaler, energy_scaler = StandardScaler(), StandardScaler()
    frames = Counter()
    phones_seen = Counter()

    for item in tqdm(mapping, desc="Features"):
        basename, gender, split = item["basename"], item["gender"], item["split"]
        if not (out_dir / "TextGrid" / gender / f"{basename}.TextGrid").exists():
            excluded.append((item["filename"], split, "alignment_failed_no_textgrid"))
            continue
        try:
            result = preprocessor.process_utterance(gender, basename)
        except Exception as error:  # logged and excluded consistently
            excluded.append((item["filename"], split, f"feature_error:{error}"))
            continue
        if result is None:
            excluded.append((item["filename"], split, "rejected_no_voicing_or_frames"))
            continue
        info, pitch, energy, n_frames = result

        reason = validate_utterance(
            out_dir, gender, basename, info, cleaners, text_to_sequence, n_mels
        )
        if reason:
            excluded.append((item["filename"], split, reason))
            continue

        # Normalization statistics come from the training split only.
        if split == "train":
            if len(pitch):
                pitch_scaler.partial_fit(pitch.reshape(-1, 1))
            if len(energy):
                energy_scaler.partial_fit(energy.reshape(-1, 1))
        metadata[split].append(info)
        kept.append((gender, basename, split))
        frames[split] += n_frames
        phones_seen.update(info.split("|")[2].strip("{}").split())

    if not metadata["train"]:
        reasons = Counter(reason.split(":")[0] for _, _, reason in excluded)
        raise RuntimeError(f"No usable training utterances: {reasons}; first: {excluded[:1]}")
    stats = normalize_features(out_dir, kept, pitch_scaler, energy_scaler)
    write_outputs(out_dir, metadata, stats, excluded, frames, hop, rate, mapping, phones_seen)


def validate_utterance(out_dir, gender, basename, info, cleaners, to_sequence, n_mels):
    """Return a reason string if this utterance's features are inconsistent."""
    load = lambda kind: np.load(out_dir / kind / f"{gender}-{kind}-{basename}.npy")
    mel, pitch, energy, duration = load("mel"), load("pitch"), load("energy"), load("duration")
    sequence = to_sequence(info.split("|")[2], cleaners)

    if mel.ndim != 2 or mel.shape[1] != n_mels:
        return f"bad_mel_shape:{mel.shape}"
    if len(sequence) != len(duration):
        return f"phone_symbol_mismatch:{len(sequence)}vs{len(duration)}"
    if not (len(pitch) == len(energy) == len(duration)):
        return "variance_length_mismatch"
    if int(duration.sum()) != mel.shape[0]:
        return f"duration_frame_mismatch:{int(duration.sum())}vs{mel.shape[0]}"
    if (duration < 0).any():
        return "negative_duration"
    if not all(np.isfinite(a).all() for a in (mel, pitch, energy)):
        return "non_finite_features"
    return None


def normalize_features(out_dir, kept, pitch_scaler, energy_scaler):
    """Normalize every kept utterance with training statistics."""
    pitch_mean, pitch_std = float(pitch_scaler.mean_[0]), float(pitch_scaler.scale_[0])
    energy_mean, energy_std = float(energy_scaler.mean_[0]), float(energy_scaler.scale_[0])
    ranges = {"pitch": [np.inf, -np.inf], "energy": [np.inf, -np.inf]}

    for gender, basename, split in tqdm(kept, desc="Normalizing"):
        for kind, mean, std in (
            ("pitch", pitch_mean, pitch_std), ("energy", energy_mean, energy_std),
        ):
            path = out_dir / kind / f"{gender}-{kind}-{basename}.npy"
            values = (np.load(path) - mean) / std
            np.save(path, values)
            if split == "train":  # embedding bins are also set from training data
                ranges[kind][0] = min(ranges[kind][0], float(values.min()))
                ranges[kind][1] = max(ranges[kind][1], float(values.max()))

    return {
        "pitch": [*ranges["pitch"], pitch_mean, pitch_std],
        "energy": [*ranges["energy"], energy_mean, energy_std],
    }


def write_outputs(out_dir, metadata, stats, excluded, frames, hop, rate, mapping, phones_seen):
    for split, lines in metadata.items():
        with open(out_dir / f"{split}.txt", "w", encoding="utf-8") as f:
            f.write("".join(line + "\n" for line in lines))
    (out_dir / "stats.json").write_text(json.dumps(stats), encoding="utf-8")
    (out_dir / "speakers.json").write_text(
        json.dumps({"Female": 0, "Male": 1}), encoding="utf-8"
    )
    with open(out_dir / "excluded_recordings.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "split", "reason"])
        writer.writerows(excluded)

    source_hours = Counter()
    for item in mapping:
        source_hours[item["split"]] += float(item["duration"]) / 3600
    report = {
        split: {
            "selected_recordings": sum(m["split"] == split for m in mapping),
            "selected_hours": round(source_hours[split], 4),
            "usable_recordings": len(metadata[split]),
            "usable_hours_after_silence_trim": round(
                frames[split] * hop / rate / 3600, 4
            ),
        }
        for split in metadata
    }
    report["excluded_by_reason"] = Counter(
        reason.split(":")[0] for _, _, reason in excluded
    )
    report["phone_counts"] = dict(phones_seen.most_common())
    (out_dir / "preprocess_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "phone_counts"}, indent=2))


def stage_pronunciations(config, _args):
    """Record which dictionary variant MFA chose for each word in the training clips."""
    from text_frontend import ALIGNED_PRONUNCIATIONS, count_aligned_pronunciations

    out_dir = Path(config["path"]["preprocessed_path"])
    grids = []
    for line in open(out_dir / "train.txt", encoding="utf-8"):
        basename, speaker = line.split("|")[:2]
        grids.append(out_dir / "TextGrid" / speaker / f"{basename}.TextGrid")
    counts = count_aligned_pronunciations(tqdm(grids, desc="Pronunciations"))
    path = Path(config["path"]["raw_path"]).parent / ALIGNED_PRONUNCIATIONS
    path.write_text(json.dumps(counts, indent=0, ensure_ascii=False), encoding="utf-8")
    print(f"{len(counts)} word types from {len(grids)} training clips -> {path}")


STAGES = {
    "export": stage_export,
    "dict": stage_dict,
    "align": stage_align,
    "features": stage_features,
    "pronunciations": stage_pronunciations,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--limit", type=int, default=0, help="only the first N recordings")
    parser.add_argument("--jobs", type=int, default=6, help="MFA parallel jobs")
    parser.add_argument("--acoustic_model", default="english_us_arpa")
    parser.add_argument("--config", default=str(common.DEFAULT_CONFIG))
    args = parser.parse_args()

    config, _, _, _ = load_configs(args.config)
    config = feature_stage_config(config)
    for name, stage in STAGES.items():
        if args.stage in (name, "all"):
            print(f"=== {name} ===")
            stage(config, args)


if __name__ == "__main__":
    sys.exit(main())
