"""Select a cleaner 5 h + 1 h English split from all of Svarah.

Why: the original random split (English/data/splits) includes many noisy recordings.
This selection keeps the recordings with the most speech energy relative to their
silences. The original split is kept unchanged for reference.

Noise is measured on aligned audio, because an alignment-free score (loud vs quiet
frames) agreed poorly with it (rank correlation 0.37 on 400 clips). So every
eligible recording is aligned first:

    python English/select_clean_split.py candidates
    python English/preprocess.py export
    python English/preprocess.py dict
    python English/preprocess.py align --jobs 4
    python English/select_clean_split.py select
    python English/preprocess.py features
    python English/make_group_variant.py
"""
import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

import common  # noqa: F401
from common import DEFAULT_CONFIG, ENGLISH_DIR, ROOT, feature_stage_config, load_configs
from text_frontend import alignment_words

CONFIG = DEFAULT_CONFIG
SILENCE = {"", "sil", "sp", "spn"}  # MFA 3 writes silence as empty intervals
FIELDS = [
    "dataset_index", "filename", "duration", "text", "gender", "age_group",
    "primary_language", "native_place_state", "native_place_district",
]
SEED, TRAIN_HOURS, VAL_HOURS = 42, 5.0, 1.0
MIN_SECONDS, MAX_SECONDS = 1.0, 20.0


def write_manifest(path, rows, extra=()):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, FIELDS + list(extra), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stage_candidates(config):
    """Every usable Svarah recording becomes an alignment candidate."""
    from datasets import load_dataset

    dataset = load_dataset("ai4bharat/Svarah", split="test", token=True)
    dataset = dataset.remove_columns(["audio_filepath"]).add_column(
        "filename", [Path(p["path"]).name for p in
                     load_dataset("ai4bharat/Svarah", split="test", token=True)
                     .cast_column("audio_filepath", __import__("datasets").Audio(decode=False))
                     .select_columns(["audio_filepath"])["audio_filepath"]]
    )
    candidates, seen, reasons = [], set(), Counter()
    for index, row in enumerate(dataset):
        text = row["text"] or ""
        if not MIN_SECONDS <= float(row["duration"]) <= MAX_SECONDS:
            reasons["outside_length_limits"] += 1
        elif not text.strip() or not alignment_words(text):
            reasons["no_words"] += 1
        elif text in seen:
            reasons["duplicate_transcript"] += 1
        else:
            seen.add(text)
            candidates.append({
                "dataset_index": index, "filename": row["filename"],
                "duration": float(row["duration"]), "text": text, "gender": row["gender"],
                "age_group": row["age-group"], "primary_language": row["primary_language"],
                "native_place_state": row["native_place_state"],
                "native_place_district": row["native_place_district"],
            })

    out = Path(config["path"]["manifest_dir"])
    out.mkdir(parents=True, exist_ok=True)
    # The preprocessing stages read train/val manifests; all candidates go in "train".
    write_manifest(out / "train_manifest.csv", candidates)
    write_manifest(out / "val_manifest.csv", [])
    (out / "candidates_summary.json").write_text(json.dumps({
        "candidates": len(candidates),
        "hours": round(sum(c["duration"] for c in candidates) / 3600, 3),
        "excluded": reasons,
    }, indent=2), encoding="utf-8")
    print(len(candidates), "candidates,", reasons)


def aligned_snr(row, preprocessed, rate_cache={}):
    import soundfile as sf
    import tgt

    grid = preprocessed / "TextGrid" / row["gender"] / f"{row['basename']}.TextGrid"
    if not grid.exists():
        return None
    audio, rate = sf.read(ROOT / row["original_wav_path"])
    tier = tgt.io.read_textgrid(str(grid), include_empty_intervals=True).get_tier_by_name("phones")
    speech, silence = [], []
    for interval in tier._objects:
        chunk = audio[int(interval.start_time * rate):int(interval.end_time * rate)]
        (silence if interval.text.strip() in SILENCE else speech).append(chunk)
    if not speech or not silence or not sum(len(c) for c in silence):
        return None
    ratio = np.mean(np.concatenate(speech) ** 2) / (np.mean(np.concatenate(silence) ** 2) + 1e-12)
    return float(10 * np.log10(ratio))


def stage_select(config):
    data_dir = Path(config["path"]["raw_path"]).parent
    mapping_path = data_dir / "recording_mapping.csv"
    candidates_path = data_dir / "recording_mapping_candidates.csv"
    if not candidates_path.exists():
        shutil.copy(mapping_path, candidates_path)
    with open(candidates_path, encoding="utf-8", newline="") as f:
        mapping = list(csv.DictReader(f))
    preprocessed = Path(config["path"]["preprocessed_path"])

    scored = []
    for row in mapping:
        snr = aligned_snr(row, preprocessed)
        if snr is not None:
            scored.append((snr, row))
    scored.sort(key=lambda item: -item[0])

    chosen, total = [], 0.0
    for snr, row in scored:
        if total >= (TRAIN_HOURS + VAL_HOURS) * 3600:
            break
        chosen.append((snr, row))
        total += float(row["duration"])
    threshold = chosen[-1][0]

    random.Random(SEED).shuffle(chosen)
    split_of, seconds = {}, 0.0
    for snr, row in chosen:
        split_of[row["basename"]] = "val" if seconds < VAL_HOURS * 3600 else "train"
        if split_of[row["basename"]] == "val":
            seconds += float(row["duration"])

    selected = []
    for snr, row in chosen:
        selected.append({**row, "split": split_of[row["basename"]]})
    with open(mapping_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(mapping[0]))
        writer.writeheader()
        writer.writerows(sorted(selected, key=lambda r: (r["split"], r["basename"])))

    # Manifests in the same format as the original split, plus the SNR estimate.
    with open(Path(config["path"]["manifest_dir"]) / "train_manifest.csv", encoding="utf-8") as f:
        metadata = {r["filename"]: r for r in csv.DictReader(f)}
    snr_of = {row["filename"]: round(snr, 2) for snr, row in scored}
    out = Path(config["path"]["manifest_dir"])
    rows = {"train": [], "val": [], "unused": []}
    for filename, meta in metadata.items():
        base = Path(filename).stem
        rows[split_of.get(base, "unused")].append({**meta, "aligned_snr_db": snr_of.get(filename, "")})
    for name, items in rows.items():
        write_manifest(out / f"{name}_manifest.csv", items, extra=["aligned_snr_db"])

    original = set()
    for name in ("train", "val"):
        with open(ENGLISH_DIR / "data" / "splits" / f"{name}_manifest.csv", encoding="utf-8") as f:
            original |= {r["filename"] for r in csv.DictReader(f)}

    def describe(split):
        items = [(snr, row) for snr, row in chosen if split_of[row["basename"]] == split]
        values = np.array([snr for snr, _ in items])
        return {
            "recordings": len(items),
            "hours": round(sum(float(r["duration"]) for _, r in items) / 3600, 4),
            "gender": dict(Counter(r["gender"] for _, r in items)),
            "demographic_groups": len({r["align_group"] for _, r in items}),
            "aligned_snr_db_median": round(float(np.median(values)), 1),
            "share_under_20_db": round(float(np.mean(values < 20)), 3),
        }

    all_values = np.array([snr for snr, _ in scored])
    summary = {
        "dataset": "ai4bharat/Svarah",
        "random_seed": SEED,
        "method": "Highest speech-to-silence energy on aligned audio, then a random recording-level split",
        "speaker_exclusive": False,
        "length_limits_seconds": [MIN_SECONDS, MAX_SECONDS],
        "aligned_candidates": len(scored),
        "candidates_aligned_snr_db_median": round(float(np.median(all_values)), 1),
        "candidates_share_under_20_db": round(float(np.mean(all_values < 20)), 3),
        "lowest_aligned_snr_kept_db": round(threshold, 2),
        "training": describe("train"),
        "validation": describe("val"),
        "overlap_with_original_split": len({r["filename"] for _, r in chosen} & original),
        "overlap_check": "Passed" if not (
            {b for b, s in split_of.items() if s == "train"} & {b for b, s in split_of.items() if s == "val"}
        ) else "FAILED",
    }
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["candidates", "select"])
    args = parser.parse_args()
    config, _, _, _ = load_configs(CONFIG)
    config = feature_stage_config(config)
    {"candidates": stage_candidates, "select": stage_select}[args.stage](config)


if __name__ == "__main__":
    main()
