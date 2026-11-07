"""IndicTTS Telugu data preparation for FastSpeech 2.

Stages (run in order, or all of them with ``all``):
    split     select recordings and write the train/validation manifests
    export    write source and 22.05 kHz WAVs from the manifests
    dict      build the pronunciation dictionary from the Telugu rules
    align     train an MFA acoustic model on the selected data and align it
    features  extract mel, pitch, energy and duration, and write metadata

MFA has no pretrained Telugu acoustic model, so ``align`` trains one on this corpus.
The feature stage and TextGrid handling are shared with English/preprocess.py.

    python Telugu/inspect_dataset.py
    python Telugu/preprocess.py split
"""
import argparse
import csv
import importlib.util
import io
import json
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

TELUGU_DIR = Path(__file__).resolve().parent
ENGLISH_DIR = TELUGU_DIR.parent / "English"

# Must be set before anything imports the symbol table.
os.environ["FASTSPEECH2_SYMBOLS"] = "telugu"
sys.path[:0] = [str(ENGLISH_DIR), str(TELUGU_DIR)]

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import soundfile as sf  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from tqdm import tqdm  # noqa: E402

from common import ROOT, load_configs  # noqa: E402
from telugu_frontend import alignment_words, normalize_text, word_to_phones  # noqa: E402

DATASET = "SPRINGLab/IndicTTS_Telugu"
MFA_DIR = ROOT / "montreal-forced-aligner"
MANIFEST_FIELDS = ["dataset_index", "filename", "duration", "text", "gender"]


def load_english_preprocess():
    """Load English/preprocess.py under another name: this file is also preprocess.py."""
    spec = importlib.util.spec_from_file_location(
        "english_preprocess", ENGLISH_DIR / "preprocess.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_split(config, _args):
    selection = config["selection"]
    with open(TELUGU_DIR / "data" / "all_recordings.csv", encoding="utf-8", newline="") as f:
        recordings = list(csv.DictReader(f))

    reasons, candidates, seen_text = Counter(), [], set()
    for record in recordings:
        duration = float(record["duration"])
        if record["gender"] != selection["voice"]:
            reasons["other_voice"] += 1
        elif record["problem"]:
            reasons[record["problem"].split(":")[0]] += 1
        elif not selection["min_seconds"] <= duration <= selection["max_seconds"]:
            reasons["outside_length_limits"] += 1
        elif record["text"] in seen_text:
            # Repeated text in both sets would blur the validation result.
            reasons["duplicate_transcript"] += 1
        else:
            seen_text.add(record["text"])
            candidates.append(record)

    random.Random(selection["seed"]).shuffle(candidates)

    def take(pool, hours):
        chosen, total = [], 0.0
        while pool and total < hours * 3600:
            record = pool.pop()
            chosen.append(record)
            total += float(record["duration"])
        return chosen, total / 3600

    pool = list(candidates)
    val, val_hours = take(pool, selection["val_hours"])
    train, train_hours = take(pool, selection["train_hours"])

    overlap = {r["dataset_index"] for r in train} & {r["dataset_index"] for r in val}
    assert not overlap, overlap

    manifest_dir = Path(config["path"]["manifest_dir"])
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val), ("unused", pool)):
        with open(manifest_dir / f"{name}_manifest.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, MANIFEST_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "filename": f"te_{int(row['dataset_index']):05d}.wav"})

    summary = {
        "dataset": DATASET,
        "random_seed": selection["seed"],
        "split_method": "Random recording-level split within one gender class",
        "voice": selection["voice"],
        "length_limits_seconds": [selection["min_seconds"], selection["max_seconds"]],
        "total_recordings": len(recordings),
        "excluded_before_sampling": reasons,
        "training": {"recordings": len(train), "hours": round(train_hours, 4)},
        "validation": {"recordings": len(val), "hours": round(val_hours, 4)},
        "unused_eligible": {
            "recordings": len(pool),
            "hours": round(sum(float(r["duration"]) for r in pool) / 3600, 4),
        },
        "overlap_check": "Passed",
    }
    (manifest_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def read_manifests(config):
    rows = []
    for split in ("train", "val"):
        path = Path(config["path"]["manifest_dir"]) / f"{split}_manifest.csv"
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row["split"] = split
                rows.append(row)
    return rows


def stage_export(config, _args):
    english = load_english_preprocess()
    rows = read_manifests(config)
    english.check_split(rows)
    wanted = {int(r["dataset_index"]): r for r in rows}

    target_rate = config["preprocessing"]["audio"]["sampling_rate"]
    original_root = Path(config["path"]["original_wav_path"])
    raw_root = Path(config["path"]["raw_path"])
    mapping, problems = [], []

    snapshot = Path(snapshot_download(DATASET, repo_type="dataset"))
    index = 0
    progress = tqdm(total=len(wanted), desc="Exporting")
    for shard in sorted(snapshot.rglob("*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=32):
            data = batch.to_pydict()
            for audio, text in zip(data["audio"], data["text"]):
                row = wanted.get(index)
                index += 1
                if row is None:
                    continue
                progress.update(1)
                filename, gender = row["filename"], row["gender"]
                if (text or "").replace("\n", " ").strip() != row["text"]:
                    problems.append((filename, "manifest text differs from the dataset"))
                    continue
                original_path = original_root / gender / filename
                original_path.parent.mkdir(parents=True, exist_ok=True)
                original_path.write_bytes(audio["bytes"])
                try:
                    samples, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                except Exception as error:  # a corrupt file must be reported, not fatal
                    problems.append((filename, f"unreadable audio: {error}"))
                    continue
                if samples.ndim != 1 or len(samples) == 0:
                    problems.append((filename, "not mono or empty"))
                    continue

                words = alignment_words(row["text"])
                original_path.with_suffix(".lab").write_text(" ".join(words), encoding="utf-8")

                resampled = librosa.resample(
                    samples, orig_sr=rate, target_sr=target_rate, res_type="soxr_hq"
                )
                wav_path = raw_root / gender / filename
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(wav_path), np.clip(resampled, -1.0, 1.0), target_rate, subtype="PCM_16")
                wav_path.with_suffix(".lab").write_text(
                    row["text"].replace("|", " ").strip(), encoding="utf-8"
                )
                mapping.append({
                    "split": row["split"], "dataset_index": row["dataset_index"],
                    "filename": filename, "basename": Path(filename).stem, "gender": gender,
                    "align_group": gender, "duration": row["duration"],
                    "original_text": row["text"], "normalized_text": normalize_text(row["text"]),
                    "original_wav_path": english.relative(original_path),
                    "wav_path": english.relative(wav_path),
                })
    progress.close()

    data_dir = raw_root.parent
    with open(data_dir / english.MAPPING_NAME, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, english.MAPPING_FIELDS)
        writer.writeheader()
        writer.writerows(mapping)
    with open(data_dir / "export_problems.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "problem"])
        writer.writerows(problems)
    print(f"Exported {len(mapping)} of {len(rows)} recordings; {len(problems)} problems.")


def stage_dict(config, _args):
    english = load_english_preprocess()
    mapping = english.read_mapping(config)
    words = Counter()
    for item in mapping:
        words.update(alignment_words(item["original_text"]))
    mfa_dir = Path(config["path"]["mfa_path"])
    mfa_dir.mkdir(parents=True, exist_ok=True)
    with open(mfa_dir / "dictionary.txt", "w", encoding="utf-8") as f:
        for word in sorted(words):
            f.write(f"{word}\t{' '.join(word_to_phones(word))}\n")
    print(f"{len(words)} word types, {sum(words.values())} word tokens.")


def stage_align(config, args):
    english = load_english_preprocess()
    mapping = english.read_mapping(config)
    mfa_dir = Path(config["path"]["mfa_path"])
    output = mfa_dir / "textgrids"
    output.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["MFA_ROOT_DIR"] = str(MFA_DIR / "mfa_root_telugu")
    env["MAMBA_ROOT_PREFIX"] = str(MFA_DIR / "mamba_root")
    command = [
        str(MFA_DIR / "bin" / "micromamba.exe"), "run", "-p", str(MFA_DIR / "env"),
        "mfa", "train", str(config["path"]["original_wav_path"]),
        str(mfa_dir / "dictionary.txt"), str(mfa_dir / "telugu_acoustic_model.zip"),
        "--output_directory", str(output), "--output_format", "long_textgrid",
        "--clean", "--overwrite", "--num_jobs", str(args.jobs),
    ]
    print(" ".join(command))
    subprocess.run(command, check=True, env=env)
    english.collect_textgrids(config, mapping, output)


def stage_features(config, args):
    load_english_preprocess().stage_features(config, args)


STAGES = {
    "split": stage_split,
    "export": stage_export,
    "dict": stage_dict,
    "align": stage_align,
    "features": stage_features,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=[*STAGES, "all"])
    parser.add_argument("--limit", type=int, default=0, help="features: only the first N recordings")
    parser.add_argument("--jobs", type=int, default=4, help="MFA parallel jobs")
    args = parser.parse_args()

    config, _, _, _ = load_configs(TELUGU_DIR / "config.yaml")
    for name, stage in STAGES.items():
        if args.stage in (name, "all"):
            print(f"=== {name} ===")
            stage(config, args)


if __name__ == "__main__":
    sys.exit(main())
