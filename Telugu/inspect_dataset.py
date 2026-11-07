"""Measure every IndicTTS Telugu recording and write Telugu/data/all_recordings.csv.

Audio is read from the Parquet shards one batch at a time and only its header is
parsed, so the dataset is never decoded into memory.

    python Telugu/inspect_dataset.py
"""
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from telugu_frontend import has_latin, normalize_text, word_to_phones, alignment_words  # noqa: E402

DATASET = "SPRINGLab/IndicTTS_Telugu"
GENDERS = {0: "Female", 1: "Male"}  # order from the dataset's ClassLabel names
DATA_DIR = Path(__file__).resolve().parent / "data"


def text_problem(text):
    """Why this transcript cannot be converted to phones, or an empty string."""
    if not text or not text.strip():
        return "empty_text"
    if has_latin(text):
        return "latin_text"
    words = alignment_words(text)
    if not words:
        return "no_words"
    try:
        for word in words:
            word_to_phones(word)
    except ValueError as error:
        return f"unsupported_character:{str(error)[:40]}"
    return ""


def main():
    snapshot = Path(snapshot_download(DATASET, repo_type="dataset"))
    shards = sorted(snapshot.rglob("*.parquet"))
    print("Shards:", [s.name for s in shards])

    rows, index = [], 0
    for shard in shards:
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=64):
            data = batch.to_pydict()
            for audio, text, gender in zip(data["audio"], data["text"], data["gender"]):
                info = sf.info(io.BytesIO(audio["bytes"]))
                rows.append({
                    "dataset_index": index,
                    "shard": shard.name,
                    "source_name": Path(audio.get("path") or "").name,
                    "gender": GENDERS[gender],
                    "duration": info.frames / info.samplerate,
                    "sample_rate": info.samplerate,
                    "channels": info.channels,
                    "subtype": info.subtype,
                    "format": info.format,
                    "n_characters": len(text or ""),
                    "problem": text_problem(text),
                    "text": (text or "").replace("\n", " ").strip(),
                })
                index += 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "all_recordings.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    durations = np.array([r["duration"] for r in rows])
    summary = {
        "recordings": len(rows),
        "hours": round(float(durations.sum()) / 3600, 3),
        "formats": Counter(
            f"{r['format']} {r['sample_rate']} Hz {r['channels']} ch {r['subtype']}" for r in rows
        ).most_common(),
        "by_gender": {
            g: {
                "recordings": sum(r["gender"] == g for r in rows),
                "hours": round(sum(r["duration"] for r in rows if r["gender"] == g) / 3600, 3),
            }
            for g in GENDERS.values()
        },
        "duration_percentiles_s": dict(zip(
            ["min", "p5", "p50", "p95", "p99", "max"],
            np.round(np.percentile(durations, [0, 5, 50, 95, 99, 100]), 2).tolist(),
        )),
        "hours_by_length": {
            f"{low}-{high}s": round(float(
                durations[(durations >= low) & (durations < high)].sum()
            ) / 3600, 3)
            for low, high in [(0, 1), (1, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 60), (60, 10000)]
        },
        "problems": Counter(r["problem"].split(":")[0] for r in rows if r["problem"]).most_common(),
        "with_digits": sum(any(c.isdigit() for c in r["text"]) for r in rows),
        "duplicate_transcripts": len(rows) - len({r["text"] for r in rows}),
    }
    (DATA_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    examples = [r for r in rows if r["problem"]][:6]
    for example in examples:
        print(example["problem"], "|", example["text"][:80])


if __name__ == "__main__":
    main()
