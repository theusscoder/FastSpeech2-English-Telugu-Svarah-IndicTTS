"""Link the English features under demographic-group names for training.

A group is recordings that share gender, age group, primary language, state and
district: 110 groups for 117 documented speakers. Groups are NOT verified speakers.
preprocess.py writes features keyed by gender into `features_path`. They are reused
without copying: every .npy file is hard-linked under the group name into
`preprocessed_path`, so the upstream Dataset loader works unchanged.

    python English/make_group_variant.py
    python English/train.py --require_gpu
"""
import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import common  # noqa: F401
from common import ENGLISH_DIR, feature_stage_config, load_configs

KINDS = ("mel", "pitch", "energy", "duration")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ENGLISH_DIR / "config.yaml"))
    parser.add_argument("--target", default=str(ENGLISH_DIR / "config.yaml"))
    args = parser.parse_args()
    source_config, _, _, _ = load_configs(args.source)
    target_config, _, _, _ = load_configs(args.target)
    source = Path(feature_stage_config(source_config)["path"]["preprocessed_path"])
    target = Path(target_config["path"]["preprocessed_path"])
    for kind in KINDS:
        (target / kind).mkdir(parents=True, exist_ok=True)

    mapping_path = Path(source_config["path"]["raw_path"]).parent / "recording_mapping.csv"
    with open(mapping_path, encoding="utf-8", newline="") as f:
        group_of = {row["basename"]: row["align_group"] for row in csv.DictReader(f)}

    counts = {"train": Counter(), "val": Counter()}
    for split in ("train", "val"):
        lines = []
        for line in open(source / f"{split}.txt", encoding="utf-8"):
            basename, gender, phones, raw_text = line.rstrip("\n").split("|")
            group = group_of[basename]
            for kind in KINDS:
                link = target / kind / f"{group}-{kind}-{basename}.npy"
                if not link.exists():
                    os.link(source / kind / f"{gender}-{kind}-{basename}.npy", link)
            lines.append("|".join([basename, group, phones, raw_text]))
            counts[split][group] += 1
        (target / f"{split}.txt").write_text("".join(l + "\n" for l in lines), encoding="utf-8")

    shutil.copy(source / "stats.json", target / "stats.json")
    groups = sorted(set(counts["train"]) | set(counts["val"]))
    (target / "speakers.json").write_text(
        json.dumps({group: i for i, group in enumerate(groups)}, indent=0), encoding="utf-8"
    )

    unseen = [g for g in counts["val"] if g not in counts["train"]]
    largest_female = max(
        (g for g in counts["train"] if g.startswith("Female")), key=counts["train"].get
    )
    summary = {
        "groups": len(groups),
        "train_utterances_per_group": {
            "min": min(counts["train"].values()),
            "median": sorted(counts["train"].values())[len(counts["train"]) // 2],
            "max": max(counts["train"].values()),
        },
        "validation_groups_unseen_in_training": unseen,
        "validation_utterances_in_unseen_groups": sum(counts["val"][g] for g in unseen),
        "largest_female_group": [largest_female, counts["train"][largest_female]],
    }
    (target / "group_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
