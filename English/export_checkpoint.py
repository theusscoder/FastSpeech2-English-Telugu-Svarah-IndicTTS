"""Save a model-only copy of a checkpoint for submission (no optimizer state).

    python English/export_checkpoint.py --step 20000
Then: python English/inference.py --checkpoint English/checkpoints/model_only_20000.pth.tar --test_sentences
"""
import argparse
from pathlib import Path

import common  # noqa: F401
import torch

from common import load_configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--config", default=str(common.DEFAULT_CONFIG))
    args = parser.parse_args()

    _, _, train_config, _ = load_configs(args.config)
    ckpt_dir = Path(train_config["path"]["ckpt_path"])
    checkpoint = torch.load(ckpt_dir / f"{args.step}.pth.tar", map_location="cpu")
    target = ckpt_dir / f"model_only_{args.step}.pth.tar"
    torch.save({"model": checkpoint["model"], "step": checkpoint["step"]}, target)
    print(f"Wrote {target} ({target.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
