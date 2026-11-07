"""Shared paths and configuration loading for the English experiment."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
ENGLISH_DIR = ROOT / "English"
DEFAULT_CONFIG = ENGLISH_DIR / "config.yaml"

# English/ stays first on sys.path; the repository root provides model, audio, etc.
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))


def load_configs(path=DEFAULT_CONFIG):
    """Return (preprocess, model, train, inference) configs with absolute paths."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for section in ("preprocess", "train"):
        paths = config[section]["path"]
        for key, value in paths.items():
            paths[key] = str((ROOT / value).resolve())
    return config["preprocess"], config["model"], config["train"], config["inference"]


def feature_stage_config(preprocess_config):
    """The preprocess config as the data stages see it.

    The final English model trains on features keyed by demographic group, made by
    linking the gender-keyed features that preprocess.py writes. When a config has
    `features_path`, the data stages write and read there instead of preprocessed_path.
    """
    paths = preprocess_config["path"]
    if "features_path" not in paths:
        return preprocess_config
    return {**preprocess_config, "path": {**paths, "preprocessed_path": paths["features_path"]}}
