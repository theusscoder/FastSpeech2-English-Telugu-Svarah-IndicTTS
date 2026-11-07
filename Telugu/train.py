"""Train the Telugu FastSpeech 2 model.

The training loop is shared with English (English/train.py), so both languages use
the same batching, logging and checkpoint code. Only the config and symbols differ.

    python Telugu/train.py --smoke_test
    python Telugu/train.py --require_gpu
"""
import os
import runpy
import sys
from pathlib import Path

TELUGU_DIR = Path(__file__).resolve().parent
ENGLISH_DIR = TELUGU_DIR.parent / "English"

# Must be set before the model imports its symbol table.
os.environ["FASTSPEECH2_SYMBOLS"] = "telugu"

sys.path.insert(0, str(ENGLISH_DIR))
if "--config" not in sys.argv:
    sys.argv += ["--config", str(TELUGU_DIR / "config.yaml")]
runpy.run_path(str(ENGLISH_DIR / "train.py"), run_name="__main__")
