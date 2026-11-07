"""Synthesize Telugu speech and record the variance adaptor's predictions.

    python Telugu/inference.py --restore_step 50000 --test_sentences
"""
import importlib.util
import os
import sys
from pathlib import Path

TELUGU_DIR = Path(__file__).resolve().parent
ENGLISH_DIR = TELUGU_DIR.parent / "English"

# Must be set before the model imports its symbol table.
os.environ["FASTSPEECH2_SYMBOLS"] = "telugu"
sys.path[:0] = [str(ENGLISH_DIR), str(TELUGU_DIR)]

from telugu_frontend import TeluguFrontend  # noqa: E402
from test_sentences import TEST_SENTENCES  # noqa: E402


def load_shared_inference():
    """Load English/inference.py under another name: this file is also inference.py."""
    spec = importlib.util.spec_from_file_location(
        "shared_inference", ENGLISH_DIR / "inference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    shared = load_shared_inference()
    args = shared.parse_args(TELUGU_DIR / "config.yaml", __doc__)
    shared.run(args, lambda _config: TeluguFrontend(), TEST_SENTENCES)


if __name__ == "__main__":
    main()
