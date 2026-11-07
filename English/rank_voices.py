"""Choose the English inference voice by intelligibility, not by data size alone.

The model is conditioned on demographic groups. For the groups with the most
training clips, the same validation sentences are synthesized in each voice and
transcribed by Whisper; voices are ranked by word error rate. Validation text is
used, so the test sentences play no part in the choice.

Optional dependency: pip install openai-whisper

    python English/rank_voices.py --candidates 10 --sentences 30
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ENGLISH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ENGLISH_DIR))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ENGLISH_DIR / "config.yaml"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--restore_step", type=int, default=20000)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--sentences", type=int, default=30)
    parser.add_argument("--whisper_model", default="small")
    parser.add_argument("--whisper_root")
    args = parser.parse_args()

    import soxr
    import whisper
    from tqdm import tqdm
    from whisper.normalizers import EnglishTextNormalizer

    from common import load_configs
    from evaluate_asr import edit_distance
    from inference import load_model, synthesize
    from utils.model import get_vocoder, vocoder_infer
    from utils.tools import device

    preprocess_config, model_config, train_config, _ = load_configs(args.config)
    data_dir = Path(preprocess_config["path"]["preprocessed_path"])
    settings = preprocess_config["preprocessing"]
    rate, hop = settings["audio"]["sampling_rate"], settings["stft"]["hop_length"]
    speakers = json.loads((data_dir / "speakers.json").read_text(encoding="utf-8"))

    train_counts = Counter(line.split("|")[1] for line in open(data_dir / "train.txt", encoding="utf-8"))
    candidates = [voice for voice, _ in train_counts.most_common(args.candidates)]
    lines = (data_dir / "val.txt").read_text(encoding="utf-8").splitlines()
    sentences = random.Random(7).sample(lines, args.sentences)

    model, _, _ = load_model(preprocess_config, model_config, train_config, args.restore_step, args.checkpoint)
    vocoder = get_vocoder(model_config, device)
    asr = whisper.load_model(args.whisper_model, device=device, download_root=args.whisper_root)
    normalize = EnglishTextNormalizer()

    rows = []
    for voice in tqdm(candidates, desc="Voices"):
        errors = words = 0
        for line in sentences:
            _, _, phones, reference = line.split("|")
            result = synthesize(model, phones.strip("{}").split(), speakers[voice], settings["text"]["text_cleaners"])
            wav = vocoder_infer(result["postnet_mel"], vocoder, model_config, preprocess_config,
                                lengths=[result["mel_length"] * hop])[0] / 32768.0
            text = asr.transcribe(soxr.resample(wav.astype(np.float32), rate, 16000), language="en",
                                  temperature=0.0, beam_size=5, condition_on_previous_text=False,
                                  fp16=device.type == "cuda")["text"]
            ref, hyp = normalize(reference).split(), normalize(text).split()
            errors += edit_distance(ref, hyp)
            words += len(ref)
        rows.append({"voice": voice, "training_clips": train_counts[voice], "wer": round(errors / words, 4)})
        print(rows[-1])

    rows.sort(key=lambda r: r["wer"])
    out = Path(train_config["path"]["log_path"]) / "evaluation" / "voice_ranking.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Best voice:", rows[0]["voice"], "->", out)


if __name__ == "__main__":
    main()
