"""Train the English FastSpeech 2 model on preprocessed Svarah data.

Reuses the repository's FastSpeech2 model, loss, optimizer schedule and Dataset.
This entry point adds frame-limited batching for a 6 GB GPU, gradient accumulation,
out-of-memory recovery, CSV loss logs, and checkpoints that resume exactly.

Examples:
    python English/train.py --smoke_test
    python English/train.py
    python English/train.py --restore_step 10000 --max_minutes 180
"""
import argparse
import csv
import gc
import os
import random
import sys
import time
from pathlib import Path

import common  # noqa: F401  (adds the repository root to sys.path)
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter

from common import load_configs
from dataset import Dataset
from model import FastSpeech2, FastSpeech2Loss, ScheduledOptim
from utils.model import get_param_num
from utils.tools import device, log, to_device

LOSS_NAMES = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]


class FrameLimitedBatchSampler(Sampler):
    """Group similar-length utterances, limiting total frames in each batch."""

    def __init__(self, lengths, max_frames, max_batch, shuffle, seed=42):
        self.lengths = lengths
        self.max_frames = max_frames
        self.max_batch = max_batch
        self.shuffle = shuffle
        self.random = random.Random(seed)
        self.batches = self._build()

    def _build(self):
        order = list(range(len(self.lengths)))
        if self.shuffle:
            # Shuffle first so equal-length utterances land in different batches.
            self.random.shuffle(order)
        order.sort(key=lambda i: self.lengths[i])

        batches, current, longest = [], [], 0
        for index in order:
            longest_next = max(longest, self.lengths[index])
            # Padding makes the cost of a batch: size x longest utterance.
            if current and (
                len(current) >= self.max_batch
                or (len(current) + 1) * longest_next > self.max_frames
            ):
                batches.append(current)
                current, longest_next = [], self.lengths[index]
            current.append(index)
            longest = longest_next
        if current:
            batches.append(current)
        return batches

    def __iter__(self):
        batches = self._build() if self.shuffle else self.batches
        if self.shuffle:
            self.random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return len(self.batches)


def make_loader(name, preprocess_config, train_config, shuffle):
    dataset = Dataset(name, preprocess_config, train_config, sort=False, drop_last=False)
    out_dir = Path(preprocess_config["path"]["preprocessed_path"])
    lengths = [
        int(np.load(out_dir / "duration" / f"{s}-duration-{b}.npy").sum())
        for s, b in zip(dataset.speaker, dataset.basename)
    ]
    optimizer = train_config["optimizer"]
    sampler = FrameLimitedBatchSampler(
        lengths,
        optimizer["max_frames_per_batch"],
        optimizer["batch_size"],
        shuffle,
        train_config["seed"],
    )

    def collate(items):
        return dataset.reprocess(items, list(range(len(items))))

    workers = train_config["num_workers"]
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=workers,
        persistent_workers=workers > 0,
    )
    return dataset, loader


def is_out_of_memory(error):
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error)


def clear_memory(model):
    model.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError as error:  # the allocator itself can fail under pressure
            print(f"empty_cache failed: {error}".splitlines()[0])


@torch.no_grad()
def validate(model, loader, loss_fn):
    """Mean validation losses per utterance, using ground-truth durations."""
    model.eval()
    totals, count, skipped = np.zeros(len(LOSS_NAMES)), 0, 0
    for raw_batch in loader:
        batch = to_device(raw_batch, device)
        try:
            losses = loss_fn(batch, model(*batch[2:]))
        except RuntimeError as error:
            if not is_out_of_memory(error):
                raise
            clear_memory(model)
            skipped += len(raw_batch[0])
            continue
        size = len(raw_batch[0])
        totals += np.array([l.item() for l in losses]) * size
        count += size
    model.train()
    if count == 0:
        raise RuntimeError("Validation ran out of memory on every batch.")
    return totals / count, count, skipped


def append_csv(path, row, header):
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        writer.writerow(row)


def trim_log_after(path, step):
    """On resume, drop rows from after the checkpoint so steps never repeat."""
    if not path.exists():
        return
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    kept = [rows[0]] + [r for r in rows[1:] if int(r[0]) <= step]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(kept)


def save_checkpoint(path, model, optimizer, step, config_note=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer._optimizer.state_dict(),
            "step": step,
            "optimizer_updates": optimizer.current_step,
            "note": config_note,
        },
        temporary,
    )
    os.replace(temporary, path)  # never leave a half-written checkpoint behind
    return path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args):
    preprocess_config, model_config, train_config, _ = load_configs(args.config)
    seed_everything(train_config["seed"])
    if args.total_step:
        train_config["step"]["total_step"] = args.total_step

    if args.require_gpu and not torch.cuda.is_available():
        sys.exit("CUDA is not available and --require_gpu was given.")
    print(f"Device: {device}" + (
        f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""
    ))

    train_set, train_loader = make_loader("train.txt", preprocess_config, train_config, True)
    val_set, val_loader = make_loader("val.txt", preprocess_config, train_config, False)
    print(f"Train: {len(train_set)} utterances, {len(train_loader)} batches per epoch")
    print(f"Val: {len(val_set)} utterances, {len(val_loader)} batches")

    model = FastSpeech2(preprocess_config, model_config).to(device)
    loss_fn = FastSpeech2Loss(preprocess_config, model_config).to(device)
    optimizer = ScheduledOptim(model, train_config, model_config, 0)
    print("Parameters:", get_param_num(model))

    ckpt_dir = Path(train_config["path"]["ckpt_path"])
    log_dir = Path(train_config["path"]["log_path"])
    if args.smoke_test:
        ckpt_dir, log_dir = ckpt_dir / "smoke_test", log_dir / "smoke_test"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_csv, val_csv = log_dir / "train_loss.csv", log_dir / "val_loss.csv"
    events_csv = log_dir / "events.csv"
    header = ["step", "updates", "elapsed_s", *LOSS_NAMES]

    step = 0
    if args.restore_step:
        checkpoint = torch.load(
            ckpt_dir / f"{args.restore_step}.pth.tar", map_location=device
        )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Upstream restarts the schedule from the step count, which is wrong
        # whenever gradient accumulation makes updates differ from steps.
        optimizer.current_step = checkpoint["optimizer_updates"]
        step = checkpoint["step"]
        for path in (train_csv, val_csv):
            trim_log_after(path, step)
        print(f"Resumed from step {step} ({optimizer.current_step} optimizer updates)")

    train_writer = SummaryWriter(str(log_dir / "train"))
    val_writer = SummaryWriter(str(log_dir / "val"))

    settings = train_config["step"]
    grad_steps = train_config["optimizer"]["grad_acc_step"]
    grad_clip = train_config["optimizer"]["grad_clip_thresh"]
    total_step = settings["total_step"]
    started = time.time()
    last_report, last_report_step = started, step
    deadline = started + args.max_minutes * 60 if args.max_minutes else None
    oom_skips, oom_limit = 0, 50
    stop_reason = None

    model.train()
    optimizer.zero_grad()
    while stop_reason is None:
        for raw_batch in train_loader:
            batch = to_device(raw_batch, device)
            try:
                output = model(*batch[2:])
                losses = loss_fn(batch, output)
                if not torch.isfinite(losses[0]):
                    raise FloatingPointError(
                        f"Non-finite loss at step {step + 1}: {[l.item() for l in losses]}"
                    )
                (losses[0] / grad_steps).backward()
            except RuntimeError as error:
                if not is_out_of_memory(error):
                    raise
                oom_skips += 1
                frames = int(raw_batch[7].max()) * len(raw_batch[0])
                print(f"OOM on batch of {len(raw_batch[0])} ({frames} frames); skipping")
                append_csv(events_csv, [step, "oom_skip", frames], ["step", "event", "detail"])
                del batch
                clear_memory(model)
                if oom_skips >= oom_limit:
                    sys.exit("Too many out-of-memory batches: lower max_frames_per_batch.")
                continue

            step += 1
            if step % grad_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step_and_update_lr()
                optimizer.zero_grad()

            elapsed = time.time() - started
            if step % settings["log_step"] == 0 or args.smoke_test:
                values = [l.item() for l in losses]
                append_csv(
                    train_csv,
                    [step, optimizer.current_step, round(elapsed, 1), *values],
                    header,
                )
                log(train_writer, step, losses=values)
                now = time.time()
                speed = (step - last_report_step) / max(now - last_report, 1e-6)
                last_report, last_report_step = now, step
                memory = (
                    torch.cuda.max_memory_allocated() / 2**30
                    if torch.cuda.is_available()
                    else 0.0
                )
                print(
                    f"step {step}/{total_step} {speed:.2f} it/s peak {memory:.2f} GB | "
                    + " ".join(f"{n} {v:.4f}" for n, v in zip(LOSS_NAMES, values)),
                    flush=True,
                )

            if step % settings["val_step"] == 0:
                values, count, skipped = validate(model, val_loader, loss_fn)
                append_csv(
                    val_csv,
                    [step, optimizer.current_step, round(elapsed, 1), *values.tolist()],
                    header,
                )
                log(val_writer, step, losses=values.tolist())
                print(
                    f"VALIDATION step {step} ({count} utts, {skipped} skipped) | "
                    + " ".join(f"{n} {v:.4f}" for n, v in zip(LOSS_NAMES, values)),
                    flush=True,
                )

            if step % settings["save_step"] == 0:
                path = save_checkpoint(ckpt_dir / f"{step}.pth.tar", model, optimizer, step)
                print("Saved", path, flush=True)

            if step >= total_step:
                stop_reason = "total_step reached"
            elif deadline and time.time() > deadline:
                stop_reason = "time limit reached"
            if stop_reason:
                break

    if step % settings["save_step"] != 0:
        path = save_checkpoint(ckpt_dir / f"{step}.pth.tar", model, optimizer, step)
        print("Saved", path)
    train_writer.close()
    val_writer.close()
    print(
        f"Stopped: {stop_reason} at step {step} after "
        f"{(time.time() - started) / 3600:.2f} h; {oom_skips} OOM batches skipped."
    )
    return step


def smoke_test(args):
    """Prove the data, forward pass, backprop and checkpoint round trip all work."""
    args.total_step = args.smoke_steps
    preprocess_config, model_config, train_config, _ = load_configs(args.config)
    ckpt_dir = Path(train_config["path"]["ckpt_path"]) / "smoke_test"

    step = train(args)
    checkpoint_path = ckpt_dir / f"{step}.pth.tar"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    reloaded = FastSpeech2(preprocess_config, model_config).to(device)
    reloaded.load_state_dict(checkpoint["model"])
    original = checkpoint["model"]
    for name, value in reloaded.state_dict().items():
        assert torch.equal(value, original[name]), f"Mismatch in {name}"

    print("Checking resume from the saved checkpoint...")
    args.restore_step = step
    args.total_step = step + 2
    resumed = train(args)
    assert resumed == step + 2
    print(f"SMOKE TEST PASSED: {step} steps, checkpoint reloaded, resumed to {resumed}.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(common.DEFAULT_CONFIG))
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--total_step", type=int, default=0, help="overrides the config")
    parser.add_argument("--max_minutes", type=float, default=0, help="stop and save after this")
    parser.add_argument("--require_gpu", action="store_true")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--smoke_steps", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.smoke_test:
        smoke_test(arguments)
    else:
        train(arguments)
