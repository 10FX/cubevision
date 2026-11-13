"""
Train the cube reader.

    python -m cubevision.train --config configs/full.yaml
    python -m cubevision.train --config configs/full.yaml --set train.epochs=5 data.limit=20000   # quick run
    python -m cubevision.train --config configs/full.yaml --resume runs/full/last.pt

A run folder (``runs/<name>``) gets the config, TensorBoard logs, metrics.csv,
``last.pt`` (to resume) and ``best.pt`` (best validation face accuracy, EMA
weights). Mixed precision (bf16 on RTX 30/40/50), AdamW with warmup and
cosine decay, gradient clipping, exponential moving average of the weights.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import random
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from .augment import AugmentConfig, augment, move_placement, roll_orientation
from .data import SyntheticDataset, dataset_seed
from .losses import LossWeights, Metrics, reader_loss
from .model import CubeReader, ModelConfig, by_colour, count_parameters


@dataclass
class DataConfig:
    train: list[str] = field(default_factory=lambda: ["data/synth/train"])
    val: list[str] = field(default_factory=lambda: ["data/synth/val"])
    limit: int = 0  # use only this many training images (0: all), for quick runs
    workers: int = 8


@dataclass
class TrainConfig:
    name: str = "full"
    out: str = "runs"
    seed: int = 0
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1.5e-3
    backbone_lr_scale: float = 0.3  # the pretrained backbone moves slower than the new head
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    grad_clip: float = 1.0
    ema: float = 0.999
    amp: str = "bf16"  # bf16 | fp16 | none
    compile: bool = False
    channels_last: bool = True
    log_every: int = 50
    #: Geometric reader: share of the images read at the true geometry, from the start to the end
    teach_start: float = 1.0
    teach_end: float = 0.3


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    loss: LossWeights = field(default_factory=LossWeights)


def _merge(obj, values: dict):
    for key, value in values.items():
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            _merge(current, value)
        else:
            setattr(obj, key, tuple(value) if isinstance(current, tuple) else value)
    return obj


def load_config(path: Path | None, overrides: list[str]) -> Config:
    cfg = Config()
    if path:
        _merge(cfg, yaml.safe_load(path.read_text()) or {})
    for item in overrides:
        key, raw = item.split("=", 1)
        *parents, leaf = key.split(".")
        target = cfg
        for p in parents:
            target = getattr(target, p)
        setattr(target, leaf, yaml.safe_load(raw))
    return cfg


def config_dict(cfg: Config) -> dict:
    """Plain types only (tuples become lists), for YAML and checkpoints."""
    return json.loads(json.dumps(dataclasses.asdict(cfg)))


def config_from_dict(raw: dict) -> Config:
    return _merge(Config(), raw)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _param_groups(model: CubeReader, cfg: TrainConfig) -> list[dict]:
    """Backbone at a lower rate; no weight decay on norms, biases, queries and positions."""
    groups: dict[tuple[bool, bool], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        backbone = name.startswith(("stem8.", "stem.", "deep."))
        decay = p.ndim > 1 and name not in ("position", "queries")
        groups.setdefault((backbone, decay), []).append(p)
    return [
        {"params": params, "lr_scale": cfg.backbone_lr_scale if backbone else 1.0, "weight_decay": cfg.weight_decay if decay else 0.0}
        for (backbone, decay), params in groups.items()
    ]


def _lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


class EMA:
    """Exponential moving average of the weights (evaluated and exported instead of the raw weights)."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.model = deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        # Warm-up of the decay so that early weights are not frozen in
        d = min(self.decay, (1 + step) / (10 + step))
        for e, m in zip(self.model.state_dict().values(), model.state_dict().values(), strict=True):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


def _autocast(cfg: TrainConfig, device: torch.device):
    if cfg.amp == "none" or device.type != "cuda":
        return torch.autocast(device.type, enabled=False)
    return torch.autocast("cuda", dtype=torch.bfloat16 if cfg.amp == "bf16" else torch.float16)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, cfg: Config) -> dict[str, float]:
    model.eval()
    metrics = Metrics()
    losses = []
    for image, present, stickers, turning, orientation, has_orientation, placement in loader:
        image = image.to(device, non_blocking=True).float().div_(255)
        if cfg.train.channels_last:
            image = image.contiguous(memory_format=torch.channels_last)
        present, stickers, turning = present.to(device), stickers.to(device), turning.to(device)
        orientation, has_orientation = orientation.to(device), has_orientation.to(device)
        with _autocast(cfg.train, device):
            out = model(image, orientation) if cfg.model.orientation_oracle else model(image)
        losses.append(reader_loss(out, present, stickers, turning, cfg.loss, orientation, has_orientation, placement.to(device))["loss"].item())
        metrics.update(by_colour(out) if out["stickers"].shape[2] == 9 else out, present, stickers, turning, orientation, has_orientation)
    return {"loss": float(np.mean(losses)), **metrics.summary()}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path)
    p.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=5 model.backbone=mobilenet_v3_small")
    p.add_argument("--resume", type=Path)
    p.add_argument("--init", type=Path, help="start from this reader's weights (best.pt), to go on with harder images")
    args = p.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg.train.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    run = Path(cfg.train.out) / cfg.train.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(yaml.safe_dump({**config_dict(cfg), "git": _git_commit()}, sort_keys=False))

    # Validation images rendered from a training seed would be training images
    shared = {dataset_seed(Path(r)) for r in cfg.data.train} & {dataset_seed(Path(r)) for r in cfg.data.val} - {None}
    if shared:
        raise SystemExit(f"train and val folders share the seed(s) {sorted(shared)}: render val with another --seed")

    size = cfg.model.image_size
    train_set = SyntheticDataset([Path(r) for r in cfg.data.train], size, train=True)
    if cfg.data.limit:
        train_set = Subset(train_set, range(min(cfg.data.limit, len(train_set))))
    val_set = SyntheticDataset([Path(r) for r in cfg.data.val], size, train=False)
    loader_args = dict(num_workers=cfg.data.workers, pin_memory=device.type == "cuda", persistent_workers=cfg.data.workers > 0)
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size * 2, shuffle=False, **loader_args)

    model = CubeReader(cfg.model).to(device)
    if cfg.train.channels_last:
        model = model.to(memory_format=torch.channels_last)
    print(f"[train] {cfg.model.backbone}: {count_parameters(model) / 1e6:.2f} M parameters, {len(train_set)} train / {len(val_set)} val images, {device}")
    optimizer = torch.optim.AdamW(_param_groups(model, cfg.train), lr=cfg.train.lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp == "fp16" and device.type == "cuda")
    ema = EMA(model, cfg.train.ema)
    steps_per_epoch = len(train_loader)
    total = steps_per_epoch * cfg.train.epochs
    warmup = int(steps_per_epoch * cfg.train.warmup_epochs)
    start_epoch, step, best = 0, 0, -1.0
    if args.init:
        weights = torch.load(args.init, map_location=device, weights_only=False)["ema"]
        model.load_state_dict(weights)
        ema.model.load_state_dict(weights)
        print(f"[train] weights from {args.init}")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, step, best = ckpt["epoch"] + 1, ckpt["step"], ckpt["best"]
        print(f"[train] resumed at epoch {start_epoch}")
    forward = torch.compile(model) if cfg.train.compile else model

    writer = SummaryWriter(run / "tb")
    csv_path = run / "metrics.csv"
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        t0 = time.time()
        for image, present, stickers, turning, orientation, has_orientation, placement in train_loader:
            lr = _lr_at(step, total, warmup, cfg.train.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr * group["lr_scale"]
            image, roll, theta = augment(image.to(device, non_blocking=True).float().div_(255), cfg.augment)
            if cfg.train.channels_last:
                image = image.contiguous(memory_format=torch.channels_last)
            present, stickers, turning = present.to(device), stickers.to(device), turning.to(device)
            # The image turned: so did the cube in the camera's frame
            orientation, has_orientation = roll_orientation(orientation.to(device), roll), has_orientation.to(device)
            # The cube moved with the image too
            placement = move_placement(placement.to(device), theta)
            teach = None
            if cfg.model.geometric:
                # The true geometry, for fewer and fewer images: at the end the reader reads where it thinks
                share = cfg.train.teach_start + (cfg.train.teach_end - cfg.train.teach_start) * min(1.0, step / max(1, total))
                use = (torch.rand(len(image), device=device) < share) & (has_orientation > 0.5) & ~torch.isnan(placement).any(-1)
                teach = (orientation, torch.nan_to_num(placement), use)
            with _autocast(cfg.train, device):
                out = forward(image, orientation) if cfg.model.orientation_oracle else forward(image, teach=teach)
            losses = reader_loss(out, present, stickers, turning, cfg.loss, orientation, has_orientation, placement)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model, step)
            if step % cfg.train.log_every == 0:
                for k, v in losses.items():
                    writer.add_scalar(f"train/{k}", v.item(), step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/grad_norm", float(grad_norm), step)
                parts = " ".join(f"{k} {v.item():.3f}" for k, v in losses.items() if k != "loss")
                print(f"[train] epoch {epoch} step {step}/{total} loss {losses['loss'].item():.4f} ({parts}) lr {lr:.2e}", flush=True)
            step += 1

        val = evaluate(ema.model, val_loader, device, cfg)
        for k, v in val.items():
            writer.add_scalar(f"val/{k}", v, step)
        row = {"epoch": epoch, "step": step, "seconds": round(time.time() - t0), **{k: round(v, 5) for k, v in val.items()}}
        new = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"[val] epoch {epoch}: " + json.dumps(row), flush=True)
        state = {"model": model.state_dict(), "ema": ema.model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "step": step, "best": best, "config": config_dict(cfg)}
        torch.save(state, run / "last.pt")
        score = val["face_exact"] if not math.isnan(val["face_exact"]) else 0.0
        if score > best:
            best = state["best"] = score
            torch.save({"ema": ema.model.state_dict(), "config": config_dict(cfg), "val": val, "epoch": epoch}, run / "best.pt")
            print(f"[val] new best face_exact {score:.4f}", flush=True)
    writer.close()


if __name__ == "__main__":
    main()
