"""
Fine-tune the cube detector (cubevision/detector.py) on whole frames rendered
with ``--framing full``.

    python -m cubevision.train_detector --config configs/detector.yaml
    python -m cubevision.train_detector --config configs/detector.yaml --set train.epochs=2 data.limit=2000   # quick run
    python -m cubevision.train_detector --evaluate runs/detector/best.pt --data data/detect/test

Same run folder as the reader's training (runs/<name>: config, TensorBoard,
metrics.csv, last.pt, best.pt with EMA weights). The best checkpoint is the
one whose crops hold the cube best (``crop_ok``, see DetectionMetrics).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from .augment import AugmentConfig, photometric
from .data import dataset_seed
from .detector import DetectionDataset, DetectionMetrics, DetectorConfig, ExportWrapper, build_detector, collate
from .train import EMA, _git_commit, _lr_at, _merge, config_dict, seed_everything


@dataclass
class DetectorData:
    train: list[str] = field(default_factory=lambda: ["data/detect/train"])
    val: list[str] = field(default_factory=lambda: ["data/detect/val"])
    limit: int = 0
    workers: int = 8
    #: Share of training frames zoomed out a lot: the cube small and far, as in a 480p webcam's view
    far: float = 0.0


@dataclass
class DetectorTrain:
    name: str = "detector"
    out: str = "runs"
    seed: int = 0
    epochs: int = 24
    batch_size: int = 64
    lr: float = 1e-3
    backbone_lr_scale: float = 0.3
    weight_decay: float = 1e-4
    warmup_epochs: float = 1.0
    grad_clip: float = 5.0
    ema: float = 0.999
    amp: str = "bf16"
    log_every: int = 50


@dataclass
class DetectorRun:
    model: DetectorConfig = field(default_factory=DetectorConfig)
    data: DetectorData = field(default_factory=DetectorData)
    train: DetectorTrain = field(default_factory=DetectorTrain)
    # Lighter than the reader's: the box only needs the cube to stand out
    augment: AugmentConfig = field(default_factory=lambda: AugmentConfig(blur_p=0.3, motion_p=0.4, lowres_p=0.3))


def load_config(path: Path | None, overrides: list[str]) -> DetectorRun:
    cfg = DetectorRun()
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


def _autocast(amp: str, device: torch.device):
    if amp == "none" or device.type != "cuda":
        return torch.autocast(device.type, enabled=False)
    return torch.autocast("cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device, size: int):
    """The box and score the browser would get (ExportWrapper: best box, no NMS) for every frame, and the targets."""
    wrapper = ExportWrapper(model, size).to(device).eval()
    boxes, scores, targets = [], [], []
    for images, batch_targets in loader:
        box, score = wrapper(images.to(device).float().div_(255))
        boxes.append(box.float().cpu())
        scores.append(score.float().cpu())
        targets += batch_targets
    return torch.cat(boxes), torch.cat(scores), targets


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device, size: int, threshold: float = 0.5) -> dict[str, float]:
    return metrics_at(*predict(model, loader, device, size), size, threshold)


def metrics_at(boxes: torch.Tensor, scores: torch.Tensor, targets: list[torch.Tensor], size: int, threshold: float) -> dict[str, float]:
    metrics = DetectionMetrics(threshold)
    metrics.update(boxes, scores, targets, size)
    return metrics.summary()


def choose_threshold(boxes: torch.Tensor, scores: torch.Tensor, targets: list[torch.Tensor], size: int, min_precision: float = 0.98) -> float:
    """The lowest score threshold (so the most cubes found) whose precision reaches ``min_precision``;
    if none does, the best balance of cubes found and false alarms (F1). Chosen on validation frames,
    as the reader's temperature. (The most precise one instead found half the cubes among look-alikes.)"""
    candidates = [round(t, 2) for t in torch.arange(0.3, 0.991, 0.01).tolist()]
    at = {t: metrics_at(boxes, scores, targets, size, t) for t in candidates}
    precise = [t for t in candidates if at[t]["precision"] >= min_precision]
    if precise:
        return precise[0]
    f1 = lambda m: 2 * m["precision"] * m["recall"] / max(1e-9, m["precision"] + m["recall"])  # noqa: E731
    return max(candidates, key=lambda t: f1(at[t]))


def calibrated_threshold(checkpoint: Path) -> float:
    """The threshold chosen for a checkpoint (threshold.json next to it), else 0.5."""
    path = checkpoint.parent / "threshold.json"
    return json.loads(path.read_text())["threshold"] if path.exists() else 0.5


def _param_groups(model: torch.nn.Module, cfg: DetectorTrain) -> list[dict]:
    groups: dict[tuple[bool, bool], list] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        key = (name.startswith("backbone."), p.ndim > 1)
        groups.setdefault(key, []).append(p)
    return [
        {"params": params, "lr_scale": cfg.backbone_lr_scale if backbone else 1.0, "weight_decay": cfg.weight_decay if decay else 0.0}
        for (backbone, decay), params in groups.items()
    ]


def load_detector(checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, DetectorRun]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = _merge(DetectorRun(), ckpt["config"])
    cfg.model.pretrained = False
    model = build_detector(cfg.model).to(device)
    model.load_state_dict(ckpt["ema"])
    return model.eval(), cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--resume", type=Path)
    p.add_argument("--init", type=Path, help="start from this detector's weights (fine-tuning on new frames), not COCO's")
    p.add_argument("--evaluate", type=Path, help="a checkpoint to evaluate on --data, instead of training")
    p.add_argument("--data", type=Path, nargs="+")
    p.add_argument("--out", type=Path, help="with --evaluate: write the report as JSON")
    p.add_argument("--calibrate", action="store_true",
                   help="with --evaluate on validation frames: choose the score threshold (threshold.json next to the checkpoint)")
    p.add_argument("--min-precision", type=float, default=0.98)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.evaluate:
        model, cfg = load_detector(args.evaluate, device)
        size = cfg.model.size
        loader = DataLoader(DetectionDataset(args.data, size, train=False), batch_size=128, num_workers=4, collate_fn=collate)
        predicted = predict(model, loader, device, size)
        if args.calibrate:
            threshold = choose_threshold(*predicted, size, args.min_precision)
            report = {"threshold": threshold, **metrics_at(*predicted, size, threshold)}
            (args.evaluate.parent / "threshold.json").write_text(json.dumps(report, indent=2))
            print(f"[detector] threshold {threshold} → {args.evaluate.parent / 'threshold.json'}")
        else:
            threshold = calibrated_threshold(args.evaluate)
            report = {"threshold": threshold, **metrics_at(*predicted, size, threshold)}
        print(json.dumps(report, indent=2))
        if args.out:
            args.out.write_text(json.dumps(report, indent=2))
        return

    cfg = load_config(args.config, args.set)
    seed_everything(cfg.train.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    shared = {dataset_seed(Path(r)) for r in cfg.data.train} & {dataset_seed(Path(r)) for r in cfg.data.val} - {None}
    if shared:
        raise SystemExit(f"train and val folders share the seed(s) {sorted(shared)}: render val with another --seed")
    run = Path(cfg.train.out) / cfg.train.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.yaml").write_text(yaml.safe_dump({**config_dict(cfg), "git": _git_commit()}, sort_keys=False))

    size = cfg.model.size
    train_set = DetectionDataset([Path(r) for r in cfg.data.train], size, train=True, far=cfg.data.far)
    if cfg.data.limit:
        train_set = Subset(train_set, range(min(cfg.data.limit, len(train_set))))
    val_set = DetectionDataset([Path(r) for r in cfg.data.val], size, train=False)
    loader_args = dict(num_workers=cfg.data.workers, pin_memory=device.type == "cuda", persistent_workers=cfg.data.workers > 0, collate_fn=collate)
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, batch_size=cfg.train.batch_size * 2, shuffle=False, **loader_args)

    model = build_detector(cfg.model).to(device)
    print(f"[detector] SSDLite320-MobileNetV3 ({'COCO' if cfg.model.pretrained else 'random'} weights), "
          f"{len(train_set)} train / {len(val_set)} val frames, {device}")
    optimizer = torch.optim.AdamW(_param_groups(model, cfg.train), lr=cfg.train.lr)
    ema = EMA(model, cfg.train.ema)
    steps_per_epoch = len(train_loader)
    total, warmup = steps_per_epoch * cfg.train.epochs, int(steps_per_epoch * cfg.train.warmup_epochs)
    start_epoch, step, best = 0, 0, -1.0
    if args.init:
        weights = torch.load(args.init, map_location=device, weights_only=False)["ema"]
        model.load_state_dict(weights)
        ema.model.load_state_dict(weights)
        print(f"[detector] weights from {args.init}")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        ema.model.load_state_dict(ckpt["ema"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch, step, best = ckpt["epoch"] + 1, ckpt["step"], ckpt["best"]

    writer = SummaryWriter(run / "tb")
    csv_path = run / "metrics.csv"
    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        t0 = time.time()
        for images, targets in train_loader:
            lr = _lr_at(step, total, warmup, cfg.train.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr * group["lr_scale"]
            images = photometric(images.to(device, non_blocking=True).float().div_(255), cfg.augment)
            batch_targets = [{"boxes": t.to(device), "labels": torch.ones(len(t), dtype=torch.int64, device=device)} for t in targets]
            with _autocast(cfg.train.amp, device):
                losses = model(list(images), batch_targets)
            loss = sum(v.float() for v in losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()
            ema.update(model, step)
            if step % cfg.train.log_every == 0:
                for k, v in losses.items():
                    writer.add_scalar(f"train/{k}", v.item(), step)
                writer.add_scalar("train/lr", lr, step)
                parts = " ".join(f"{k} {v.item():.3f}" for k, v in losses.items())
                print(f"[detector] epoch {epoch} step {step}/{total} loss {loss.item():.4f} ({parts}) grad {float(grad_norm):.2f} lr {lr:.2e}", flush=True)
            step += 1

        val = evaluate(ema.model, val_loader, device, size)
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
        score = val["crop_ok"] if not math.isnan(val["crop_ok"]) else 0.0
        if score > best:
            best = state["best"] = score
            torch.save({"ema": ema.model.state_dict(), "config": config_dict(cfg), "val": val, "epoch": epoch}, run / "best.pt")
            print(f"[val] new best crop_ok {score:.4f}", flush=True)
    writer.close()


if __name__ == "__main__":
    main()
