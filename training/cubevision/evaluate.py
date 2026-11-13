"""
Evaluate a checkpoint on a dataset, overall and per kind of image (cube
style, hands, background, mid-turn, state), and fit the softmax temperature
that makes the sticker probabilities honest (used by export).

    python -m cubevision.evaluate --checkpoint runs/full/best.pt --data data/synth/test
    python -m cubevision.evaluate --checkpoint runs/full/best.pt --data data/synth/val --calibrate
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import SyntheticDataset
from .losses import Metrics, rotated_targets, sticker_ce
from .model import OUTER_CELLS, CubeReader, by_colour
from .train import config_from_dict


def load_model(checkpoint: Path, device: torch.device) -> tuple[CubeReader, object]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = config_from_dict(ckpt["config"])
    cfg.model.pretrained = False  # the weights come from the checkpoint
    model = CubeReader(cfg.model).to(device)
    model.load_state_dict(ckpt["ema"])
    return model.eval(), cfg


def _groups(meta: dict, turning: float) -> list[str]:
    kind = meta.get("recipe", "?").split(":")[0]
    return [
        "all",
        f"style={meta.get('style')}",
        f"hands={meta.get('hands')}",
        f"background={meta.get('background')}",
        f"centres={meta.get('centres')}",
        f"lighting={meta.get('lighting', 'white')}",
        f"state={kind}",
        f"turning={int(turning > 0.5)}",
    ]


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Temperature minimizing the NLL of the sticker probabilities (golden-section search on log T)."""
    def nll(log_t: float) -> float:
        return F.cross_entropy(logits / math.exp(log_t), targets).item()
    lo, hi = math.log(0.3), math.log(5.0)
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(40):
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    return math.exp((lo + hi) / 2)


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, nargs="+", required=True)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--calibrate", action="store_true", help="fit the sticker temperature and save calibration.json next to the checkpoint")
    p.add_argument("--out", type=Path, help="write the report as JSON")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(args.checkpoint, device)
    dataset = SyntheticDataset(args.data, cfg.model.image_size, train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    metrics: dict[str, Metrics] = defaultdict(Metrics)
    calib_logits, calib_targets = [], []
    seen = 0
    for image, present, stickers, turning, orientation, has_orientation, _placement in loader:
        image = image.to(device).float().div_(255)
        present, stickers, turning = present.to(device), stickers.to(device), turning.to(device)
        orientation, has_orientation = orientation.to(device), has_orientation.to(device)
        out = {k: v.float() for k, v in model(image).items()}
        if out["stickers"].shape[2] == 9:
            out = by_colour(out)  # what the browser gets: faces by centre colour
        groups: dict[str, list[int]] = defaultdict(list)
        for j in range(image.shape[0]):
            for g in _groups(dataset.meta(seen + j), float(turning[j])):
                groups[g].append(j)
        for g, idx in groups.items():
            ix = torch.tensor(idx, device=device)
            metrics[g].update({k: v[ix] for k, v in out.items()}, present[ix], stickers[ix], turning[ix], orientation[ix], has_orientation[ix])
        if args.calibrate:
            _, rotation = sticker_ce(out["stickers"], stickers)
            targets = torch.gather(rotated_targets(stickers), 2, rotation[..., None, None].expand(-1, -1, 1, len(OUTER_CELLS))).squeeze(2)
            mask = present > 0.5
            calib_logits.append(out["stickers"][mask].reshape(-1, out["stickers"].shape[-1]).cpu())
            calib_targets.append(targets[mask].reshape(-1).cpu())
        seen += image.shape[0]

    report = {g: m.summary() | {"images": int(m.counts["images"])} for g, m in sorted(metrics.items())}
    width = max(len(g) for g in report)
    print(f"{'group':<{width}}  images  present  stickers  visible  hidden  face_exact  two_right  turning  orient_med°  orient≤15°")
    for g, r in report.items():
        cols = ("present_acc", "sticker_acc", "visible_sticker_acc", "hidden_acc", "face_exact", "two_faces_right", "turning_acc",
                "orientation_median_deg", "orientation_within_15deg")
        print(f"{g:<{width}}  {r['images']:6d}  " + "  ".join(f"{r[c]:.4f}" for c in cols))
    if args.calibrate:
        t = fit_temperature(torch.cat(calib_logits), torch.cat(calib_targets))
        path = args.checkpoint.parent / "calibration.json"
        path.write_text(json.dumps({"temperature": t}))
        print(f"sticker temperature {t:.3f} → {path}")
        report["calibration"] = {"temperature": t}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
