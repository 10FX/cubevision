"""How often a detector sees a cube in frames that have none (real photos without a cube), at its threshold.

    PYTHONPATH=. uv run python tools/false_alarms.py runs/detector-v3/best.pt data/detect/real-neg-test
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cubevision.detector import DetectionDataset, collate
from cubevision.train_detector import calibrated_threshold, load_detector, predict

checkpoint, data = Path(sys.argv[1]), Path(sys.argv[2])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, cfg = load_detector(checkpoint, device)
loader = DataLoader(DetectionDataset([data], cfg.model.size, train=False), batch_size=64, num_workers=2, collate_fn=collate)
_, scores, _ = predict(model, loader, device, cfg.model.size)
threshold = calibrated_threshold(checkpoint)
print(f"{len(scores)} frames, threshold {threshold}: false alarms {(scores >= threshold).float().mean():.1%} "
      f"(at 0.3: {(scores >= 0.3).float().mean():.1%}, median score {scores.median():.2f})")
