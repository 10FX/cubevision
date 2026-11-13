"""The detector's recall and crops by cube size, with one look at the frame and with the app's second,
closer look at small cubes (src/lib/vision/detector.ts: under 25 % of the longer side, a square of 4
times the box). From ml2/: PYTHONPATH=. uv run python tools/detector_sizes.py runs/detector/best.pt data/detect/test
"""
import math
import sys
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, decode_jpeg, read_file
from torchvision.transforms.v2 import functional as TF

from cubevision.data import LabelTable
from cubevision.detector import CROP_MARGIN, MIN_INSIDE, PAD, ExportWrapper, inside_share
from cubevision.train_detector import load_detector

SMALL, ZOOM, THRESHOLD = 0.25, 4.0, 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, cfg = load_detector(Path(sys.argv[1]), device)
size = cfg.model.size
wrapper = ExportWrapper(model, size).to(device).eval()
table = LabelTable([Path(sys.argv[2])])


def look(image: torch.Tensor, region: tuple[float, float, float, float]):
    """Letterbox a region (pixels, may pass the edges; grey there) and detect: box in frame pixels, score."""
    _, h, w = image.shape
    x, y, rw, rh = math.floor(region[0]), math.floor(region[1]), math.ceil(region[2]), math.ceil(region[3])
    scale = size / max(rw, rh)
    nw, nh = max(1, round(rw * scale)), max(1, round(rh * scale))
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas = torch.full((3, rh, rw), round(PAD * 255), dtype=torch.uint8)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(w, x + rw), min(h, y + rh)
    if x1 > x0 and y1 > y0:
        canvas[:, y0 - y:y1 - y, x0 - x:x1 - x] = image[:, y0:y1, x0:x1]
    square = torch.full((3, size, size), round(PAD * 255), dtype=torch.uint8)
    square[:, oy:oy + nh, ox:ox + nw] = TF.resize(canvas, [nh, nw], antialias=True)
    with torch.no_grad():
        box, score = wrapper(square[None].to(device).float() / 255)
    b = (box[0].float().cpu() * size).tolist()
    return [(b[0] - ox) / scale + x, (b[1] - oy) / scale + y, (b[2] - ox) / scale + x, (b[3] - oy) / scale + y], float(score[0, 0])


def crop_ok(box, truth) -> bool:
    """The reader's crop around the box holds the cube, at the size it was trained on (as DetectionMetrics)."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = max(box[2] - box[0], box[3] - box[1]) * CROP_MARGIN / 2
    crop = [cx - half, cy - half, cx + half, cy + half]
    ix = max(0.0, min(crop[2], truth[2]) - max(crop[0], truth[0]))
    iy = max(0.0, min(crop[3], truth[3]) - max(crop[1], truth[1]))
    held = ix * iy / ((truth[2] - truth[0]) * (truth[3] - truth[1]))
    fill = max(truth[2] - truth[0], truth[3] - truth[1]) / (2 * half)
    return held >= 0.98 and 0.5 <= fill <= 0.85


buckets = [(0, 0.06), (0.06, 0.1), (0.1, 0.2), (0.2, 1.01)]
stats = {b: {"n": 0, "one": 0, "two": 0, "crop1": 0, "crop2": 0} for b in buckets}
fp: dict[str, int] = {}
negatives: dict[str, int] = {}
for i in range(len(table)):
    image = decode_jpeg(read_file(table.path(i)), mode=ImageReadMode.RGB)
    _, h, w = image.shape
    box, score = look(image, (0, 0, w, h))
    found = score >= THRESHOLD
    truth = table.box(i)
    if truth is None or inside_share(truth) < MIN_INSIDE:
        # No cube at all, or a cube mostly out of the frame (not to be read: still a cube though)
        kind = "none" if truth is None else "outside"
        negatives[kind] = negatives.get(kind, 0) + 1
        fp[kind] = fp.get(kind, 0) + found
        continue
    t = [truth[0] * w, truth[1] * h, truth[2] * w, truth[3] * h]
    share = max(t[2] - t[0], t[3] - t[1]) / max(w, h)
    bucket = next(b for b in buckets if b[0] <= share < b[1])
    s = stats[bucket]
    s["n"] += 1
    s["one"] += found
    s["crop1"] += found and crop_ok(box, t)
    # The app's second look, closer, when the cube found is small
    if found and max(box[2] - box[0], box[3] - box[1]) < SMALL * max(w, h):
        side = max(box[2] - box[0], box[3] - box[1]) * ZOOM
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        zoomed, zscore = look(image, (cx - side / 2, cy - side / 2, side, side))
        if zscore >= THRESHOLD:
            box = zoomed
    s["two"] += found
    s["crop2"] += found and crop_ok(box, t)
for kind, n in negatives.items():
    print(f"{'no cube' if kind == 'none' else 'cube mostly out of the frame'}: {n} frames, found in {fp[kind]}")
print("cube size (share of the longer side) | frames | recall | crop_ok one look | crop_ok with the closer look")
for (lo, hi), s in stats.items():
    n = max(1, s["n"])
    print(f"{lo:.2f}-{min(hi, 1):.2f} | {s['n']:5d} | {s['one'] / n:.3f} | {s['crop1'] / n:.3f} | {s['crop2'] / n:.3f}")
total = {k: sum(s[k] for s in stats.values()) for k in ("n", "one", "crop1", "crop2")}
print(f"all | {total['n']} | {total['one'] / total['n']:.3f} | {total['crop1'] / total['n']:.3f} | {total['crop2'] / total['n']:.3f}")
