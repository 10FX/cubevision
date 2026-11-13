"""What the geometric reader sees in crops: its projected stickers drawn over each crop, filled with the
colour it reads there (faces it finds readable only), and its turning score.

    PYTHONPATH=. uv run python tools/reader_view.py <checkpoint> <crop sheet.png (8×6 tiles, from vision:eval CROPS)> <out.png>
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from cubevision.evaluate import load_model
from cubevision.geometry import project
from cubevision.model import rotation_from_6d

COL = [(245, 245, 245), (220, 30, 30), (20, 170, 60), (250, 220, 20), (250, 130, 20), (30, 80, 220), (0, 0, 0)]

model, cfg = load_model(Path(sys.argv[1]), torch.device("cpu"))
sheet = Image.open(sys.argv[2]).convert("RGB")
size = cfg.model.image_size
tiles = [sheet.crop((c * size, r * size, (c + 1) * size, (r + 1) * size)) for r in range(sheet.height // size) for c in range(sheet.width // size)]
images = torch.from_numpy(np.stack([np.asarray(t) for t in tiles])).permute(0, 3, 1, 2).float() / 255
with torch.no_grad():
    out = model(images)
pose = rotation_from_6d(out["orientation"].float())
where = project(pose, out["placement"].float())  # [N, 54, 2]
colours = out["stickers"].argmax(-1)  # [N, 6, 9] (faces of the pose)
present = torch.sigmoid(out["present"])  # [N, 6]
turning = torch.sigmoid(out["turning"])
scale = 2
result = Image.new("RGB", (sheet.width * scale, sheet.height * scale))
for k, tile in enumerate(tiles):
    big = tile.resize((size * scale, size * scale))
    d = ImageDraw.Draw(big)
    for f in range(6):
        if present[k, f] < 0.5:
            continue
        for c in range(9):
            x, y = (where[k, f * 9 + c] * size * scale).tolist()
            r = 7 if c == 4 else 5
            d.ellipse([x - r, y - r, x + r, y + r], fill=COL[int(colours[k, f, c])], outline=(255, 0, 255) if c == 4 else (0, 0, 0), width=2)
    d.text((6, 6), f"turn {turning[k].item():.2f}  faces {int((present[k] > 0.5).sum())}", fill=(255, 255, 0))
    result.paste(big, ((k % (sheet.width // size)) * size * scale, (k // (sheet.width // size)) * size * scale))
result.save(sys.argv[3])
print(f"{len(tiles)} crops; readable faces per crop: {(present > 0.5).sum(1).float().mean():.2f}; turning > 0.5: {(turning > 0.5).float().mean():.0%}")
