"""Real photos without a cube (people, rooms, streets) as detector frames whose answer is "no cube".

    PYTHONPATH=. uv run python tools/real_negatives.py data/backgrounds/val2017 data/detect/real-neg-train --start 0 --count 4000
    PYTHONPATH=. uv run python tools/real_negatives.py data/backgrounds/val2017 data/detect/real-neg-test --start 4000 --count 1000

The images are linked, not copied; the folder is a dataset as the renderer writes them (labels-*.jsonl,
dataset.json with no seed: no rendered folder can share its scenarios).
"""
import argparse
import json
import time
from pathlib import Path

from cubevision.labels import HIDDEN, FaceLabel, ImageLabel

p = argparse.ArgumentParser()
p.add_argument("photos", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--start", type=int, default=0)
p.add_argument("--count", type=int, default=1000)
args = p.parse_args()

photos = sorted(x for x in args.photos.iterdir() if x.suffix.lower() in (".jpg", ".jpeg", ".png"))[args.start:args.start + args.count]
(args.out / "images").mkdir(parents=True, exist_ok=True)
nothing = [FaceLabel(False, [HIDDEN] * 9) for _ in range(6)]
with (args.out / "labels-0000000.jsonl").open("w") as f:
    for k, photo in enumerate(photos):
        link = args.out / "images" / photo.name
        if not link.exists():
            link.symlink_to(photo.resolve())
        meta = {"index": args.start + k, "source": str(args.photos), "cube": False, "framing": "full", "real": True}
        f.write(ImageLabel(f"images/{photo.name}", nothing, False, meta, None, None).to_json() + "\n")
(args.out / "dataset.json").write_text(json.dumps({"seed": None, "framing": "full", "real": True, "source": str(args.photos),
                                                    "start": args.start, "count": len(photos),
                                                    "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
print(f"{len(photos)} photos without a cube → {args.out}")
