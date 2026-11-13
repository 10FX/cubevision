"""The reader's raw outputs (tools/diagnose.py) as frames read (eval.test.ts DUMP format), with thresholds
of one's own: what cubevision.decode reads.

    PYTHONPATH=. uv run python tools/raw_to_dump.py runs/diagnose-real runs/dumps-p03 --present 0.3 [--turning 0.5] [--hidden 0.5]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

FLOOR = 0.02


def faces_of(out: dict, present: float, turning: float, hidden: float) -> list[dict]:
    """As src/lib/vision/model.ts readingsFromOutput: faces readable enough, colours given seen, floored."""
    if float(np.ravel(out["turning"])[0]) > turning:
        return []
    faces = []
    stickers = np.asarray(out["stickers"])
    for face in range(6):
        if out["present"][face] < present:
            continue
        cells = []
        for cell in range(9):
            p = stickers[face, cell]
            if p[6] > hidden:
                cells.append(None)
                continue
            seen = 1 - p[6] or 1
            cells.append((FLOOR + (1 - 6 * FLOOR) * p[:6] / seen).round(4).tolist())
        faces.append({"center": face, "stickers": cells})
    return faces


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("raw", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--present", type=float, default=0.5)
    p.add_argument("--turning", type=float, default=0.5)
    p.add_argument("--hidden", type=float, default=0.5)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.raw.glob("*.raw.json")):
        raw = json.loads(path.read_text())
        frames = [{"t": r["t"], "faces": faces_of(r, args.present, args.turning, args.hidden), "orientation": np.ravel(r["orientation"]).tolist()} for r in raw]
        (args.out / path.name.replace(".raw.json", ".json")).write_text(json.dumps(frames))
    return 0


if __name__ == "__main__":
    sys.exit(main())
