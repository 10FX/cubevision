"""Why frames of a video are lost: the app's detector and reader on every frame, each frame put in one
bin, and sheets of crops per bin to look at.

    PYTHONPATH=. uv run python tools/diagnose.py evals/synthetic/synthetic.json runs/diagnose-synthetic [--models ../public/models]
    PYTHONPATH=. uv run python tools/diagnose.py evals/videos.json runs/diagnose --decoded runs/decode.json

Also writes the reader's raw outputs per video (``<video>.raw.json``), to try other thresholds offline.

Bins: ``no box`` (no cube found nor followed), ``turning`` (the reader saw a layer mid-turn: not read),
``no face`` (no face readable enough), ``read``. Against the cube's state, when known (the truth of a
synthetic video; for a real one the decoder's path, cubevision.decode): ``turning`` is split in
``turning, truly`` and ``turning, wrongly`` (no layer turning then), and a frame read with a face two
stickers off or more is ``read wrong``; a frame read mid-turn is ``read mid-turn``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from video import Detector, Tracker, crop_of, frames_of, square  # noqa: E402

from cubevision.cube import ROT, Cube, invert_alg  # noqa: E402

PRESENT, TURNING, HIDDEN = 0.5, 0.5, 0.5
COLOURS = "WRGYOB"


class Reader:
    def __init__(self, models: Path):
        meta = json.loads((models / "cube-reader.json").read_text())
        self.size = meta["variants"]["lite"]["size"]
        self.session = ort.InferenceSession(str(models / meta["variants"]["lite"]["file"]), providers=["CPUExecutionProvider"])

    def read(self, crop: np.ndarray) -> dict:
        x = crop.astype(np.float32).transpose(2, 0, 1)[None] / 255
        names = [o.name for o in self.session.get_outputs()]
        return {k: v[0] for k, v in zip(names, self.session.run(None, {"image": x}), strict=True)}


def _state(alg: str) -> np.ndarray:
    return Cube().apply(alg).colors()


def truth_track(truth: dict) -> list[tuple[float, float, str]]:
    """(start, end, move) of a synthetic solve, in ms of the video."""
    return [(m["start"], m["end"], m["move"]) for m in truth["moves"]]


def state_at(scramble_alg: str, track: list[tuple[float, float, str]], t: float) -> tuple[np.ndarray, bool]:
    """The cube's colours at ``t`` and whether a layer is turning then (rotations x y z are not)."""
    done = [m for s, e, m in track if e <= t]
    turning = any(s < t < e and m[0] not in "xyz" for s, e, m in track)
    return _state(f"{scramble_alg} {' '.join(done)}".strip()), turning


def wrong_stickers(state: np.ndarray, out: dict) -> tuple[int, list[str]]:
    """Faces read whose stickers are two or more off the state (best rotation each), and what was read."""
    present, stickers = out["present"], out["stickers"]
    bad, notes = 0, []
    for c in range(6):
        if present[c] < PRESENT:
            continue
        # The face whose centre has colour c (a rotation may have moved it)
        f = next(p for p in range(6) if state[p * 9 + 4] == c)
        read = [None if stickers[c, i, 6] > HIDDEN else int(np.argmax(stickers[c, i, :6])) for i in range(9)]
        errors = min(sum(1 for i in range(9) if i != 4 and read[i] is not None and read[i] != state[f * 9 + ROT[r][i]]) for r in range(4))
        if errors >= 2:
            bad += 1
            notes.append(f"{COLOURS[c]}:{errors}")
    return bad, notes


def sheet(items: list[tuple[np.ndarray, str]], path: Path, cols: int = 8, size: int = 160) -> None:
    if not items:
        return
    rows = -(-len(items) // cols)
    img = Image.new("RGB", (cols * size, rows * (size + 14)), (30, 30, 30))
    draw = ImageDraw.Draw(img)
    for k, (crop, caption) in enumerate(items):
        x, y = (k % cols) * size, (k // cols) * (size + 14)
        img.paste(Image.fromarray(crop).resize((size, size)), (x, y))
        draw.text((x + 2, y + size), caption[:26], fill=(255, 255, 255))
    img.save(path)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("cases", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--models", type=Path, default=Path("../public/models"))
    p.add_argument("--decoded", type=Path, help="the decoder's moves (cubevision.decode) as the state of real videos")
    p.add_argument("--per-bin", type=int, default=32)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    detector, reader = Detector(args.models), Reader(args.models)
    decoded = json.loads(args.decoded.read_text()) if args.decoded else {}
    rng = np.random.default_rng(0)
    totals: Counter = Counter()
    seen: Counter = Counter()
    samples: dict[str, list[tuple[np.ndarray, str]]] = {}
    report = {}
    for case in json.loads(args.cases.read_text()):
        video = args.cases.parent / case["video"]
        frames, times = frames_of(video)
        track, base = None, case["scramble"]
        if case.get("truth"):
            track = truth_track(json.loads((args.cases.parent / case["truth"]).read_text()))
        elif case["video"] in decoded and decoded[case["video"]]["moves"]:
            d = decoded[case["video"]]
            hold = d.get("hold", "")
            base = f"{hold} {case['scramble']} {invert_alg(hold) if hold else ''}".strip()
            track = [(m["t"] - 100, m["t"], m["move"]) for m in d["moves"]]
        tracker, bins = Tracker(), Counter()
        raw: list[dict] = []
        for frame, t in zip(frames, times, strict=False):
            box = tracker.update(detector.detect(frame, tracker.box))
            if box is None:
                kind, crop, caption = "no box", np.asarray(Image.fromarray(frame).resize((reader.size, reader.size))), f"{t}"
            else:
                crop = square(frame, crop_of(box, detector.margin), reader.size)
                out = reader.read(crop)
                raw.append({"t": t, **{k: np.asarray(v).round(4).tolist() for k, v in out.items()}})
                state, mid = state_at(base, track, t) if track else (None, None)
                turning = float(np.ravel(out["turning"])[0])
                caption = f"{t} tu{turning:.2f} pr{out['present'].max():.2f}"
                if turning > TURNING:
                    kind = "turning" if mid is None else ("turning, truly" if mid else "turning, wrongly")
                elif out["present"].max() < PRESENT:
                    kind = "no face"
                elif state is None:
                    kind = "read"
                else:
                    bad, notes = wrong_stickers(state, out)
                    kind = "read mid-turn" if mid else ("read wrong" if bad else "read")
                    if bad:
                        caption = f"{t} {' '.join(notes)}"
            bins[kind] += 1
            # A random few per bin (reservoir): the crops of every frame would not fit in memory
            seen[kind] += 1
            kept = samples.setdefault(kind, [])
            item = (crop, f"{Path(case['video']).stem[-3:]} {caption}")
            if len(kept) < args.per_bin:
                kept.append(item)
            elif (j := int(rng.integers(seen[kind]))) < args.per_bin:
                kept[j] = item
        # The reader's outputs on every frame framed, to try other thresholds without the models (tools/raw_to_dump.py)
        (args.out / f"{Path(case['video']).stem}.raw.json").write_text(json.dumps(raw))
        totals.update(bins)
        report[case["video"]] = dict(bins)
        print(f"{case['video']}: " + ", ".join(f"{k} {v / sum(bins.values()):.0%}" for k, v in sorted(bins.items())), flush=True)
    n = sum(totals.values())
    print("all: " + ", ".join(f"{k} {v / n:.1%}" for k, v in sorted(totals.items())))
    for kind, items in samples.items():
        sheet(items, args.out / f"{kind.replace(', ', '-').replace(' ', '-')}.png")
    (args.out / "report.json").write_text(json.dumps({"videos": report, "all": dict(totals)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
