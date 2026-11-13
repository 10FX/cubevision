"""Frames of a video and the shipped detector run on them, as the app does (for the tools)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

SMALL_CUBE, ZOOM = 0.25, 4


def frames_of(video: Path, width: int = 1280) -> tuple[np.ndarray, list[float]]:
    probe = ["ffprobe", "-v", "error", "-select_streams", "v", "-of", "csv=p=0", "-show_entries"]
    w, h = (int(v) for v in subprocess.check_output([*probe, "stream=width,height", str(video)], text=True).strip().split(","))
    width = min(width, w)
    height = round(h * width / w / 2) * 2
    times = [round(float(x) * 1000) for x in subprocess.check_output([*probe, "frame=pts_time", str(video)], text=True).split()]
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"scale={width}:{height}", "-vsync", "0",
                                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
    frames = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)
    return frames, times[:len(frames)]


def square(frame: np.ndarray, box: tuple[float, float, float, float], size: int, pad: int = 128) -> np.ndarray:
    """The region (x, y, w, h, may pass the frame's edges) resized to size×size, grey beyond the frame."""
    x, y, w, h = (round(v) for v in box)
    w, h = max(1, w), max(1, h)
    canvas = np.full((h, w, 3), pad, np.uint8)
    fx0, fy0 = max(0, x), max(0, y)
    fx1, fy1 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if fx1 > fx0 and fy1 > fy0:
        canvas[fy0 - y:fy1 - y, fx0 - x:fx1 - x] = frame[fy0:fy1, fx0:fx1]
    # Area average when shrinking, as the app's letterbox
    return np.asarray(Image.fromarray(canvas).resize((size, size), Image.BOX if w > size else Image.BILINEAR))


class Detector:
    def __init__(self, models: Path):
        meta = json.loads((models / "cube-detector.json").read_text())
        self.size, self.threshold, self.margin = meta["input"]["size"], meta["threshold"], meta["cropMargin"]
        self.session = ort.InferenceSession(str(models / meta["variants"]["lite"]["file"]), providers=["CPUExecutionProvider"])

    def _run(self, frame: np.ndarray, region: tuple[float, float, float, float]):
        x, y, w, h = region
        side = max(w, h)
        sq = (x + w / 2 - side / 2, y + h / 2 - side / 2, side, side)
        img = square(frame, sq, self.size).astype(np.float32).transpose(2, 0, 1)[None] / 255
        out = dict(zip([o.name for o in self.session.get_outputs()], self.session.run(None, {"image": img}), strict=True))
        score = float(np.ravel(out["score"])[0])
        if score < self.threshold:
            return None
        x0, y0, x1, y1 = np.ravel(out["box"])
        return (sq[0] + x0 * side, sq[1] + y0 * side, (x1 - x0) * side, (y1 - y0) * side)

    def detect(self, frame: np.ndarray, hint):
        h, w = frame.shape[:2]
        found = self._run(frame, (0, 0, w, h))
        near = found or hint
        if not near or max(near[2], near[3]) >= SMALL_CUBE * max(w, h):
            return found
        side = max(near[2], near[3]) * ZOOM
        return self._run(frame, (near[0] + near[2] / 2 - side / 2, near[1] + near[3] / 2 - side / 2, side, side)) or found


class Tracker:
    """BoxTracker of the app: smoothed, kept a few frames when the cube is lost."""

    def __init__(self, keep: int = 8, follow: float = 0.6):
        self.box, self.missed, self.keep, self.follow = None, 0, keep, follow

    def update(self, found):
        if not found:
            self.missed += 1
            if self.box and self.missed > self.keep:
                self.box = None
            return self.box
        self.missed = 0
        if self.box is None or _iou(self.box, found) < 0.3:
            self.box = found
        else:
            self.box = tuple(a + (b - a) * self.follow for a, b in zip(self.box, found, strict=True))
        return self.box


def _iou(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy / (a[2] * a[3] + b[2] * b[3] - ix * iy or 1)


def crop_of(box, margin: float):
    side = max(box[2], box[3]) * margin
    return (box[0] + box[2] / 2 - side / 2, box[1] + box[3] / 2 - side / 2, side, side)

