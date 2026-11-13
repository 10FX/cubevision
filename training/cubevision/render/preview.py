"""
Contact sheet of a dataset with its labels drawn beside each image, to check
by eye that the labels say what the images show, and statistics of what the
dataset holds (to compare with real footage and tune the sampler).

    python -m cubevision.render.preview data/synth/train --count 24 --out preview.png
    python -m cubevision.render.preview data/synth/train --stats
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

from cubevision.labels import HIDDEN, ImageLabel

SWATCH = [(240, 240, 240), (200, 20, 30), (20, 170, 70), (250, 220, 20), (255, 120, 10), (20, 70, 220), (0, 0, 0)]


def load_labels(root: Path) -> list[ImageLabel]:
    labels = []
    for path in sorted(root.glob("labels-*.jsonl")):
        labels += [ImageLabel.from_json(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(labels, key=lambda label: label.meta.get("index", 0))


def draw_label(label: ImageLabel, size: int) -> Image.Image:
    """The present faces as small 3×3 grids (hidden stickers black), and the turning flag."""
    panel = Image.new("RGB", (size, size), (40, 40, 40))
    d = ImageDraw.Draw(panel)
    cell = size // 13
    x, y = 4, 4
    for color, face in enumerate(label.faces):
        if not face.present:
            continue
        for i, s in enumerate(face.stickers):
            r, c = divmod(i, 3)
            box = (x + c * cell, y + r * cell, x + (c + 1) * cell - 1, y + (r + 1) * cell - 1)
            d.rectangle(box, fill=SWATCH[s], outline=(90, 90, 90))
            if s == HIDDEN:
                d.line(box, fill=(120, 120, 120))
        d.rectangle((x - 2, y - 2, x + 3 * cell + 1, y + 3 * cell + 1), outline=SWATCH[color], width=2)
        x += 4 * cell
        if x + 3 * cell > size:
            x, y = 4, y + 4 * cell
    if label.turning:
        d.text((4, size - 14), "turning", fill=(255, 200, 0))
    d.text((size - 70, size - 14), f"{label.meta.get('index', '')}", fill=(160, 160, 160))
    return panel


def draw_geometry(image: Image.Image, label: ImageLabel, size: int) -> Image.Image:
    """The image fitted in a size×size square, with the cube's box and its orientation drawn:
    from the box's centre, where the white (+y), red (+x) and green (+z) centres point."""
    w, h = image.size
    scale = size / max(w, h)
    out = Image.new("RGB", (size, size), (20, 20, 20))
    ox, oy = (size - round(w * scale)) // 2, (size - round(h * scale)) // 2
    out.paste(image.resize((round(w * scale), round(h * scale))), (ox, oy))
    if label.box is None:
        return out
    d = ImageDraw.Draw(out)
    x0, y0, x1, y1 = label.box
    box = (ox + x0 * w * scale, oy + y0 * h * scale, ox + x1 * w * scale, oy + y1 * h * scale)
    d.rectangle(box, outline=(255, 0, 255), width=1)
    if label.orientation is not None:
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        reach = (box[2] - box[0]) / 2
        m = label.orientation
        for axis, color in ((1, SWATCH[0]), (0, SWATCH[1]), (2, SWATCH[2])):
            # Column ``axis``: that centre's direction in the camera's frame (y up)
            d.line((cx, cy, cx + m[axis] * reach, cy - m[3 + axis] * reach), fill=color, width=2)
    return out


def stats(labels: list[ImageLabel]) -> dict:
    """Shares of images per number of readable faces, turning, and per kind of scene."""
    n = len(labels)
    share = lambda counter: {k: round(v / n, 3) for k, v in sorted(counter.items(), key=lambda kv: str(kv[0]))}  # noqa: E731
    read = [f for label in labels for f in label.faces if f.present]
    return {
        "images": n,
        "readable_faces": share(Counter(sum(f.present for f in label.faces) for label in labels)),
        "turning": round(sum(label.turning for label in labels) / n, 3),
        "hidden_stickers_of_readable_faces": round(sum(s == HIDDEN for f in read for s in f.stickers) / max(1, 9 * len(read)), 3),
        **{key: share(Counter(label.meta.get(key) for label in labels))
           for key in ("style", "hands", "background", "centres", "lighting", "framing", "body", "cube")},
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--count", type=int, default=24)
    p.add_argument("--columns", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("preview.png"))
    p.add_argument("--stats", action="store_true", help="print the dataset's statistics instead")
    args = p.parse_args()
    if args.stats:
        import json

        print(json.dumps(stats(load_labels(args.root)), indent=2))
        return
    labels = load_labels(args.root)[: args.count]
    size = max(Image.open(args.root / labels[0].image).size)
    rows = -(-len(labels) // args.columns)
    sheet = Image.new("RGB", (args.columns * size * 2, rows * size), (0, 0, 0))
    for k, label in enumerate(labels):
        r, c = divmod(k, args.columns)
        sheet.paste(draw_geometry(Image.open(args.root / label.image).convert("RGB"), label, size), (c * size * 2, r * size))
        sheet.paste(draw_label(label, size), (c * size * 2 + size, r * size))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out} ({len(labels)} images)")


if __name__ == "__main__":
    main()
