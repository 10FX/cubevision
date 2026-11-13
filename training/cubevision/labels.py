"""
Label of one image, shared by the generator (which writes it) and the
training code (which reads it). See ml2/spec.json.

A face is identified by the colour of its centre (0..5 = U R F D L B in the
WCA scrambling orientation). For each face:

* ``present``: the face can be read: its centre and at least
  ``MIN_VISIBLE`` of its stickers are visible, it faces the camera, and no
  layer cuts through it (a side face of a layer being turned is not read);
* ``stickers``: for each of the 9 cells in StickerCube order, the colour
  seen, or ``HIDDEN``. Only meaningful when present. The model's reading may
  come out in any of the 4 rotations of the face: the loss takes the best.

``turning`` marks frames where a layer sits between two positions.

``box`` is the whole cube's box in the image (x0 y0 x1 y1, shares of the width
and height, y down, pieces hidden by a hand included, not clipped), for the
detector; None when there is no cube. ``orientation`` is the rotation (3×3,
row-major) taking the WCA directions of the centres (white +y, red +x, green
+z) to where they point in the camera's frame (x right, y up, z towards the
viewer), for the reader's orientation head.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

HIDDEN = 6
NUM_CLASSES = 7
#: Fewest visible stickers (centre included) for a face to count as readable
MIN_VISIBLE = 5


@dataclass
class FaceLabel:
    present: bool
    stickers: list[int]  # 9 values in 0..6

    def __post_init__(self):
        if len(self.stickers) != 9 or any(not 0 <= s <= HIDDEN for s in self.stickers):
            raise ValueError(f"bad stickers {self.stickers}")


@dataclass
class ImageLabel:
    image: str
    faces: list[FaceLabel]  # 6, by centre colour
    turning: bool
    meta: dict = field(default_factory=dict)
    box: list[float] | None = None
    orientation: list[float] | None = None

    def __post_init__(self):
        if len(self.faces) != 6:
            raise ValueError("6 faces expected")
        if self.box is not None and len(self.box) != 4:
            raise ValueError("box: x0 y0 x1 y1 expected")
        if self.orientation is not None and len(self.orientation) != 9:
            raise ValueError("orientation: 9 values expected")

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @staticmethod
    def from_json(line: str) -> ImageLabel:
        raw = json.loads(line)
        return ImageLabel(
            image=raw["image"],
            faces=[FaceLabel(bool(f["present"]), list(f["stickers"])) for f in raw["faces"]],
            turning=bool(raw["turning"]),
            meta=raw.get("meta", {}),
            box=raw.get("box"),
            orientation=raw.get("orientation"),
        )
