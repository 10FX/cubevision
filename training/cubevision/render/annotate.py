"""
Labels of a rendered image from what the renderer measured (no Blender here):
which of the 54 sticker tiles are visible, and how squarely each face looks
at the camera. See cubevision/labels.py for the meaning of the labels.
"""
from __future__ import annotations

from ..cube import parse_move
from ..labels import HIDDEN, MIN_VISIBLE, FaceLabel
from .sampler import Turn

#: Axis of each face's normal (0 x, 1 y, 2 z), faces U R F D L B
FACE_AXIS = (1, 0, 2, 1, 0, 2)
#: A face turned further from the camera than this (cosine) is too slanted to read
MIN_FACING = 0.25
#: Wild datasets read every sticker one can still see: a face down to ~2° from edge-on, a tile from about
#: 30 pixels of a 192 px crop (a sticker 36 px wide and under 1 px high), a face from its centre and one more
#: (3 stickers before relabel.py)
WILD_FACING = 0.03
WILD_AREA = 30 / 192**2
WILD_VISIBLE = 2


def annotate(stickers: list[int], turn: Turn | None, visible: list[bool], facing: list[float],
             area: list[float] | None = None, wild: bool = False, min_visible: int | None = None) -> tuple[list[FaceLabel], bool]:
    """
    ``stickers``: the state drawn (StickerCube order); ``visible``: per tile, in
    the same order (a tile keeps the index of its resting position even while
    its layer is turned); ``facing``: per face position, cosine between its
    normal and the direction of the camera; ``area``: per tile, its share of
    the image. ``wild``: read faces and tiles almost edge-on (WILD_*). ``min_visible``: stickers a face
    needs, if not the usual.
    """
    min_facing, usual = (WILD_FACING, WILD_VISIBLE) if wild else (MIN_FACING, MIN_VISIBLE)
    min_visible = min_visible or usual
    if wild and area is not None:
        visible = [v and a >= WILD_AREA for v, a in zip(visible, area, strict=True)]
    colors = [s // 9 for s in stickers]
    turning = bool(turn and turn.turning)
    broken: set[int] = set()
    if turning:
        axis = parse_move(turn.move).axis  # type: ignore[union-attr]
        # The four faces around the turning axis are cut in two
        broken = {f for f in range(6) if FACE_AXIS[f] != axis}

    faces: list[FaceLabel | None] = [None] * 6
    for f in range(6):
        cells = range(f * 9, f * 9 + 9)
        seen = [visible[i] for i in cells]
        present = f not in broken and seen[4] and sum(seen) >= min_visible and facing[f] >= min_facing
        # Keyed by the colour of the centre: a slice turn may have moved the centres
        faces[colors[f * 9 + 4]] = FaceLabel(present, [colors[i] if visible[i] else HIDDEN for i in cells])
    assert all(face is not None for face in faces)
    return faces, turning  # type: ignore[return-value]
