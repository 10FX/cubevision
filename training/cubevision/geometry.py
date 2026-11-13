"""
Where each sticker is in a crop. The reader learns the cube's orientation in
the camera's frame and where the cube sits in the crop (its ``placement``: the
image position of the cube's centre and its scale); together they say where
all 54 stickers are, and the reader reads each colour there, instead of
searching the image for it.

The view is orthographic: the cube is small and far from the camera compared
with its size (a crop's perspective moves a sticker by a few percent of the
cube at most). Units: the cube spans -1..1 along each axis, x R, y U, z F
(the WCA directions of the centres, as the orientation labels).
"""
from __future__ import annotations

import numpy as np
import torch

from .cube import POSITIONS

#: Sticker centres in the cube's frame (54, 3): a cubie is 2/3 wide, the faces at ±1
STICKERS = np.array(POSITIONS, dtype=np.float32) / 3.0
CORNERS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=np.float32)


def _flat(points: np.ndarray, orientation: np.ndarray) -> np.ndarray:
    """Cube-frame points → image directions (x right, y down) of the oriented cube, orthographic."""
    cam = points @ orientation.T
    return np.stack([cam[:, 0], -cam[:, 1]], 1)


def placement_from_box(orientation: np.ndarray, box: list[float]) -> np.ndarray:
    """(cx, cy, s): the cube centre's position (shares of the image) and the image share per cube unit,
    fitted so that the oriented cube's corners span the box (hidden pieces included, as the labels')."""
    corners = _flat(CORNERS, orientation)
    lo, hi = corners.min(0), corners.max(0)
    x0, y0, x1, y1 = box
    scale = np.array([x1 - x0, y1 - y0]) / np.maximum(hi - lo, 1e-6)
    s = float(scale.mean())
    centre = np.array([x0, y0]) - lo * scale
    return np.array([centre[0], centre[1], s], dtype=np.float32)


def project(orientation: torch.Tensor, placement: torch.Tensor) -> torch.Tensor:
    """Orientations [B, 3, 3] and placements [B, 3] → sticker centres [B, 54, 2] (shares of the image)."""
    stickers = torch.as_tensor(STICKERS, device=orientation.device, dtype=orientation.dtype)
    cam = stickers @ orientation.transpose(1, 2)  # [B, 54, 3]
    flat = torch.stack([cam[..., 0], -cam[..., 1]], -1)
    return placement[:, None, :2] + placement[:, None, 2:3] * flat


#: Outward normals of the faces U R F D L B (the cube's frame)
NORMALS = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1], [0, -1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.float32)


def _symmetries() -> tuple[np.ndarray, np.ndarray]:
    """The 24 rotations that map the cube onto itself (3×3, entries 0 and ±1, determinant 1), and for
    each, where it sends each face: SYMMETRIES[g] @ NORMALS[p] = NORMALS[FACE_PERM[g, p]]."""
    mats = []
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        for signs in np.ndindex(2, 2, 2):
            m = np.zeros((3, 3), dtype=np.float32)
            for row, col in enumerate(perm):
                m[row, col] = 1 - 2 * signs[row]
            if np.linalg.det(m) > 0:
                mats.append(m)
    mats_arr = np.stack(mats)
    face_perm = np.array([[int(np.argmax(NORMALS @ (m @ n))) for n in NORMALS] for m in mats_arr])
    return mats_arr, face_perm


SYMMETRIES, FACE_PERM = _symmetries()


def _cells() -> np.ndarray:
    """[24, 54]: for each symmetry g, the sticker of the pose at each sticker of the coloured cube
    (StickerCube order): the pose's point y is the cube's SYMMETRIES[g] @ y."""
    index = {tuple(int(v) for v in p): i for i, p in enumerate(POSITIONS)}
    return np.array([[index[tuple(int(v) for v in np.rint(m.T @ np.array(p)))] for p in POSITIONS] for m in SYMMETRIES])


CELLS = _cells()


def nearest_symmetric(orientation: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Of the 24 orientations [B, 3, 3] that put the cube's faces where ``orientation`` does (it composed
    with the cube's symmetries), the one nearest ``reference``; and which symmetry it is [B]."""
    sym = torch.as_tensor(SYMMETRIES, device=orientation.device, dtype=orientation.dtype)
    candidates = orientation[:, None] @ sym[None]  # [B, 24, 3, 3]
    g = ((candidates - reference[:, None]) ** 2).sum((-1, -2)).argmin(1)
    return candidates[torch.arange(len(g), device=g.device), g], g
