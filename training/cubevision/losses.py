"""
Loss and metrics of the cube reader.

A face's reading may come out in any of its 4 rotations (which corner is
"first" is not always knowable from one image, and the app's search tries
all four anyway): the sticker loss of a face is the smallest over the four
rotations of its target, the usual trick for outputs defined up to a
symmetry.

The orientation has no such ambiguity (the six centres differ): its loss is
the squared chordal distance between rotations (smooth, and equivalent to the
geodesic angle for small errors); the metrics report the angle itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .cube import ROT
from .geometry import FACE_PERM, nearest_symmetric
from .labels import HIDDEN, NUM_CLASSES
from .model import OUTER_CELLS, rotation_from_6d

# For rotation r and output cell j (one of the 8 outer cells): the target cell it must match
_PERM = torch.tensor([[ROT[r][c] for c in OUTER_CELLS] for r in range(4)], dtype=torch.long)  # [4, 8]
# The same for readers that read all 9 cells (symmetric: the centre too, which tells the face)
_PERM9 = torch.tensor([[ROT[r][c] for c in range(9)] for r in range(4)], dtype=torch.long)  # [4, 9]


@dataclass
class LossWeights:
    present: float = 1.0
    stickers: float = 1.0
    turning: float = 0.3
    orientation: float = 0.5
    label_smoothing: float = 0.05
    #: "best": a face's stickers scored at the rotation that fits best (a reading defined up to a quarter
    #: turn); "exact": in StickerCube order, which the cube's orientation fixes. Under "best", one colour
    #: on every cell fits every rotation alike: the reader settled there, reading no sticker (44 %)
    sticker_rotation: str = "best"
    #: The cube's placement in the crop (geometric reader): L1 on its centre and relative scale
    placement: float = 2.0


def rotated_targets(stickers: torch.Tensor, cells: int = 8) -> torch.Tensor:
    """[B, 6, 9] → [B, 6, 4, cells]: the outer cells' (or all 9 cells') targets for each of the 4 rotations."""
    return stickers[:, :, (_PERM if cells == 8 else _PERM9).to(stickers.device)]


def sticker_ce(logits: torch.Tensor, stickers: torch.Tensor, smoothing: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-face loss at the best rotation [B, 6], and that rotation [B, 6]. Logits [B, 6, 8 or 9, C]."""
    b, cells = logits.shape[0], logits.shape[2]
    targets = rotated_targets(stickers, cells)  # [B, 6, 4, cells]
    expanded = logits.unsqueeze(2).expand(-1, -1, 4, -1, -1)  # [B, 6, 4, cells, C]
    ce = F.cross_entropy(expanded.reshape(-1, NUM_CLASSES), targets.reshape(-1), reduction="none", label_smoothing=smoothing)
    per_rotation = ce.view(b, 6, 4, cells).sum(-1)  # [B, 6, 4]
    best, rotation = per_rotation.min(-1)
    return best, rotation


def rotation_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle (radians) of the rotation between matching rotation matrices [..., 3, 3]."""
    cos = ((a.transpose(-1, -2) @ b).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.acos(cos.clamp(-1.0, 1.0))


def reader_loss(out: dict[str, torch.Tensor], present: torch.Tensor, stickers: torch.Tensor, turning: torch.Tensor, w: LossWeights,
                orientation: torch.Tensor | None = None, has_orientation: torch.Tensor | None = None,
                placement: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    symmetric = out["stickers"].shape[2] == 9
    face_known = torch.ones_like(present, dtype=torch.bool)
    if symmetric and orientation is not None and has_orientation is not None:
        # Faces of the predicted pose: each shows the true face the nearest symmetric pose puts there
        _, g = nearest_symmetric(orientation.float(), rotation_from_6d(out["orientation"].float()).detach())
        shown = torch.as_tensor(FACE_PERM, device=present.device)[g]  # [B, 6]
        present, stickers = present.gather(1, shown), stickers.gather(1, shown[..., None].expand(-1, -1, 9))
        face_known = (has_orientation > 0.5)[:, None].expand_as(present)
    l_present = F.binary_cross_entropy_with_logits(out["present"].float()[face_known], present[face_known])
    if w.sticker_rotation == "exact":
        logits = out["stickers"].float()
        targets = rotated_targets(stickers)[:, :, 0]  # [B, 6, 8], rotation 0: StickerCube order
        best = F.cross_entropy(logits.reshape(-1, NUM_CLASSES), targets.reshape(-1), reduction="none",
                               label_smoothing=w.label_smoothing).view(logits.shape[0], 6, -1).sum(-1)
    else:
        best, _ = sticker_ce(out["stickers"].float(), stickers, w.label_smoothing)
    mask = (present > 0.5) & face_known
    if "placement" in out:
        # Read where the geometry says: a layer mid-turn is not there (its frames are ignored by the app anyway)
        mask = mask & (turning < 0.5)[:, None]
    # Per present face, averaged over its cells
    l_stickers = (best[mask] / out["stickers"].shape[2]).mean() if mask.any() else best.sum() * 0.0
    l_turning = F.binary_cross_entropy_with_logits(out["turning"].float(), turning)
    total = w.present * l_present + w.stickers * l_stickers + w.turning * l_turning
    losses = {"present": l_present.detach(), "stickers": l_stickers.detach(), "turning": l_turning.detach()}
    if orientation is not None and has_orientation is not None and "orientation" in out:
        predicted = rotation_from_6d(out["orientation"].float())
        target = nearest_symmetric(orientation.float(), predicted.detach())[0] if symmetric else orientation
        chordal = ((predicted - target) ** 2).sum((-1, -2))  # [B]
        known = has_orientation > 0.5
        l_orientation = chordal[known].mean() if known.any() else chordal.sum() * 0.0
        total = total + w.orientation * l_orientation
        losses["orientation"] = l_orientation.detach()
    if placement is not None and "placement" in out:
        known = ~torch.isnan(placement).any(-1)
        if known.any():
            pred, true = out["placement"].float()[known], placement[known]
            l_placement = (pred[:, :2] - true[:, :2]).abs().sum(-1).mean() + (pred[:, 2] / true[:, 2]).log().abs().mean()
        else:
            l_placement = out["placement"].sum() * 0.0
        total = total + w.placement * l_placement
        losses["placement"] = l_placement.detach()
    return {"loss": total, **losses}


@dataclass
class Metrics:
    """Accumulated over batches; ``summary()`` gives the rates."""
    counts: dict[str, float] = field(default_factory=lambda: dict.fromkeys(
        ["images", "faces", "present_tp", "present_fp", "present_fn", "present_ok", "read_faces",
         "cells", "cells_ok", "visible", "visible_ok", "hidden", "hidden_ok", "face_exact", "turning_ok",
         "two_images", "two_right"], 0.0))
    #: Orientation errors, degrees (one per image whose orientation is known)
    angles: list[float] = field(default_factory=list)

    @torch.no_grad()
    def update(self, out: dict[str, torch.Tensor], present: torch.Tensor, stickers: torch.Tensor, turning: torch.Tensor,
               orientation: torch.Tensor | None = None, has_orientation: torch.Tensor | None = None) -> None:
        c = self.counts
        if orientation is not None and has_orientation is not None and "orientation" in out:
            known = has_orientation > 0.5
            if known.any():
                angle = rotation_angle(rotation_from_6d(out["orientation"].float()[known]), orientation[known].float())
                self.angles += torch.rad2deg(angle).tolist()
        pred_present = out["present"] > 0
        truth = present > 0.5
        c["images"] += present.shape[0]
        c["faces"] += truth.numel()
        c["present_ok"] += (pred_present == truth).sum().item()
        c["present_tp"] += (pred_present & truth).sum().item()
        c["present_fp"] += (pred_present & ~truth).sum().item()
        c["present_fn"] += (~pred_present & truth).sum().item()
        c["turning_ok"] += ((out["turning"] > 0) == (turning > 0.5)).sum().item()
        # Stickers of the present faces, at the rotation that fits best
        _, rotation = sticker_ce(out["stickers"].float(), stickers)
        targets = rotated_targets(stickers)  # [B, 6, 4, 8]
        best = torch.gather(targets, 2, rotation[..., None, None].expand(-1, -1, 1, len(OUTER_CELLS))).squeeze(2)  # [B, 6, 8]
        pred = out["stickers"].argmax(-1)  # [B, 6, 8]
        ok = (pred == best)[truth]  # [F, 8]
        tgt = best[truth]
        c["read_faces"] += truth.sum().item()
        c["cells"] += ok.numel()
        c["cells_ok"] += ok.sum().item()
        visible = tgt != HIDDEN
        c["visible"] += visible.sum().item()
        c["visible_ok"] += (ok & visible).sum().item()
        c["hidden"] += (~visible).sum().item()
        c["hidden_ok"] += (ok & ~visible).sum().item()
        c["face_exact"] += ok.all(-1).sum().item()
        # What the decoder needs: two faces or more read right (seen as there, every sticker seen right), when the
        # cube is still and shows two
        right = pred_present & truth & ((pred == best) | (best == HIDDEN)).all(-1)  # [B, 6]
        still = (turning < 0.5) & (truth.sum(1) >= 2)
        c["two_images"] += still.sum().item()
        c["two_right"] += (still & (right.sum(1) >= 2)).sum().item()

    def summary(self) -> dict[str, float]:
        c = self.counts
        div = lambda a, b: a / b if b else float("nan")  # noqa: E731
        precision = div(c["present_tp"], c["present_tp"] + c["present_fp"])
        recall = div(c["present_tp"], c["present_tp"] + c["present_fn"])
        return {
            "present_acc": div(c["present_ok"], c["faces"]),
            "present_precision": precision,
            "present_recall": recall,
            "sticker_acc": div(c["cells_ok"], c["cells"]),
            "visible_sticker_acc": div(c["visible_ok"], c["visible"]),
            "hidden_acc": div(c["hidden_ok"], c["hidden"]),
            "face_exact": div(c["face_exact"], c["read_faces"]),
            "turning_acc": div(c["turning_ok"], c["images"]),
            "two_faces_right": div(c["two_right"], c["two_images"]),
            "orientation_median_deg": float(torch.tensor(self.angles).median()) if self.angles else float("nan"),
            "orientation_within_15deg": div(sum(a <= 15 for a in self.angles), len(self.angles)),
        }
