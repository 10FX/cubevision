"""
The cube detector: where the whole cube is in a video frame (the pieces hidden
by a hand included), so that the reader always gets the cube, well framed.

An existing, pretrained detector fine-tuned to one class: torchvision's
SSDLite320 with a MobileNetV3-Large backbone, trained on COCO (BSD-3 licence,
3.4 M parameters, 0.6 GFLOPs at 320 px). The COCO box regression is kept, the
classifier is replaced by "cube or background". Ultralytics' YOLO nano models
would do as well but are AGPL-3.0, which would bind the site's code.

* input: the frame letterboxed into a 320×320 square (longer side fitted,
  bars of grey 0.5), RGB in 0..1;
* outputs (see ExportWrapper): the best box, x0 y0 x1 y1 as shares of the
  square (may go past its edges: the box is amodal), and its score.

The app crops a square of the box's longer side times CROP_MARGIN around its
centre for the reader (src/lib/vision/detector.ts), which is how the reader's
training crops are framed (render/sampler.py).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg, read_file
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
from torchvision.transforms.v2 import functional as TF

from .data import LabelTable

#: The reader's crop: a square of the box's longer side times this, around its centre
CROP_MARGIN = 1.5
#: Grey of the letterbox bars
PAD = 0.5
#: Share of the cube's box that must be in the frame for it to count as there
MIN_INSIDE = 0.3


@dataclass
class DetectorConfig:
    size: int = 320
    pretrained: bool = True
    #: Smallest anchor, share of the input (SSDLite's 0.2 is 64 px: a cube spanning a tenth of the frame,
    #: 32 px, is found poorly); the anchors per location stay the same, so the COCO head still loads
    min_scale: float = 0.2


def build_detector(cfg: DetectorConfig) -> nn.Module:
    """SSDLite320-MobileNetV3, COCO weights, a fresh 2-class head (background, cube)."""
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1 if cfg.pretrained else None
    model = ssdlite320_mobilenet_v3_large(weights=weights, weights_backbone=None)
    # torchvision resizes every image to 320 before the backbone: at another size, trained as the browser runs it
    model.transform.min_size, model.transform.max_size, model.transform.fixed_size = (cfg.size,), cfg.size, (cfg.size, cfg.size)
    if cfg.min_scale != 0.2:
        model.anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=cfg.min_scale, max_ratio=0.95)
    in_channels = det_utils.retrieve_out_channels(model.backbone, (cfg.size, cfg.size))
    anchors = model.anchor_generator.num_anchors_per_location()
    norm = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    model.head.classification_head = SSDLiteClassificationHead(in_channels, anchors, 2, norm)
    return model


class ExportWrapper(nn.Module):
    """What the browser runs: the best box and its score (no NMS: there is one cube)."""

    def __init__(self, model: nn.Module, size: int):
        super().__init__()
        self.backbone, self.head = model.backbone, model.head
        self.size = size
        self.weights = tuple(float(w) for w in model.box_coder.weights)
        self.clip = float(model.box_coder.bbox_xform_clip)
        self.register_buffer("mean", torch.tensor(model.transform.image_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(model.transform.image_std).view(1, 3, 1, 1), persistent=False)
        with torch.no_grad():
            x = torch.zeros(1, 3, size, size, device=next(model.parameters()).device)
            features = list(model.backbone(x).values())
            anchors = model.anchor_generator(ImageList(x, [(size, size)]), features)[0]
        self.register_buffer("anchors", anchors, persistent=False)  # [A, 4], pixels

    def decode(self, rel: torch.Tensor) -> torch.Tensor:
        """Box regression [N, A, 4] → boxes [N, A, 4] in pixels (torchvision's BoxCoder, batched)."""
        a = self.anchors
        w, h = a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]
        cx, cy = a[:, 0] + 0.5 * w, a[:, 1] + 0.5 * h
        wx, wy, ww, wh = self.weights
        dx, dy = rel[..., 0] / wx, rel[..., 1] / wy
        dw, dh = (rel[..., 2] / ww).clamp(max=self.clip), (rel[..., 3] / wh).clamp(max=self.clip)
        pcx, pcy = dx * w + cx, dy * h + cy
        pw, ph = torch.exp(dw) * w, torch.exp(dh) * h
        return torch.stack([pcx - 0.5 * pw, pcy - 0.5 * ph, pcx + 0.5 * pw, pcy + 0.5 * ph], -1)

    def forward(self, image: torch.Tensor):
        """``image``: [N, 3, S, S] in 0..1. Returns box [N, 4] (shares of the square) and score [N, 1]."""
        x = (image - self.mean) / self.std
        out = self.head(list(self.backbone(x).values()))
        scores = torch.softmax(out["cls_logits"], -1)[..., 1]  # [N, A]
        best = scores.argmax(-1, keepdim=True)  # [N, 1]
        boxes = self.decode(out["bbox_regression"])
        box = torch.gather(boxes, 1, best.unsqueeze(-1).expand(-1, -1, 4)).squeeze(1) / self.size
        return box, torch.gather(scores, 1, best)


def letterbox(image: torch.Tensor, size: int) -> tuple[torch.Tensor, float, int, int]:
    """uint8 [3, H, W] → uint8 [3, size, size], the longer side fitted, grey bars; and the scale and offsets."""
    _, h, w = image.shape
    scale = size / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = TF.resize(image, [nh, nw], antialias=True)
    out = torch.full((3, size, size), round(PAD * 255), dtype=torch.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    out[:, oy:oy + nh, ox:ox + nw] = resized
    return out, scale, ox, oy


def inside_share(box: list[float]) -> float:
    """Share of a box (shares of the image) that lies in the image."""
    x0, y0, x1, y1 = box
    area = max(1e-9, (x1 - x0) * (y1 - y0))
    return max(0.0, min(x1, 1) - max(x0, 0)) * max(0.0, min(y1, 1) - max(y0, 0)) / area


class DetectionDataset(Dataset):
    """Full frames (render --framing full), letterboxed; the box in pixels of the square (or none).

    Training adds a mirror and a random zoom and shift (the photometric changes run on the GPU,
    cubevision/augment.py)."""

    def __init__(self, roots: list[Path], size: int, *, train: bool, zoom: tuple[float, float] = (0.6, 1.6), shift: float = 0.25,
                 far: float = 0.0):
        """``far``: share of training frames zoomed out to a quarter to 0.6 (a small cube, far from the camera)."""
        self.labels = LabelTable(roots)
        self.size, self.train, self.zoom, self.shift, self.far = size, train, zoom, shift, far

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        image = decode_jpeg(read_file(self.labels.path(i)), mode=ImageReadMode.RGB)
        _, h, w = image.shape
        square, scale, ox, oy = letterbox(image, self.size)
        box = None
        cube = self.labels.box(i)
        if cube is not None and inside_share(cube) >= MIN_INSIDE:
            x0, y0, x1, y1 = cube
            box = [x0 * w * scale + ox, y0 * h * scale + oy, x1 * w * scale + ox, y1 * h * scale + oy]
        if self.train:
            if random.random() < 0.5:
                square = TF.horizontal_flip(square)
                if box:
                    box = [self.size - box[2], box[1], self.size - box[0], box[3]]
            far = random.random() < self.far
            if far or random.random() < 0.7:
                # Zoom and shift about the centre: p → c + s (p - c) + t
                s = random.uniform(0.25, 0.6) if far else random.uniform(*self.zoom)
                tx, ty = (round(random.uniform(-self.shift, self.shift) * self.size) for _ in range(2))
                square = TF.affine(square, angle=0.0, translate=[tx, ty], scale=s, shear=[0.0, 0.0], fill=round(PAD * 255))
                if box:
                    c = self.size / 2
                    box = [c + s * (box[0] - c) + tx, c + s * (box[1] - c) + ty, c + s * (box[2] - c) + tx, c + s * (box[3] - c) + ty]
                    if inside_share([v / self.size for v in box]) < MIN_INSIDE:
                        box = None
        target = torch.tensor([box], dtype=torch.float32) if box else torch.zeros((0, 4), dtype=torch.float32)
        return square, target


def collate(batch):
    images, boxes = zip(*batch, strict=True)
    return torch.stack(images), list(boxes)


def crop_of(box: torch.Tensor) -> torch.Tensor:
    """The reader's square crop around boxes [N, 4]: side = longer side × CROP_MARGIN."""
    cx, cy = (box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2
    half = torch.maximum(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]) * CROP_MARGIN / 2
    return torch.stack([cx - half, cy - half, cx + half, cy + half], 1)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU of matching rows of [N, 4] boxes."""
    ix = (torch.minimum(a[:, 2], b[:, 2]) - torch.maximum(a[:, 0], b[:, 0])).clamp(min=0)
    iy = (torch.minimum(a[:, 3], b[:, 3]) - torch.maximum(a[:, 1], b[:, 1])).clamp(min=0)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]).clamp(min=0) * (x[:, 3] - x[:, 1]).clamp(min=0)  # noqa: E731
    return inter / (area(a) + area(b) - inter).clamp(min=1e-9)


class DetectionMetrics:
    """What matters downstream: is there a cube, and does the crop hold all of it at the size the reader expects."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.c = dict.fromkeys(["images", "positives", "tp", "fp", "fn", "iou", "iou50", "iou75", "crop_ok"], 0.0)

    @torch.no_grad()
    def update(self, box: torch.Tensor, score: torch.Tensor, targets: list[torch.Tensor], size: int) -> None:
        c = self.c
        found = score.view(-1) >= self.threshold
        has = torch.tensor([len(t) > 0 for t in targets], device=box.device)
        c["images"] += len(targets)
        c["positives"] += has.sum().item()
        c["tp"] += (found & has).sum().item()
        c["fp"] += (found & ~has).sum().item()
        c["fn"] += (~found & has).sum().item()
        if not has.any():
            return
        truth = torch.stack([t[0] for t in targets if len(t)]).to(box.device) / size
        pred = box[has]
        iou = box_iou(pred, truth) * found[has]
        c["iou"] += iou.sum().item()
        c["iou50"] += (iou >= 0.5).sum().item()
        c["iou75"] += (iou >= 0.75).sum().item()
        # The crop holds (almost) all of the cube, which fills it as in the reader's training
        crop = crop_of(pred)
        held = box_iou(torch.stack([torch.maximum(crop[:, 0], truth[:, 0]), torch.maximum(crop[:, 1], truth[:, 1]),
                                    torch.minimum(crop[:, 2], truth[:, 2]), torch.minimum(crop[:, 3], truth[:, 3])], 1), truth)
        fill = torch.maximum(truth[:, 2] - truth[:, 0], truth[:, 3] - truth[:, 1]) / (crop[:, 2] - crop[:, 0]).clamp(min=1e-9)
        c["crop_ok"] += ((held >= 0.98) & (fill >= 0.5) & (fill <= 0.85) & found[has]).sum().item()

    def summary(self) -> dict[str, float]:
        c = self.c
        div = lambda a, b: a / b if b else float("nan")  # noqa: E731
        return {
            "precision": div(c["tp"], c["tp"] + c["fp"]),
            "recall": div(c["tp"], c["tp"] + c["fn"]),
            "iou": div(c["iou"], c["positives"]),
            "recall_iou50": div(c["iou50"], c["positives"]),
            "recall_iou75": div(c["iou75"], c["positives"]),
            "crop_ok": div(c["crop_ok"], c["positives"]),
        }
