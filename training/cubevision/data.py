"""
Datasets for training: synthetic shards written by cubevision.render.generate.

Images are decoded on the CPU workers (with a random JPEG re-compression, as
a video codec would do); the heavier augmentations run batched on the GPU
(cubevision/augment.py). Labels are held as numpy arrays (LabelTable): the
DataLoader's workers would otherwise copy Python objects they only read.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg, read_file
from torchvision.transforms.v2 import functional as TF

from .geometry import placement_from_box
from .labels import ImageLabel


def read_labels(root: Path) -> list[ImageLabel]:
    labels: list[ImageLabel] = []
    for path in sorted(root.glob("labels-*.jsonl")):
        with path.open() as f:
            labels += [ImageLabel.from_json(line) for line in f if line.strip()]
    if not labels:
        raise FileNotFoundError(f"no labels-*.jsonl in {root}")
    return labels


def dataset_seed(root: Path) -> int | None:
    """Seed of a generated folder (its dataset.json), None if unknown."""
    manifest = root / "dataset.json"
    return json.loads(manifest.read_text())["seed"] if manifest.exists() else None


class LabelTable:
    """The labels of generated folders as numpy arrays. DataLoader workers start sharing the parent's
    memory, but reading a Python object writes its reference count, so each worker ends up with its
    own copy of every label it touched: 4 KB per label, 625 MB per process for 150 000 images.
    Arrays hold no Python objects: they stay shared (and take 0.4 KB per label)."""

    def __init__(self, roots: list[Path]):
        labels = [(root, label) for root in roots for label in read_labels(root)]
        self.paths = np.array([str(root / label.image).encode() for root, label in labels])
        self.present = np.array([[f.present for f in label.faces] for _, label in labels], dtype=np.float32).reshape(-1, 6)
        self.stickers = np.array([[f.stickers for f in label.faces] for _, label in labels], dtype=np.int8).reshape(-1, 6, 9)
        self.turning = np.array([label.turning for _, label in labels], dtype=np.float32)
        self.has_orientation = np.array([label.orientation is not None for _, label in labels], dtype=np.float32)
        identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.orientation = np.array([label.orientation or identity for _, label in labels], dtype=np.float32).reshape(-1, 3, 3)
        self.boxes = np.array([label.box or [np.nan] * 4 for _, label in labels], dtype=np.float32).reshape(-1, 4)
        # Where the cube sits in the image (geometry.py), when its box and orientation are known
        self.placement = np.array([placement_from_box(np.array(label.orientation, dtype=np.float32).reshape(3, 3), label.box)
                                   if label.box and label.orientation else [np.nan] * 3 for _, label in labels], dtype=np.float32).reshape(-1, 3)
        self.metas = np.array([json.dumps(label.meta, separators=(",", ":")).encode() for _, label in labels])

    def __len__(self) -> int:
        return len(self.paths)

    def path(self, i: int) -> str:
        return self.paths[i].decode()

    def box(self, i: int) -> list[float] | None:
        return None if np.isnan(self.boxes[i, 0]) else self.boxes[i].tolist()

    def meta(self, i: int) -> dict:
        return json.loads(self.metas[i])


class SyntheticDataset(Dataset):
    """Images and labels of one or more generated folders, resized to ``size``."""

    def __init__(self, roots: list[Path], size: int, *, train: bool, jpeg_p: float = 0.6, jpeg_quality: tuple[int, int] = (25, 92)):
        self.labels = LabelTable(roots)
        self.size = size
        self.train = train
        self.jpeg_p = jpeg_p
        self.jpeg_quality = jpeg_quality

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        t = self.labels
        image = decode_jpeg(read_file(t.path(i)), mode=ImageReadMode.RGB)
        if self.train and random.random() < self.jpeg_p:
            # Compression of the browser's recorder (VP8/VP9 blocks are coarser than JPEG's, so go low)
            image = decode_jpeg(encode_jpeg(image, quality=random.randint(*self.jpeg_quality)))
        if image.shape[-1] != self.size or image.shape[-2] != self.size:
            image = TF.resize(image, [self.size, self.size], antialias=True)
        return (image, torch.from_numpy(t.present[i]), torch.from_numpy(t.stickers[i]).long(), torch.tensor(t.turning[i]),
                torch.from_numpy(t.orientation[i]), torch.tensor(t.has_orientation[i]), torch.from_numpy(t.placement[i]))

    def meta(self, i: int) -> dict:
        return self.labels.meta(i)
