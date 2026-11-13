"""
The cube reader: one cropped image of the cube in, for each of the six faces
(named by their centre colour) whether it can be read and the colour of its
nine stickers, plus whether a layer is mid-turn. See ml2/spec.json.

Architecture, sized for browsers (WebGPU / WASM):

* backbone: ImageNet MobileNetV3 (large or small), features at strides 16
  and 32 fused into a 12×12 grid of tokens (at 192 px), or with the stride-8
  features too into a 24×24 grid (``token_stride: 8``);
* six learned face queries attend to the tokens (a 2-layer transformer
  decoder, DETR-style): query k learns to find the face whose centre is
  colour k wherever it is in the image, and to read its stickers (or, with
  ``cell_queries``, 54 more queries, face plus cell, read one sticker each);
* heads: presence and 8 stickers × 7 classes per face (shared weights),
  turning from the pooled tokens;
* a seventh query reads the cube's orientation in the camera's frame, as a
  continuous rotation (6D representation of Zhou et al., 2019: the first two
  columns, made orthonormal), so that the app can follow how the cuber turns
  the whole cube (x, y, z) and tilts it.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .geometry import CELLS, FACE_PERM, NORMALS, SYMMETRIES, nearest_symmetric, project
from .labels import NUM_CLASSES

#: Cells of a face other than the centre (whose colour is the face's)
OUTER_CELLS = (0, 1, 2, 3, 5, 6, 7, 8)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ModelConfig:
    backbone: str = "mobilenet_v3_large"  # or mobilenet_v3_small, efficientnet_b0..b2, mobilenetv4_conv_medium...
    pretrained: bool = True
    image_size: int = 192
    dim: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1
    #: Tokens the decoder reads: 16 (a 12×12 grid at 192 px), or 8 (24×24, stride-8 features added):
    #: at stride 16 a sticker is about one token, read poorly
    token_stride: int = 16
    #: One query per sticker (a face's query plus a cell's), instead of one per face that must pack its
    #: eight stickers into one vector: each can look where its sticker is
    cell_queries: bool = False
    #: Each token also gets the pixels of its square, projected (a ViT's patch embedding): the deep
    #: ImageNet features learnt to ignore exact hues (colour jitter), the stickers' colours are exact
    colour_tokens: bool = False
    #: Experiment: the true orientation given to the queries (does reading stickers wait on knowing it?)
    orientation_oracle: bool = False
    #: Read each sticker where it is: the orientation and the cube's placement in the crop (both
    #: predicted) project the 54 stickers (geometry.py), and each is read from the tokens there
    geometric: bool = False
    #: With geometric: the pose is learnt up to the cube's 24 symmetries (where its faces are, whatever
    #: their colours), the six faces read where that pose puts them, centres included, and the centres
    #: read tell which face is which (by_colour): the outputs are those of the other readers
    symmetric: bool = False


def _backbone(name: str, pretrained: bool) -> nn.Sequential:
    if name == "mobilenet_v3_large":
        weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        return torchvision.models.mobilenet_v3_large(weights=weights).features
    if name == "mobilenet_v3_small":
        weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        return torchvision.models.mobilenet_v3_small(weights=weights).features
    if name in ("efficientnet_b0", "efficientnet_b1", "efficientnet_b2"):
        weights = torchvision.models.get_model_weights(name).IMAGENET1K_V1 if pretrained else None
        return torchvision.models.get_model(name, weights=weights).features
    if name in _TIMM:
        import timm

        net = timm.create_model(f"{name}.{_TIMM[name]}" if pretrained else name, pretrained=pretrained)
        # The stem then each stage, as torchvision's features: sliced at strides 8, 16, 32 alike
        return nn.Sequential(nn.Sequential(net.conv_stem, net.bn1), *net.blocks)
    raise ValueError(f"unknown backbone {name}")


#: Backbones from timm, by their pretrained weights (ImageNet means and deviations, as torchvision's)
_TIMM = {"mobilenetv4_conv_medium": "e250_r384_in12k_ft_in1k", "mobilenetv4_conv_large": "e600_r384_in1k",
         "mobilenetv4_hybrid_medium": "ix_e550_r384_in1k"}


def _split_at_stride(features: nn.Sequential, size: int, stride: int = 16) -> tuple[int, int, int]:
    """Index after which the features reach ``stride`` (their last layer there), their channels, and the last channels."""
    x = torch.zeros(1, 3, size, size)
    split, channels = -1, 0
    with torch.no_grad():
        for i, layer in enumerate(features):
            x = layer(x)
            if x.shape[-1] == size // stride:
                split, channels = i + 1, x.shape[1]
    if split < 0:
        raise RuntimeError(f"backbone never reaches stride {stride}")
    return split, channels, x.shape[1]


def rotation_from_6d(x: torch.Tensor) -> torch.Tensor:
    """[..., 6] → rotation matrices [..., 3, 3] whose first two columns follow the two 3-vectors (Gram-Schmidt)."""
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


class CubeReader(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        features = _backbone(cfg.backbone, cfg.pretrained)
        split, c16, c32 = _split_at_stride(features, cfg.image_size)
        d = cfg.dim
        if cfg.token_stride == 8:
            split8, c8, _ = _split_at_stride(features, cfg.image_size, 8)
            self.stem8 = features[:split8]  # → stride 8
            self.stem = features[split8:split]  # → stride 16
            self.lateral8 = nn.Sequential(nn.Conv2d(c8, d, 1, bias=False), nn.BatchNorm2d(d))
        elif cfg.token_stride == 16:
            self.stem = features[:split]  # → stride 16
        else:
            raise ValueError(f"token_stride {cfg.token_stride}: 8 or 16")
        self.deep = features[split:]  # → stride 32
        self.lateral16 = nn.Sequential(nn.Conv2d(c16, d, 1, bias=False), nn.BatchNorm2d(d))
        self.lateral32 = nn.Sequential(nn.Conv2d(c32, d, 1, bias=False), nn.BatchNorm2d(d))
        self.fuse = nn.Sequential(nn.Conv2d(d, d, 3, padding=1, bias=False), nn.BatchNorm2d(d), nn.Hardswish())
        if cfg.colour_tokens:
            s = cfg.token_stride
            self.colour = nn.Sequential(nn.Conv2d(3, d, s, stride=s, bias=False), nn.BatchNorm2d(d))
        grid = cfg.image_size // cfg.token_stride
        self.position = nn.Parameter(torch.zeros(1, grid * grid, d))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.queries = nn.Parameter(torch.zeros(7, d))  # six faces, then the orientation
        nn.init.trunc_normal_(self.queries, std=0.02)
        if cfg.orientation_oracle:
            self.orientation_in = nn.Linear(9, d)
        if cfg.geometric:
            # Placement: the cube centre's position (shares of the crop) and its scale, from the orientation query
            self.placement_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
            # A sticker: the tokens where it is, with its face's query (which face, how it is lit)
            self.read_head = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, NUM_CLASSES))
            if cfg.symmetric:
                # A face of the pose is readable: from its stickers' tokens and how squarely it faces the camera
                self.pose_present = nn.Sequential(nn.Linear(d + 1, d), nn.GELU(), nn.Linear(d, 1))
        if cfg.cell_queries:
            self.cell_queries = nn.Parameter(torch.zeros(9, d))  # added to each face's query: 54 sticker queries
            nn.init.trunc_normal_(self.cell_queries, std=0.02)
            self.cell_head = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, NUM_CLASSES))
        layer = nn.TransformerDecoderLayer(d, cfg.heads, dim_feedforward=4 * d, dropout=cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg.layers)
        self.norm = nn.LayerNorm(d)
        self.present_head = nn.Linear(d, 1)
        self.sticker_head = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, len(OUTER_CELLS) * NUM_CLASSES))
        self.turning_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.orientation_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 6))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor, orientation: torch.Tensor | None = None,
                teach: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """``image``: [B, 3, S, S] RGB in 0..1. Returns logits."""
        x = (image - self.mean) / self.std
        f8 = self.stem8(x) if self.cfg.token_stride == 8 else None
        f16 = self.stem(f8 if f8 is not None else x)
        f32 = self.deep(f16)
        tokens = self.lateral16(f16) + F.interpolate(self.lateral32(f32), size=f16.shape[-2:], mode="nearest")
        if f8 is not None:
            # Finer tokens: the stride-8 features, plus the coarser ones brought up (a feature pyramid)
            tokens = self.lateral8(f8) + F.interpolate(tokens, size=f8.shape[-2:], mode="nearest")
        tokens = self.fuse(tokens)
        if self.cfg.colour_tokens:
            tokens = tokens + self.colour(x)
        grid_map = tokens  # [B, d, H, W]: where geometric reading samples
        tokens = tokens.flatten(2).transpose(1, 2) + self.position  # [B, T, d]
        b = image.shape[0]
        queries = self.queries
        if self.cfg.cell_queries:
            cells = (self.queries[:6, None] + self.cell_queries[None]).flatten(0, 1)  # [54, d]
            queries = torch.cat([queries, cells])
        queries = queries.unsqueeze(0).expand(b, -1, -1)
        if self.cfg.orientation_oracle and orientation is not None:
            queries = queries + self.orientation_in(orientation.flatten(1).to(queries.dtype))[:, None]
        decoded = self.norm(self.decoder(queries, tokens))  # [B, 7 (+ 54), d]
        faces = decoded[:, :6]
        placement = None
        if self.cfg.geometric:
            raw = self.placement_head(decoded[:, 6])
            # Centre near the crop's middle, scale near a cube spanning 1/1.5 of it
            placement = torch.cat([torch.sigmoid(raw[:, :2]), 0.27 * torch.exp(raw[:, 2:3])], 1)
            orientation_used = rotation_from_6d(self.orientation_head(decoded[:, 6]).float()).detach()
            placement_used = placement.detach().float()
            if teach is not None:
                # Training: the true geometry for the images chosen (and known), else the reader's own
                true_orientation, true_placement, use = teach
                if self.cfg.symmetric:
                    # The true pose that matches the predicted one (the loss pairs the faces the same way)
                    true_orientation, _ = nearest_symmetric(true_orientation.float(), orientation_used)
                use = use.view(-1, 1)
                orientation_used = torch.where(use.view(-1, 1, 1), true_orientation.float(), orientation_used)
                placement_used = torch.where(use, true_placement.float(), placement_used)
            where = project(orientation_used, placement_used)  # [B, 54, 2], shares of the crop
            sampled = F.grid_sample(grid_map.float(), (where * 2 - 1).unsqueeze(2), mode="bilinear", padding_mode="zeros",
                                    align_corners=False).squeeze(-1).transpose(1, 2)  # [B, 54, d]
            if self.cfg.symmetric:
                # The faces of the pose, not of a colour: each one's context is its own stickers
                face_tokens = sampled.view(b, 6, 9, -1).mean(2)  # [B, 6, d]
                facing = (torch.as_tensor(NORMALS, device=image.device) @ orientation_used.transpose(1, 2))[..., 2:3]  # [B, 6, 1]
                pose_present = self.pose_present(torch.cat([face_tokens, facing], -1).to(decoded.dtype)).squeeze(-1)
                context = face_tokens.repeat_interleave(9, dim=1)
            else:
                context = faces.float().repeat_interleave(9, dim=1)  # [B, 54, d]
            read = self.read_head(torch.cat([sampled, context], -1).to(decoded.dtype))  # [B, 54, 7]
            stickers = read.view(b, 6, 9, NUM_CLASSES)
            if not self.cfg.symmetric:
                stickers = stickers[:, :, list(OUTER_CELLS)]
        elif self.cfg.cell_queries:
            stickers = self.cell_head(decoded[:, 7:].reshape(b, 6, 9, -1))[:, :, list(OUTER_CELLS)]
        else:
            stickers = self.sticker_head(faces).view(b, 6, len(OUTER_CELLS), NUM_CLASSES)
        return {
            # [B, 6] and [B, 6, 8, 7] by centre colour; symmetric: [B, 6] and [B, 6, 9, 7] by face of the pose (by_colour)
            "present": pose_present if self.cfg.geometric and self.cfg.symmetric else self.present_head(faces).squeeze(-1),
            "stickers": stickers,
            "turning": self.turning_head(tokens.mean(1)).squeeze(-1),  # [B]
            "orientation": self.orientation_head(decoded[:, 6]),  # [B, 6], see rotation_from_6d
            **({"placement": placement} if placement is not None else {}),  # [B, 3], geometry.py
        }


def by_colour(out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """A symmetric reader's outputs (faces of the pose, all 9 stickers) as the other readers' (faces by
    centre colour, 8 outer stickers, the orientation of the colours): the symmetry whose faces best
    match the centres read on the faces seen. Each face's stickers come in StickerCube order, as that
    symmetry turns them (right when two faces or more are seen; one alone leaves its turn a guess).
    In ONNX too (gathers and an argmax)."""
    pose = rotation_from_6d(out["orientation"].float())
    b = pose.shape[0]
    perm = torch.as_tensor(FACE_PERM, device=pose.device)  # [24, 6]: the colour of each face of the pose
    centre = torch.log_softmax(out["stickers"][:, :, 4, :6].float(), -1)  # [B, 6 faces, 6 colours]
    fits = centre[:, None].expand(b, 24, 6, 6).gather(3, perm[None, :, :, None].expand(b, 24, 6, 1)).squeeze(-1)  # [B, 24, 6]
    g = (fits * torch.sigmoid(out["present"].float())[:, None]).sum(-1).argmax(1)  # [B]
    face_of = torch.argsort(perm[g], dim=1)  # [B, 6]: the face of the pose that has colour k
    cells = torch.as_tensor(CELLS, device=pose.device)[g]  # [B, 54]: the pose's sticker at each of the cube's
    stickers = out["stickers"].flatten(1, 2).gather(1, cells[..., None].expand(b, 54, NUM_CLASSES))
    stickers = stickers.view(b, 6, 9, NUM_CLASSES)[:, :, list(OUTER_CELLS)]
    orientation = pose @ torch.as_tensor(SYMMETRIES, device=pose.device)[g].transpose(1, 2)  # the colours' orientation
    return {**out, "present": out["present"].gather(1, face_of), "stickers": stickers,
            "orientation": torch.cat([orientation[..., 0], orientation[..., 1]], -1)}


class ExportWrapper(nn.Module):
    """What the browser runs: probabilities, the centre cell filled in (see ml2/spec.json)."""

    def __init__(self, model: CubeReader, temperature: float = 1.0):
        super().__init__()
        self.model = model
        self.temperature = temperature
        centre = torch.zeros(1, 6, 1, NUM_CLASSES)
        centre[0, torch.arange(6), 0, torch.arange(6)] = 1.0
        self.register_buffer("centre", centre, persistent=False)

    def forward(self, image: torch.Tensor):
        out = self.model(image)
        if self.model.cfg.geometric and self.model.cfg.symmetric:
            out = by_colour(out)
        outer = torch.softmax(out["stickers"] / self.temperature, dim=-1)  # [N, 6, 8, 7]
        centre = self.centre.expand(image.shape[0], -1, -1, -1)
        stickers = torch.cat([outer[:, :, :4], centre, outer[:, :, 4:]], dim=2)  # [N, 6, 9, 7]
        orientation = rotation_from_6d(out["orientation"])  # [N, 3, 3]
        return torch.sigmoid(out["present"]), stickers, torch.sigmoid(out["turning"]).unsqueeze(-1), orientation


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
