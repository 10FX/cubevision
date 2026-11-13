"""
Batched augmentations on the GPU, for images in 0..1 ([B, 3, H, W] float).

What a phone or webcam does to a cube that the renderer does not: shaky
framing, white balance and exposure, defocus, motion blur, sensor noise,
low resolution, vignetting, and the video codec's own marks (colours at half
resolution, 8×8 blocks quantized as JPEG and VP8/VP9 do). Geometric changes keep the labels valid: readings are
compared up to a rotation of each face, so the image may be rotated freely;
it is never mirrored (a mirrored cube has an impossible colour scheme).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AugmentConfig:
    rotate_deg: float = 180.0  # any roll: the face readings are rotation-invariant
    scale: tuple[float, float] = (0.85, 1.15)
    translate: float = 0.08
    brightness: float = 0.35
    contrast: float = 0.35
    saturation: float = 0.35
    hue: float = 0.025  # small: red and orange are close
    white_balance: float = 0.12
    blur_p: float = 0.3
    blur_sigma: tuple[float, float] = (0.3, 1.8)
    motion_p: float = 0.35
    motion_length: tuple[int, int] = (3, 15)
    #: A camera out of focus: a disc (the lens's aperture), not a Gaussian: edges smeared flat, highlights as rings
    defocus_p: float = 0.0
    defocus_radius: tuple[float, float] = (1.0, 4.5)
    noise_p: float = 0.5
    noise_sigma: tuple[float, float] = (0.005, 0.04)
    lowres_p: float = 0.25
    lowres_scale: tuple[float, float] = (0.35, 0.8)
    vignette_p: float = 0.25
    chroma_p: float = 0.5  # 4:2:0, as every webcam and browser recording
    jpeg_p: float = 0.4
    jpeg_quality: tuple[float, float] = (15.0, 90.0)
    #: Mains flicker under a rolling shutter: bright and dark horizontal bands
    flicker_p: float = 0.0
    #: Blown highlights: a gain then clipping (white and yellow alike, colours washed out)
    overexpose_p: float = 0.0
    #: A video uploaded then squeezed again: a second pass of the codec, at its own quality
    recompress_p: float = 0.0


def _uniform(b: int, lo: float, hi: float, device) -> torch.Tensor:
    return torch.rand(b, device=device) * (hi - lo) + lo


def _affine(x: torch.Tensor, cfg: AugmentConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rotate (any roll), zoom and shift each image. Returns the images, the angle each image's
    content turned by, counter-clockwise on screen (radians), and the sampling matrices [B, 2, 3]
    (output → input, normalized coordinates) for move_placement."""
    b = x.shape[0]
    angle = _uniform(b, -cfg.rotate_deg, cfg.rotate_deg, x.device) * math.pi / 180
    scale = _uniform(b, *cfg.scale, x.device)
    tx, ty = (_uniform(b, -cfg.translate, cfg.translate, x.device) * 2 for _ in range(2))
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    # Border pixels repeated: no black corners the real crops never have
    out = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    # The grid samples the input at R(angle)·p (y down): the content turns by -angle in image
    # coordinates, which is +angle counter-clockwise as seen on screen
    return out, angle, theta


def _color(x: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    b = x.shape[0]
    shape = (b, 1, 1, 1)
    gain = 1 + _uniform(b * 3, -cfg.white_balance, cfg.white_balance, x.device).view(b, 3, 1, 1)
    x = x * gain
    x = x * (1 + _uniform(b, -cfg.brightness, cfg.brightness, x.device)).view(shape)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * (1 + _uniform(b, -cfg.contrast, cfg.contrast, x.device)).view(shape) + mean
    grey = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (x - grey) * (1 + _uniform(b, -cfg.saturation, cfg.saturation, x.device)).view(shape) + grey
    return _hue(x.clamp(0, 1), _uniform(b, -cfg.hue, cfg.hue, x.device))


def _hue(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Rotate the hue (fraction of a turn) in YIQ space: cheap and smooth."""
    angle = (shift * 2 * math.pi).view(-1, 1, 1)
    cos, sin = torch.cos(angle), torch.sin(angle)
    to_yiq = x.new_tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])
    from_yiq = torch.linalg.inv(to_yiq)
    b = x.shape[0]
    rot = torch.zeros(b, 3, 3, device=x.device, dtype=x.dtype)
    rot[:, 0, 0] = 1
    rot[:, 1, 1], rot[:, 1, 2] = cos[:, 0, 0], -sin[:, 0, 0]
    rot[:, 2, 1], rot[:, 2, 2] = sin[:, 0, 0], cos[:, 0, 0]
    m = from_yiq @ rot @ to_yiq  # [B, 3, 3]
    return torch.einsum("bij,bjhw->bihw", m, x).clamp(0, 1)


def _gaussian_kernels(sigma: torch.Tensor, size: int) -> torch.Tensor:
    r = torch.arange(size, device=sigma.device, dtype=sigma.dtype) - size // 2
    k = torch.exp(-(r[None] ** 2) / (2 * sigma[:, None] ** 2))
    k = k / k.sum(1, keepdim=True)
    return k[:, :, None] * k[:, None, :]  # [B, size, size]


def _disc_kernels(radius: torch.Tensor, size: int) -> torch.Tensor:
    """A disc of ``radius`` pixels, anti-aliased, normalized."""
    r = torch.arange(size, device=radius.device, dtype=torch.float32) - size // 2
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    k = torch.clamp(radius[:, None, None] + 0.5 - torch.sqrt(xx**2 + yy**2)[None], 0, 1)
    return k / k.sum(dim=(1, 2), keepdim=True).clamp(min=1e-6)


def _motion_kernels(length: torch.Tensor, angle: torch.Tensor, size: int) -> torch.Tensor:
    """A line of ``length`` pixels at ``angle``, anti-aliased, normalized."""
    r = torch.arange(size, device=length.device, dtype=torch.float32) - size // 2
    yy, xx = torch.meshgrid(r, r, indexing="ij")
    c, s = torch.cos(angle)[:, None, None], torch.sin(angle)[:, None, None]
    along = xx[None] * c + yy[None] * s
    across = -xx[None] * s + yy[None] * c
    k = (along.abs() <= length[:, None, None] / 2).float() * torch.clamp(1 - across.abs(), min=0)
    return k / k.sum(dim=(1, 2), keepdim=True).clamp(min=1e-6)


def _convolve(x: torch.Tensor, kernels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-image kernel (grouped convolution); images where mask is False are left alone."""
    if not mask.any():
        return x
    b, c, h, w = x.shape
    size = kernels.shape[-1]
    weight = kernels[:, None].repeat_interleave(c, dim=0).to(x.dtype)  # [B*C, 1, k, k]
    y = F.conv2d(F.pad(x.reshape(1, b * c, h, w), [size // 2] * 4, mode="replicate"), weight, groups=b * c).view(b, c, h, w)
    return torch.where(mask.view(b, 1, 1, 1), y, x)


def move_placement(placement: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Cube placements [B, 3] (centre in shares of the image, scale) after the affine whose sampling
    matrices are ``theta``: the centre moves to where the content went, the scale with the zoom."""
    a, t = theta[:, :, :2], theta[:, :, 2]
    centre = torch.linalg.solve(a, (placement[:, :2] * 2 - 1 - t).unsqueeze(-1)).squeeze(-1)
    scale = placement[:, 2] / torch.linalg.det(a).abs().sqrt()
    return torch.cat([(centre + 1) / 2, scale[:, None]], 1)


def roll_orientation(orientation: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Orientation labels [B, 3, 3] (camera frame: x right, y up, z to the viewer) of images whose
    content turned counter-clockwise on screen by ``angle``: a rotation about the camera's z."""
    c, s = torch.cos(angle), torch.sin(angle)
    z, o = torch.zeros_like(angle), torch.ones_like(angle)
    roll = torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)
    return roll.to(orientation.dtype) @ orientation


@torch.no_grad()
def augment(x: torch.Tensor, cfg: AugmentConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The reader's augmentation: geometry then photometry. Returns the images, their roll (radians,
    counter-clockwise on screen) for roll_orientation, and the affine for move_placement."""
    x, angle, theta = _affine(x, cfg)
    return photometric(x, cfg), angle, theta


@torch.no_grad()
def photometric(x: torch.Tensor, cfg: AugmentConfig) -> torch.Tensor:
    """Colour, white balance, blur, motion blur, low resolution and noise (no geometry: boxes stay put)."""
    b, dev = x.shape[0], x.device
    x = _color(x, cfg)
    if cfg.overexpose_p > 0:
        blown = (torch.rand(b, device=dev) < cfg.overexpose_p).view(b, 1, 1, 1)
        x = torch.where(blown, (x * _uniform(b, 1.4, 2.6, dev).view(b, 1, 1, 1)).clamp(0, 1), x)
    if cfg.flicker_p > 0:
        x = _flicker(x, torch.rand(b, device=dev) < cfg.flicker_p)
    size = 2 * (cfg.motion_length[1] // 2) + 1
    blur = torch.rand(b, device=dev) < cfg.blur_p
    # Three sigmas each side: a fixed 9 pixels cut the blurs past 1.5 (an earlier 3.0 blurred as a 2.2, nearly a box)
    x = _convolve(x, _gaussian_kernels(_uniform(b, *cfg.blur_sigma, dev), max(9, 2 * math.ceil(3 * cfg.blur_sigma[1]) + 1)), blur)
    if cfg.defocus_p > 0:
        out_of_focus = torch.rand(b, device=dev) < cfg.defocus_p
        x = _convolve(x, _disc_kernels(_uniform(b, *cfg.defocus_radius, dev), 2 * math.ceil(cfg.defocus_radius[1]) + 3), out_of_focus)
    motion = torch.rand(b, device=dev) < cfg.motion_p
    x = _convolve(x, _motion_kernels(_uniform(b, *cfg.motion_length, dev), _uniform(b, 0, math.pi, dev), size), motion)
    if cfg.lowres_p > 0:
        low = torch.rand(b, device=dev) < cfg.lowres_p
        if low.any():
            f = float(_uniform(1, *cfg.lowres_scale, dev))
            small = F.interpolate(x, scale_factor=f, mode="bilinear", antialias=True, align_corners=False)
            back = F.interpolate(small, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.where(low.view(b, 1, 1, 1), back, x)
    if cfg.vignette_p > 0:
        x = _vignette(x, torch.rand(b, device=dev) < cfg.vignette_p)
    noisy = (torch.rand(b, device=dev) < cfg.noise_p).view(b, 1, 1, 1)
    x = x + noisy * torch.randn_like(x) * _uniform(b, *cfg.noise_sigma, dev).view(b, 1, 1, 1)
    x = x.clamp(0, 1)
    if cfg.chroma_p > 0 or cfg.jpeg_p > 0:
        x = _codec(x, torch.rand(b, device=dev) < cfg.chroma_p, torch.rand(b, device=dev) < cfg.jpeg_p, _uniform(b, *cfg.jpeg_quality, dev))
    if cfg.recompress_p > 0:
        again = torch.rand(b, device=dev) < cfg.recompress_p
        if again.any():
            x = _codec(x, again, again, _uniform(b, *cfg.jpeg_quality, dev))
    return x.clamp(0, 1)


def _flicker(x: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
    """Horizontal bands of ±5 to 25 % brightness, 2 to 8 across the image, anywhere."""
    b, _, h, _w = x.shape
    yy = torch.linspace(0, 1, h, device=x.device).view(1, h)
    bands = _uniform(b, 2, 8, x.device).view(b, 1)
    phase = _uniform(b, 0, 2 * math.pi, x.device).view(b, 1)
    depth = (_uniform(b, 0.05, 0.25, x.device) * on).view(b, 1)
    return x * (1 + depth * torch.sin(2 * math.pi * bands * yy + phase)).view(b, 1, h, 1)


def _vignette(x: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
    """Corners darker, by up to half, as cheap lenses."""
    b, _, h, w = x.shape
    yy = torch.linspace(-1, 1, h, device=x.device).view(1, h, 1)
    xx = torch.linspace(-1, 1, w, device=x.device).view(1, 1, w)
    strength = _uniform(b, 0.15, 0.5, x.device).view(b, 1, 1) * on.view(b, 1, 1)
    return x * (1 - strength * ((xx ** 2 + yy ** 2) / 2)).unsqueeze(1)


#: JPEG's quantization tables (luma, chroma) at quality 50
_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
         18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
_CHROMA = [17, 18, 24, 47] + [99] * 4 + [18, 21, 26, 66] + [99] * 4 + [24, 26, 56] + [99] * 5 + [47, 66] + [99] * 6 + [99] * 32


def _dct_matrix(device) -> torch.Tensor:
    k = torch.arange(8, device=device, dtype=torch.float32)
    d = torch.cos((2 * k.view(1, 8) + 1) * k.view(8, 1) * math.pi / 16) * math.sqrt(2 / 8)
    d[0] /= math.sqrt(2)
    return d


def _codec(x: torch.Tensor, chroma: torch.Tensor, jpeg: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
    """YCbCr; colours at half resolution where ``chroma``; 8×8 DCT blocks quantized at ``quality`` (1-100) where ``jpeg``."""
    b, _, h, w = x.shape
    r, g, bl = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * bl
    cb = 0.5 + (bl - y) * 0.564
    cr = 0.5 + (r - y) * 0.713
    ycc = torch.stack([y, cb, cr], 1)
    half = F.interpolate(F.avg_pool2d(ycc[:, 1:], 2), size=(h, w), mode="bilinear", align_corners=False)
    ycc = torch.cat([ycc[:, :1], torch.where(chroma.view(b, 1, 1, 1), half, ycc[:, 1:])], 1)
    if jpeg.any() and h % 8 == 0 and w % 8 == 0:
        d = _dct_matrix(x.device)
        # Quality → scale of the tables (libjpeg's rule)
        scale = torch.where(quality < 50, 50 / quality, 2 - quality / 50).view(b, 1, 1, 1, 1, 1)
        tables = torch.tensor([_LUMA, _CHROMA, _CHROMA], device=x.device, dtype=torch.float32).view(1, 3, 1, 1, 8, 8)
        step = (tables * scale).clamp(min=1) / 255
        blocks = ycc.view(b, 3, h // 8, 8, w // 8, 8).permute(0, 1, 2, 4, 3, 5)  # [B, 3, H/8, W/8, 8, 8]
        coef = d @ (blocks - 0.5) @ d.T
        coef = torch.round(coef / step) * step
        back = (d.T @ coef @ d + 0.5).permute(0, 1, 2, 4, 3, 5).reshape(b, 3, h, w)
        ycc = torch.where(jpeg.view(b, 1, 1, 1), back, ycc)
    y, cb, cr = ycc[:, 0], ycc[:, 1] - 0.5, ycc[:, 2] - 0.5
    return torch.stack([y + 1.403 * cr, y - 0.344 * cb - 0.714 * cr, y + 1.773 * cb], 1)
