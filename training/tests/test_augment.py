"""The reader's augmentation."""
import torch

from cubevision.augment import AugmentConfig, photometric


def _spread(image: torch.Tensor) -> float:
    """Standard deviation (pixels) of a blurred point along a row."""
    row = image[0, 0, 32]
    r = torch.arange(row.numel(), dtype=row.dtype) - 32
    w = row / row.sum()
    return float((w * r**2).sum().sqrt())


def test_blur_as_wide_as_asked():
    point = torch.zeros(1, 3, 65, 65)
    point[..., 32, 32] = 1
    none = dict(brightness=0, contrast=0, saturation=0, hue=0, white_balance=0, motion_p=0, noise_p=0, lowres_p=0,
                vignette_p=0, chroma_p=0, jpeg_p=0)
    for sigma in (1.0, 3.0, 4.5):
        cfg = AugmentConfig(**none, blur_p=1.0, blur_sigma=(sigma, sigma))
        # A separable Gaussian: the row through the point is a Gaussian of the same sigma
        assert abs(_spread(photometric(point, cfg)) - sigma) < 0.1 * sigma, sigma
