"""Where the stickers are in a crop (needs PyTorch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cubevision.geometry import CORNERS, placement_from_box, project  # noqa: E402


def test_a_cube_seen_face_on_fills_its_box():
    # Facing the camera, the box spans the cube's front face: the F centre sits in the box's middle
    box = [0.2, 0.3, 0.6, 0.7]
    p = placement_from_box(np.eye(3, dtype=np.float32), box)
    assert p == pytest.approx([0.4, 0.5, 0.2])
    xy = project(torch.eye(3)[None], torch.tensor(p)[None])[0]
    assert xy[2 * 9 + 4].tolist() == pytest.approx([0.4, 0.5])  # F centre
    assert xy[2 * 9 + 0].tolist() == pytest.approx([0.4 - 0.2 * 2 / 3, 0.5 - 0.2 * 2 / 3])  # F top-left: up is y down


def test_the_placement_comes_back_from_the_box_it_makes():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        q = (q * np.sign(np.linalg.det(q))).astype(np.float32)
        placement = np.array([0.45, 0.55, 0.18], dtype=np.float32)
        # The box a cube so placed spans, as the renderer measures it
        cam = CORNERS @ q.T
        xy = placement[:2] + placement[2] * np.stack([cam[:, 0], -cam[:, 1]], 1)
        box = [*xy.min(0), *xy.max(0)]
        assert placement_from_box(q, box) == pytest.approx(placement, abs=1e-5)


def test_the_24_symmetries_of_the_cube_permute_its_faces():
    from cubevision.geometry import FACE_PERM, NORMALS, SYMMETRIES

    assert SYMMETRIES.shape == (24, 3, 3) and len({m.tobytes() for m in SYMMETRIES}) == 24
    for m, perm in zip(SYMMETRIES, FACE_PERM, strict=True):
        assert np.allclose(m @ m.T, np.eye(3)) and np.linalg.det(m) == pytest.approx(1)
        assert sorted(perm) == list(range(6)) and np.allclose(NORMALS @ m.T, NORMALS[perm])
        # Opposite faces stay opposite
        assert all(perm[(p + 3) % 6] == (perm[p] + 3) % 6 for p in range(6))
