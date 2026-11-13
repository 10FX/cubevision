"""Datasets read from a generated folder (need PyTorch: run on the training machine)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from cubevision.data import LabelTable, SyntheticDataset  # noqa: E402
from cubevision.detector import DetectionDataset  # noqa: E402
from cubevision.labels import HIDDEN, FaceLabel, ImageLabel  # noqa: E402


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "images").mkdir()
    labels = []
    for i in range(3):
        Image.new("RGB", (64, 48), (i * 80, 10, 10)).save(tmp_path / "images" / f"{i}.jpg")
        faces = [FaceLabel(k == i, [k] * 8 + [HIDDEN]) for k in range(6)]
        labels.append(ImageLabel(f"images/{i}.jpg", faces, i == 1, {"index": i, "style": "stickerless"},
                                 box=[0.25, 0.25, 0.75, 0.75] if i else None,
                                 orientation=[0.0, -1, 0, 1, 0, 0, 0, 0, 1] if i == 2 else None))
    (tmp_path / "labels-0000000.jsonl").write_text("\n".join(label.to_json() for label in labels) + "\n")
    return tmp_path


def test_the_label_table_holds_no_python_objects(folder):
    table = LabelTable([folder])
    assert len(table) == 3
    # Workers share arrays of plain numbers and bytes; Python objects they would copy
    for array in (table.paths, table.present, table.stickers, table.turning, table.orientation, table.boxes, table.metas):
        assert array.dtype != object
    assert table.path(1).endswith("images/1.jpg") and table.meta(2) == {"index": 2, "style": "stickerless"}
    assert table.box(0) is None and table.box(1) == [0.25, 0.25, 0.75, 0.75]


def test_the_reader_dataset_gives_the_labels(folder):
    ds = SyntheticDataset([folder], 32, train=False)
    image, present, stickers, turning, orientation, has_orientation, placement = ds[2]
    assert placement.shape == (3,) and not torch.isnan(placement).any()
    assert image.shape == (3, 32, 32) and image.dtype == torch.uint8
    assert present.tolist() == [0, 0, 1, 0, 0, 0]
    assert stickers.dtype == torch.long and stickers[3].tolist() == [3] * 8 + [HIDDEN]
    assert turning.item() == 0 and has_orientation.item() == 1
    assert torch.equal(orientation, torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    assert ds[0][5].item() == 0 and torch.equal(ds[0][4], torch.eye(3)) and ds[1][3].item() == 1
    assert torch.isnan(ds[0][6]).all()  # no box: no placement
    assert ds.meta(1)["index"] == 1


def test_the_detector_dataset_letterboxes_the_box(folder):
    ds = DetectionDataset([folder], 64, train=False)
    square, target = ds[1]
    assert square.shape == (3, 64, 64)
    # 64×48 fitted into 64×64: 8 px bars above and below
    assert np.allclose(target.numpy(), [[16, 8 + 12, 48, 8 + 36]])
    assert ds[0][1].shape == (0, 4)


def test_the_codec_marks_are_a_jpeg_s():
    import torch

    from cubevision.augment import _codec

    x = torch.rand(2, 3, 32, 32)
    off = torch.tensor([False, False])
    assert torch.allclose(_codec(x, off, off, torch.tensor([50.0, 50.0])), x, atol=1e-3)
    on = torch.tensor([True, True])
    err = (_codec(x, off, on, torch.tensor([90.0, 15.0])) - x).abs().mean((1, 2, 3))
    assert 0 < err[0] < err[1]
    # Colours at half resolution: a 2×2 block shares its chroma, its grey levels stay
    grey = torch.rand(1, 1, 8, 8).repeat(1, 3, 1, 1)
    assert torch.allclose(_codec(grey, torch.tensor([True]), torch.tensor([False]), torch.tensor([50.0])), grey, atol=1e-3)
