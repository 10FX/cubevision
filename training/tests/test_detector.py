"""The cube detector (needs PyTorch: run on the training machine)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from cubevision.detector import (  # noqa: E402
    CROP_MARGIN,
    DetectionMetrics,
    DetectorConfig,
    ExportWrapper,
    build_detector,
    crop_of,
    inside_share,
    letterbox,
)


@pytest.fixture(scope="module")
def model():
    return build_detector(DetectorConfig(pretrained=False))


def test_training_takes_frames_with_and_without_a_cube(model):
    model.train()
    images = [torch.rand(3, 320, 320), torch.rand(3, 320, 320)]
    targets = [{"boxes": torch.tensor([[100.0, 80.0, 180.0, 170.0]]), "labels": torch.ones(1, dtype=torch.int64)},
               {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64)}]
    losses = model(images, targets)
    assert set(losses) == {"bbox_regression", "classification"} and all(torch.isfinite(v) for v in losses.values())


def test_export_wrapper_decodes_like_torchvision(model):
    model.eval()
    wrapper = ExportWrapper(model, 320).eval()
    rel = torch.randn(1, wrapper.anchors.shape[0], 4) * 0.5
    want = model.box_coder.decode_single(rel[0], wrapper.anchors)
    assert torch.allclose(wrapper.decode(rel)[0], want, atol=1e-4)
    box, score = wrapper(torch.rand(2, 3, 320, 320))
    assert box.shape == (2, 4) and score.shape == (2, 1) and ((score >= 0) & (score <= 1)).all()


def test_letterbox_keeps_the_frame_centred():
    image = torch.randint(0, 255, (3, 180, 320), dtype=torch.uint8)
    square, scale, ox, oy = letterbox(image, 320)
    assert square.shape == (3, 320, 320) and scale == 1.0 and (ox, oy) == (0, 70)
    assert (square[:, :70] == 128).all() and (square[:, 250:] == 128).all()


def test_crop_and_metrics():
    box = torch.tensor([[0.4, 0.4, 0.6, 0.5]])
    crop = crop_of(box)
    assert torch.allclose(crop, torch.tensor([[0.5 - 0.1 * CROP_MARGIN, 0.45 - 0.1 * CROP_MARGIN, 0.5 + 0.1 * CROP_MARGIN, 0.45 + 0.1 * CROP_MARGIN]]))
    m = DetectionMetrics()
    m.update(box, torch.tensor([[0.9]]), [box * 320], 320)
    m.update(box, torch.tensor([[0.1]]), [torch.zeros((0, 4))], 320)
    s = m.summary()
    assert s["precision"] == 1 and s["recall"] == 1 and s["iou"] == pytest.approx(1) and s["crop_ok"] == 1
    assert inside_share([0.9, 0.0, 1.1, 0.5]) == pytest.approx(0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_export_wrapper_of_a_model_on_the_gpu():
    # Validation during training wraps the model where it trains
    model = build_detector(DetectorConfig(pretrained=False)).cuda().eval()
    box, score = ExportWrapper(model, 320).cuda()(torch.rand(1, 3, 320, 320, device="cuda"))
    assert box.shape == (1, 4) and score.shape == (1, 1)


def test_smaller_anchors_keep_the_head_and_reach_small_cubes():
    default = ExportWrapper(build_detector(DetectorConfig(pretrained=False)).eval(), 320)
    small = build_detector(DetectorConfig(pretrained=False, min_scale=0.05)).eval()
    wrapper = ExportWrapper(small, 320)
    assert wrapper.anchors.shape == default.anchors.shape  # same anchors per location: the COCO head still fits
    side = lambda a: ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])).sqrt().min().item()  # noqa: E731  (scale: the square root of the area)
    assert side(default.anchors) == pytest.approx(64, abs=1) and side(wrapper.anchors) == pytest.approx(16, abs=1)
    box, score = wrapper(torch.rand(1, 3, 320, 320))
    assert box.shape == (1, 4)


def test_the_threshold_is_the_lowest_that_keeps_the_precision():
    from cubevision.train_detector import choose_threshold

    # Ten frames with a cube found at score 0.9, two without a cube where something scores 0.6 and 0.4
    box = torch.tensor([[0.4, 0.4, 0.6, 0.6]])
    boxes = box.repeat(12, 1)
    scores = torch.tensor([[0.9]] * 10 + [[0.6], [0.4]])
    targets = [box * 320] * 10 + [torch.zeros((0, 4))] * 2
    assert choose_threshold(boxes, scores, targets, 320, min_precision=0.98) == pytest.approx(0.61)
    assert choose_threshold(boxes, scores, targets, 320, min_precision=0.9) == pytest.approx(0.41)


def test_far_frames_shrink_the_cube(tmp_path):
    import random

    from PIL import Image

    from cubevision.detector import DetectionDataset
    from cubevision.labels import HIDDEN, FaceLabel, ImageLabel

    (tmp_path / "images").mkdir()
    Image.new("RGB", (160, 90), (90, 90, 90)).save(tmp_path / "images" / "a.jpg")
    label = ImageLabel("images/a.jpg", [FaceLabel(False, [HIDDEN] * 9)] * 6, False, {"index": 0}, [0.4, 0.3, 0.6, 0.7], None)
    (tmp_path / "labels-0000000.jsonl").write_text(label.to_json() + "\n")
    plain = DetectionDataset([tmp_path], 64, train=False)[0][1][0]
    far = DetectionDataset([tmp_path], 64, train=True, far=1.0)
    random.seed(0)
    for _ in range(20):
        box = far[0][1]
        if len(box):
            ratio = float((box[0, 2] - box[0, 0]) / (plain[2] - plain[0]))
            assert 0.24 < ratio < 0.61
