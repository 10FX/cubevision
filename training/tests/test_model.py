"""Model, loss, metrics and augmentations (need PyTorch: run on the training machine)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

import math  # noqa: E402

import numpy as np  # noqa: E402

from cubevision.augment import AugmentConfig, _affine, augment, roll_orientation  # noqa: E402
from cubevision.cube import ROT  # noqa: E402
from cubevision.labels import HIDDEN, NUM_CLASSES  # noqa: E402
from cubevision.losses import LossWeights, Metrics, reader_loss, rotation_angle, sticker_ce  # noqa: E402
from cubevision.model import OUTER_CELLS, CubeReader, ExportWrapper, ModelConfig, rotation_from_6d  # noqa: E402


@pytest.fixture(scope="module", params=["mobilenet_v3_large", "mobilenet_v3_small"])
def model(request):
    return CubeReader(ModelConfig(backbone=request.param, pretrained=False, image_size=160)).eval()


def test_output_shapes(model):
    out = model(torch.rand(2, 3, 160, 160))
    assert out["present"].shape == (2, 6)
    assert out["stickers"].shape == (2, 6, 8, NUM_CLASSES)
    assert out["turning"].shape == (2,)
    assert out["orientation"].shape == (2, 6)


def test_export_wrapper_fills_the_centre(model):
    present, stickers, turning, orientation = ExportWrapper(model)(torch.rand(1, 3, 160, 160))
    assert present.shape == (1, 6) and turning.shape == (1, 1)
    assert orientation.shape == (1, 3, 3)
    assert torch.allclose(orientation[0] @ orientation[0].T, torch.eye(3), atol=1e-5) and torch.det(orientation[0]) > 0.99
    assert stickers.shape == (1, 6, 9, NUM_CLASSES)
    assert torch.allclose(stickers.sum(-1), torch.ones(1, 6, 9))
    for k in range(6):
        assert stickers[0, k, 4].argmax() == k


def _logits_for(targets_9: torch.Tensor, rotation: int) -> torch.Tensor:
    """Confident logits that read the face in the given rotation."""
    b = targets_9.shape[0]
    logits = torch.full((b, 6, 8, NUM_CLASSES), -10.0)
    for j, cell in enumerate(OUTER_CELLS):
        idx = targets_9[:, :, ROT[rotation][cell]]
        logits[:, :, j].scatter_(-1, idx.unsqueeze(-1), 10.0)
    return logits


def test_sticker_loss_is_rotation_invariant():
    targets = torch.randint(0, NUM_CLASSES, (3, 6, 9))
    losses = [sticker_ce(_logits_for(targets, r), targets)[0] for r in range(4)]
    for loss in losses:
        assert torch.all(loss < 1e-3)
    wrong = sticker_ce(_logits_for(torch.randint(0, NUM_CLASSES, (3, 6, 9)), 0), targets)[0]
    assert wrong.mean() > 1.0


def test_metrics_of_a_perfect_reading():
    targets = torch.randint(0, 6, (4, 6, 9))
    targets[:, :, 3] = HIDDEN
    present = torch.ones(4, 6)
    out = {"present": torch.full((4, 6), 5.0), "stickers": _logits_for(targets, 2), "turning": torch.full((4,), -5.0)}
    m = Metrics()
    m.update(out, present, targets, torch.zeros(4))
    s = m.summary()
    assert s["sticker_acc"] == 1.0 and s["face_exact"] == 1.0 and s["hidden_acc"] == 1.0 and s["present_acc"] == 1.0
    assert s["two_faces_right"] == 1.0
    loss = reader_loss(out, present, targets, torch.zeros(4), LossWeights(label_smoothing=0.0))
    assert loss["loss"] < 0.05


def test_loss_ignores_faces_that_are_not_present():
    targets = torch.randint(0, 6, (2, 6, 9))
    present = torch.zeros(2, 6)
    out = {"present": torch.full((2, 6), -5.0), "stickers": torch.zeros(2, 6, 8, NUM_CLASSES), "turning": torch.zeros(2)}
    loss = reader_loss(out, present, targets, torch.zeros(2), LossWeights())
    assert loss["stickers"] == 0


def test_augment_keeps_shape_and_range():
    x = torch.rand(8, 3, 96, 96)
    y, roll, _ = augment(x, AugmentConfig(blur_p=1, motion_p=1, noise_p=1, lowres_p=1))
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1
    assert roll.shape == (8,)
    assert not torch.allclose(x, y)


def test_the_orientation_label_turns_with_the_image():
    # A bright spot right of the centre: the direction of +x in the camera's frame
    x = torch.zeros(16, 1, 65, 65)
    x[:, :, 32, 52] = 1.0
    y, roll, _ = _affine(x, AugmentConfig(rotate_deg=180, scale=(1.0, 1.0), translate=0.0))
    turned = roll_orientation(torch.eye(3).expand(16, 3, 3), roll)
    for k in range(16):
        row, col = divmod(int(y[k, 0].argmax()), 65)
        seen = torch.tensor([col - 32.0, -(row - 32.0)])  # image → camera axes (y up)
        expected = turned[k, :2, 0] * 20
        assert torch.dist(seen, expected) < 2.5, (k, seen, expected)


def test_orientation_loss_and_metric_of_the_truth():
    truth = rotation_from_6d(torch.randn(5, 6))
    six = torch.cat([truth[..., 0], truth[..., 1]], -1)  # first two columns
    assert torch.allclose(rotation_from_6d(six), truth, atol=1e-5)
    out = {"present": torch.full((5, 6), -5.0), "stickers": torch.zeros(5, 6, 8, NUM_CLASSES), "turning": torch.full((5,), -5.0), "orientation": six}
    stickers, has = torch.zeros(5, 6, 9, dtype=torch.long), torch.ones(5)
    loss = reader_loss(out, torch.zeros(5, 6), stickers, torch.zeros(5), LossWeights(), truth, has)
    assert loss["orientation"] < 1e-6
    m = Metrics()
    m.update(out, torch.zeros(5, 6), stickers, torch.zeros(5), truth, has)
    assert m.summary()["orientation_median_deg"] < 0.1
    quarter = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert math.degrees(rotation_angle(quarter, torch.eye(3))) == pytest.approx(90, abs=1e-3)


def test_onnx_export_matches(tmp_path, model):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    torch.backends.mha.set_fastpath_enabled(False)
    wrapper = ExportWrapper(model).eval()
    x = torch.rand(2, 3, 160, 160)
    path = tmp_path / "m.onnx"
    torch.onnx.export(wrapper, (x,), str(path), input_names=["image"], output_names=["present", "stickers", "turning", "orientation"],
                      opset_version=17, dynamic_axes={"image": {0: "batch"}}, dynamo=False)
    onnx.checker.check_model(onnx.load(str(path)))
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})
    with torch.no_grad():
        want = wrapper(x)
    for g, w in zip(got, want, strict=True):
        assert abs(g - w.numpy()).max() < 1e-4


def test_fp16_export_loads_and_matches(tmp_path, model):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxconverter_common")
    from cubevision.export import to_fp16

    torch.backends.mha.set_fastpath_enabled(False)
    wrapper = ExportWrapper(model).eval()
    x = torch.rand(2, 3, 160, 160)
    path = tmp_path / "m.onnx"
    torch.onnx.export(wrapper, (x,), str(path), input_names=["image"], output_names=["present", "stickers", "turning", "orientation"],
                      opset_version=17, dynamic_axes={"image": {0: "batch"}}, dynamo=False)
    half = tmp_path / "m.fp16.onnx"
    onnx.save(to_fp16(onnx.load(str(path))), str(half))
    # The attention's casts once made a graph onnxruntime refused to load
    got = ort.InferenceSession(str(half), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})
    with torch.no_grad():
        want = wrapper(x)
    for g, w in zip(got, want, strict=True):
        assert g.dtype == w.numpy().dtype and abs(g - w.numpy()).max() < 2e-2


def test_metadata_is_valid_json_for_the_browser():
    pytest.importorskip("onnxruntime")
    import json

    from cubevision.export import metadata_json

    text = metadata_json({"validation": {"present_precision": float("nan"), "face_exact": 0.5}, "sizes": [1.0, float("inf")]})
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"validation": {"present_precision": None, "face_exact": 0.5}, "sizes": [1.0, None]}


def test_finer_tokens_read_a_24_by_24_grid_and_export():
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = CubeReader(ModelConfig(pretrained=False, image_size=192, token_stride=8)).eval()
    assert model.position.shape == (1, 24 * 24, model.cfg.dim)
    out = model(torch.rand(2, 3, 192, 192))
    assert out["stickers"].shape == (2, 6, 8, NUM_CLASSES) and out["orientation"].shape == (2, 6)
    torch.backends.mha.set_fastpath_enabled(False)
    wrapper = ExportWrapper(model).eval()
    x = torch.rand(1, 3, 192, 192)
    import io

    buffer = io.BytesIO()
    torch.onnx.export(wrapper, (x,), buffer, input_names=["image"], output_names=["present", "stickers", "turning", "orientation"],
                      opset_version=17, dynamic_axes={"image": {0: "batch"}}, dynamo=False)
    got = ort.InferenceSession(buffer.getvalue(), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})
    with torch.no_grad():
        want = wrapper(x)
    assert all(abs(g - w.numpy()).max() < 1e-4 for g, w in zip(got, want, strict=True))
    onnx.checker.check_model(onnx.load_from_string(buffer.getvalue()))


def test_sticker_queries_read_the_same_outputs():
    model = CubeReader(ModelConfig(pretrained=False, image_size=160, cell_queries=True)).eval()
    out = model(torch.rand(2, 3, 160, 160))
    assert out["present"].shape == (2, 6) and out["stickers"].shape == (2, 6, 8, NUM_CLASSES) and out["orientation"].shape == (2, 6)
    present, stickers, turning, orientation = ExportWrapper(model)(torch.rand(1, 3, 160, 160))
    assert stickers.shape == (1, 6, 9, NUM_CLASSES) and all(stickers[0, k, 4].argmax() == k for k in range(6))


def test_colour_tokens_add_the_pixels_of_each_square():
    model = CubeReader(ModelConfig(pretrained=False, image_size=192, colour_tokens=True, token_stride=8)).eval()
    out = model(torch.rand(2, 3, 192, 192))
    assert out["stickers"].shape == (2, 6, 8, NUM_CLASSES)
    assert model.colour[0].weight.shape == (model.cfg.dim, 3, 8, 8)


def test_the_cube_placement_moves_with_the_image():
    from cubevision.augment import move_placement

    # A bright spot where the cube's centre is: after any affine, the placement points at it again
    x = torch.zeros(16, 1, 65, 65)
    x[:, :, 20, 40] = 1.0
    placement = torch.tensor([[40.5 / 65, 20.5 / 65, 0.2]]).repeat(16, 1)
    y, _, theta = _affine(x, AugmentConfig(rotate_deg=180, scale=(0.8, 1.2), translate=0.08))
    moved = move_placement(placement, theta)
    for k in range(16):
        row, col = divmod(int(y[k, 0].argmax()), 65)
        assert abs(moved[k, 0] * 65 - (col + 0.5)) < 1.5 and abs(moved[k, 1] * 65 - (row + 0.5)) < 1.5, (k, moved[k])
    # Zoomed in, the cube looks bigger: its scale grows with the zoom
    _, _, zoom = _affine(x, AugmentConfig(rotate_deg=0, scale=(1.2, 1.2), translate=0.0))
    assert move_placement(placement, zoom)[0, 2].item() == pytest.approx(0.24, abs=1e-4)


def test_the_geometric_reader_reads_at_the_stickers_and_learns_where_the_cube_is():
    model = CubeReader(ModelConfig(pretrained=False, image_size=192, token_stride=8, geometric=True)).train()
    image = torch.rand(2, 3, 192, 192)
    out = model(image)
    assert out["stickers"].shape == (2, 6, 8, NUM_CLASSES) and out["placement"].shape == (2, 3)
    teach = (torch.eye(3).repeat(2, 1, 1), torch.tensor([[0.5, 0.5, 0.25]] * 2), torch.tensor([True, False]))
    out = model(image, teach=teach)
    loss = reader_loss(out, torch.ones(2, 6), torch.randint(0, 7, (2, 6, 9)), torch.zeros(2), LossWeights(),
                       torch.eye(3).repeat(2, 1, 1), torch.ones(2), torch.tensor([[0.5, 0.5, 0.25], [float("nan")] * 3]))
    loss["loss"].backward()
    assert loss["placement"] > 0 and model.read_head[0].weight.grad.abs().sum() > 0
    present, stickers, turning, orientation = ExportWrapper(model.eval())(image[:1])
    assert stickers.shape == (1, 6, 9, NUM_CLASSES)


def test_a_symmetric_reader_tells_the_faces_by_their_centres():
    from cubevision.geometry import FACE_PERM, SYMMETRIES
    from cubevision.model import by_colour

    # A pose off by a symmetry g from the true orientation, and centres read as the faces it shows
    g = 7
    true = torch.eye(3)
    pose = true @ torch.as_tensor(SYMMETRIES[g])
    stickers = torch.full((1, 6, 9, NUM_CLASSES), -10.0)
    for p in range(6):
        stickers[0, p, :, FACE_PERM[g][p]] = 10.0  # every sticker of face p is its true face's colour
    out = {"present": torch.full((1, 6), 5.0), "stickers": stickers, "turning": torch.zeros(1),
           "orientation": torch.cat([pose[:, 0], pose[:, 1]])[None]}
    colour = by_colour(out)
    assert colour["stickers"].shape == (1, 6, 8, NUM_CLASSES)
    assert all(colour["stickers"][0, k].argmax(-1).eq(k).all() for k in range(6))
    assert torch.allclose(rotation_from_6d(colour["orientation"])[0], true, atol=1e-5)


def test_a_symmetric_reader_gives_each_face_in_sticker_cube_order_whatever_the_pose():
    from cubevision.cube import POSITIONS, Cube
    from cubevision.geometry import SYMMETRIES
    from cubevision.model import OUTER_CELLS, by_colour

    colours = Cube().apply("R U2 F' L D B2 R' U F2").colors()
    index = {p: i for i, p in enumerate(POSITIONS)}
    for g in range(24):
        pose = torch.as_tensor(SYMMETRIES[g])
        stickers = torch.full((1, 54, NUM_CLASSES), -10.0)
        for i, y in enumerate(POSITIONS):
            # The pose's sticker i is where the cube's sticker at SYMMETRIES[g] @ y is
            stickers[0, i, colours[index[tuple(int(v) for v in SYMMETRIES[g] @ np.array(y))]]] = 10.0
        out = {"present": torch.full((1, 6), 5.0), "stickers": stickers.view(1, 6, 9, NUM_CLASSES), "turning": torch.zeros(1),
               "orientation": torch.cat([pose[:, 0], pose[:, 1]])[None]}
        read = by_colour(out)["stickers"][0].argmax(-1)
        assert read.tolist() == [[colours[f * 9 + c] for c in OUTER_CELLS] for f in range(6)], g


def test_the_symmetric_loss_pairs_the_faces_of_the_pose_with_the_true_ones():
    from cubevision.geometry import FACE_PERM, SYMMETRIES

    g = 11
    true = torch.eye(3)[None]
    pose = (true[0] @ torch.as_tensor(SYMMETRIES[g]))[None]
    labels = torch.arange(6).view(1, 6, 1).repeat(1, 1, 9)  # each face all its own colour
    stickers = torch.full((1, 6, 9, NUM_CLASSES), -10.0)
    for p in range(6):
        stickers[0, p, :, FACE_PERM[g][p]] = 10.0
    out = {"present": torch.full((1, 6), 5.0), "stickers": stickers, "turning": torch.full((1,), -5.0),
           "orientation": torch.cat([pose[0, :, 0], pose[0, :, 1]])[None]}
    loss = reader_loss(out, torch.ones(1, 6), labels, torch.zeros(1), LossWeights(label_smoothing=0.0), true, torch.ones(1))
    assert loss["stickers"] < 1e-3 and loss["orientation"] < 1e-6 and loss["present"] < 0.01


def test_a_symmetric_geometric_reader_exports_the_usual_outputs():
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    model = CubeReader(ModelConfig(pretrained=False, image_size=160, geometric=True, symmetric=True)).eval()
    torch.backends.mha.set_fastpath_enabled(False)
    wrapper = ExportWrapper(model).eval()
    x = torch.rand(2, 3, 160, 160)
    import io

    buffer = io.BytesIO()
    torch.onnx.export(wrapper, (x,), buffer, input_names=["image"], output_names=["present", "stickers", "turning", "orientation"],
                      opset_version=17, dynamic_axes={"image": {0: "batch"}}, dynamo=False)
    onnx.checker.check_model(onnx.load_from_string(buffer.getvalue()))
    got = ort.InferenceSession(buffer.getvalue(), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})
    with torch.no_grad():
        want = wrapper(x)
    assert [g.shape for g in got] == [(2, 6), (2, 6, 9, NUM_CLASSES), (2, 1), (2, 3, 3)]
    assert all(abs(g - w.numpy()).max() < 1e-3 for g, w in zip(got, want, strict=True))


def test_a_mobilenetv4_reader_reads_and_exports():
    pytest.importorskip("timm")
    ort = pytest.importorskip("onnxruntime")
    model = CubeReader(ModelConfig(backbone="mobilenetv4_conv_medium", pretrained=False, image_size=160, dim=64, heads=4,
                                   token_stride=8, colour_tokens=True, geometric=True, symmetric=True)).eval()
    torch.backends.mha.set_fastpath_enabled(False)
    wrapper = ExportWrapper(model).eval()
    x = torch.rand(2, 3, 160, 160)
    import io

    buffer = io.BytesIO()
    torch.onnx.export(wrapper, (x,), buffer, input_names=["image"], output_names=["present", "stickers", "turning", "orientation"],
                      opset_version=17, dynamic_axes={"image": {0: "batch"}}, dynamo=False)
    got = ort.InferenceSession(buffer.getvalue(), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})
    assert [g.shape for g in got] == [(2, 6), (2, 6, 9, NUM_CLASSES), (2, 1), (2, 3, 3)]
