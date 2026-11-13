"""Scenario sampling and labels; the rendering checks run only where bpy is installed."""
import dataclasses
import importlib.util

import pytest

from cubevision.cube import Cube, parse_move
from cubevision.labels import HIDDEN
from cubevision.render import generate
from cubevision.render.annotate import annotate
from cubevision.render.generate import _args, _manifest, done_indices
from cubevision.render.sampler import HOLDS, SETTLED, Turn, sample

HAS_BPY = importlib.util.find_spec("bpy") is not None


def test_sampling_is_deterministic():
    assert sample(3, 17) == sample(3, 17)
    assert sample(3, 17) != sample(3, 18)


def test_a_dataset_folder_keeps_its_settings(tmp_path):
    _manifest(_args(["--out", str(tmp_path), "--seed", "1"]))
    _manifest(_args(["--out", str(tmp_path), "--seed", "1", "--start", "500"]))  # more of the same dataset
    with pytest.raises(SystemExit):
        _manifest(_args(["--out", str(tmp_path), "--seed", "2"]))


def test_a_resumed_run_sees_every_worker_s_images(tmp_path):
    # A first run of 6 images by 2 workers; resumed with 3 workers, worker 0 now covers 0..3
    for start, indices in ((0, [0, 1, 2]), (3, [3, 4])):
        lines = [f'{{"image": "images/{i:07d}.jpg", "meta": {{"index": {i}}}}}' for i in indices]
        (tmp_path / f"labels-{start:07d}.jsonl").write_text("\n".join(lines) + "\n")
    assert done_indices(tmp_path) == {0, 1, 2, 3, 4}


def test_a_worker_that_hands_over_is_replaced_until_its_range_is_done(monkeypatch):
    launched = []
    exits = iter([generate.RECYCLED, 0, generate.RECYCLED, 0])

    class Proc:
        def __init__(self, cmd, cwd):
            launched.append(cmd[cmd.index("--start") + 1])
            self.returncode = next(exits)

        def poll(self):
            return self.returncode

    monkeypatch.setattr(generate.subprocess, "Popen", Proc)
    monkeypatch.setattr(generate.time, "sleep", lambda s: None)
    assert generate._spawn(_args(["--out", "x", "--count", "10", "--workers", "2", "--recycle", "3"])) == 0
    # Worker 0 (images 0-4) handed over twice, worker 5-9 finished at once; each fresh one goes on from its range
    assert launched == ["0", "5", "0", "0"]


def test_24_holds():
    assert len(HOLDS) == 24


def test_annotate_reads_visible_faces_by_centre_colour():
    stickers = Cube().apply("R U").stickers.tolist()
    visible = [False] * 54
    for i in range(18, 27):  # the F face, fully visible
        visible[i] = True
    facing = [0.0, 0.0, 0.9, 0.0, 0.0, 0.0]
    faces, turning = annotate(stickers, None, visible, facing)
    assert not turning
    assert [f.present for f in faces] == [False, False, True, False, False, False]
    assert faces[2].stickers == [s // 9 for s in stickers[18:27]]
    assert faces[0].stickers == [HIDDEN] * 9


def test_annotate_needs_the_centre_and_enough_stickers():
    stickers = list(range(54))
    visible = [False] * 54
    for i in (18, 19, 20, 21, 23, 24):  # six stickers but not the centre (22)
        visible[i] = True
    faces, _ = annotate(stickers, None, visible, [0.0, 0.0, 0.9, 0.0, 0.0, 0.0])
    assert not faces[2].present


def test_annotate_drops_the_faces_cut_by_a_turn():
    stickers = list(range(54))
    visible = [True] * 54
    facing = [0.9] * 6
    faces, turning = annotate(stickers, Turn("R", -40.0, True), visible, facing)
    assert turning
    # R turns about x: U F D B are cut, R and L stay whole
    assert [f.present for f in faces] == [False, True, False, False, True, False]
    faces, turning = annotate(stickers, Turn("R", -3.0, False), visible, facing)
    assert not turning and all(f.present for f in faces)


def test_annotate_wild_reads_faces_almost_edge_on_but_not_specks():
    stickers = list(range(54))
    visible = [i in range(18, 27) for i in range(54)]
    facing = [0.0, 0.0, 0.06, 0.0, 0.0, 0.0]  # F about 3.5° from edge-on
    area = [4e-3 if i in range(18, 27) else 0.0 for i in range(54)]
    assert not annotate(stickers, None, visible, facing, area)[0][2].present
    assert annotate(stickers, None, visible, facing, area, wild=True)[0][2].present
    specks = [1e-4 if i in range(18, 27) else 0.0 for i in range(54)]  # under 4 pixels of a 192 px crop
    face = annotate(stickers, None, visible, facing, specks, wild=True)[0][2]
    assert not face.present and face.stickers == [HIDDEN] * 9


def test_wild_leaves_the_state_and_the_hands_alone():
    tags = set()
    for i in range(300):
        plain, wild = sample(7, i, variety=True), sample(7, i, variety=True, wild=True)
        assert (wild.stickers, wild.turn, wild.hands, wild.style) == (plain.stickers, plain.turn, plain.hands, plain.style)
        tags |= set(wild.meta["wild"].split("+"))
    assert {"rotation", "wobble", "glare", "overexposed", "bright", "backlit", "glossy", "frosted", "printed", "fluo"} <= tags


def test_loose_leaves_the_state_and_the_turn_alone():
    tags = set()
    for i in range(300):
        wild, loose = sample(7, i, variety=True, wild=True), sample(7, i, variety=True, wild=True, loose=True)
        assert (loose.stickers, loose.turn, loose.hands, loose.orientation) == (wild.stickers, wild.turn, wild.hands, wild.orientation)
        if loose.slack:
            # Settled however far off: along the turning layer's axis, every layer under SETTLED
            assert all(abs(a) < SETTLED for a in loose.slack["angles"])
            if loose.turn:
                assert loose.slack["axis"] == parse_move(loose.turn.move).axis
        tags |= set(loose.meta["loose"].split("+"))
    assert {"slack", "smear"} <= tags


def test_stages_draw_the_states_of_real_solves():
    import numpy as np

    from cubevision.render.sampler import _stage_state

    seen = set()
    for i in range(400):
        recipe, cube = _stage_state(np.random.default_rng([1, i, 105]))
        stage, hold = recipe.split(": ")[0], recipe.split(": ")[1].split(" |")[0]
        seen.add(stage.split("+")[0])
        if hold or "+" in stage or stage in ("scrambled", "roux"):
            continue
        colours = cube.colors().reshape(6, 9)
        # Held as built: the cross on D whatever the step after it, the top oriented from the PLL on
        assert all(colours[3][k] == 3 for k in (1, 3, 5, 7)), recipe
        if stage in ("pll", "auf"):
            assert (colours[0] == 0).all(), recipe
    assert seen == {"scrambled", "cross", "f2l", "last_slot", "oll", "pll", "auf", "roux"}
    # The rest of the scenario stays the same
    plain, staged = sample(7, 3, wild=True), sample(7, 3, wild=True, stages=True)
    assert (plain.hands, plain.orientation, plain.turn and plain.turn.move) == (staged.hands, staged.orientation, staged.turn and staged.turn.move)


@pytest.mark.skipif(not HAS_BPY, reason="bpy not installed")
def test_slack_turns_the_layers_a_little_and_any_corner_seen_counts(tmp_path):
    import numpy as np
    from PIL import Image

    from cubevision.render.scene import CubeScene

    sc = dataclasses.replace(
        sample(1, 0),
        stickers=list(range(54)), turn=None, orientation=[1.0, 0.0, 0.0, 0.0],
        azimuth=0.0, elevation=0.0, offset=[0.0, 0.0], roll=0.0, hands=[], occluders=[], shutter=0.0,
    )
    scene = CubeScene(resolution=96, samples=4, device="cpu")
    straight = scene.render(sc, tmp_path / "a.jpg")
    off = scene.render(dataclasses.replace(sc, slack={"axis": 1, "angles": [-6.0, 0.0, 6.0]}), tmp_path / "b.jpg")
    a, b = (np.asarray(Image.open(tmp_path / n), dtype=float) for n in ("a.jpg", "b.jpg"))
    assert np.abs(a - b).mean() > 1.0
    # Still the near face, all of it, and settled
    faces, turning = annotate(sc.stickers, None, off.visible, off.facing)
    assert faces[2].present and faces[2].stickers == [2] * 9 and not turning
    # A tile seen by any of its points is read: never fewer than by the rule before (3 of the 5 inner ones)
    assert all(v or n < 3 for v, n in zip(straight.visible, straight.inner, strict=True))


@pytest.mark.skipif(not HAS_BPY, reason="bpy not installed")
def test_render_sees_the_near_face_of_a_solved_cube(tmp_path):
    from cubevision.render.scene import CubeScene

    sc = dataclasses.replace(
        sample(1, 0),
        stickers=list(range(54)), turn=None, orientation=[1.0, 0.0, 0.0, 0.0],
        azimuth=0.0, elevation=0.0, offset=[0.0, 0.0], roll=0.0, hands=[], occluders=[], shutter=0.0,
    )
    scene = CubeScene(resolution=96, samples=4, device="cpu")
    r = scene.render(sc, tmp_path / "x.jpg")
    faces, turning = annotate(sc.stickers, sc.turn, r.visible, r.facing)
    assert (tmp_path / "x.jpg").exists()
    # Seen from behind the cuber: the near face (F, green) fills the view
    assert faces[2].present and faces[2].stickers == [2] * 9
    assert not any(f.present for k, f in enumerate(faces) if k != 2)
    # The crop is the box times the margin, centred on it
    x0, y0, x1, y1 = r.box
    assert max(x1 - x0, y1 - y0) == pytest.approx(1 / sc.crop_margin, abs=0.02)
    assert (x0 + x1) / 2 == pytest.approx(0.5, abs=0.02) and (y0 + y1) / 2 == pytest.approx(0.5, abs=0.02)
    # Held as scrambled and seen from behind the cuber: the camera's axes are the cube's
    assert r.orientation == pytest.approx([1, 0, 0, 0, 1, 0, 0, 0, 1], abs=1e-4)


@pytest.mark.skipif(not HAS_BPY, reason="bpy not installed")
def test_full_frames_label_the_box_and_the_frames_without_cube(tmp_path):
    from cubevision.render.scene import CubeScene

    scene = CubeScene(resolution=96, samples=2, device="cpu")
    seen = dataclasses.replace(sample(1, 3, framing="full"), cube_hidden=False, offset=[0.0, 0.0], roll=0.0, aspect=16 / 9, frame_fraction=0.3)
    r = scene.render(seen, tmp_path / "a.jpg")
    x0, y0, x1, y1 = r.box
    assert 0 < x0 < x1 < 1 and 0 < y0 < y1 < 1
    # The cube spans about frame_fraction of the (longer) width
    assert 0.4 * seen.frame_fraction < x1 - x0 < 2 * seen.frame_fraction
    hidden = dataclasses.replace(seen, cube_hidden=True)
    r = scene.render(hidden, tmp_path / "b.jpg")
    assert r.box is None and r.orientation is None and not any(r.visible)


@pytest.mark.skipif(not HAS_BPY, reason="bpy not installed")
def test_a_scene_rendered_again_after_another_is_the_same(tmp_path):
    # Cycles keeps its data between renders (use_persistent_data): every change must still reach it
    import numpy as np
    from PIL import Image

    from cubevision.render.scene import CubeScene

    scene = CubeScene(resolution=64, samples=2, device="cpu")
    a, b = sample(7, 1), sample(7, 2)
    scene.render(a, tmp_path / "a1.jpg")
    scene.render(b, tmp_path / "b.jpg")
    scene.render(a, tmp_path / "a2.jpg")
    a1, a2, other = (np.asarray(Image.open(tmp_path / n), dtype=float) for n in ("a1.jpg", "a2.jpg", "b.jpg"))
    assert np.abs(a1 - a2).mean() < 1.0
    assert np.abs(a1 - other).mean() > 5.0


@pytest.mark.skipif(not HAS_BPY, reason="bpy not installed")
def test_the_exposure_scales_the_lights_not_the_view(tmp_path):
    # A new view exposure with each switch of view transform leaked 1.5 MB per image (WSL froze)
    from cubevision.render.scene import CubeScene

    scene = CubeScene(resolution=48, samples=1, device="cpu")
    sc = dataclasses.replace(sample(5, 0), exposure=1.0)
    scene.render(sc, tmp_path / "a.jpg")
    assert scene.scene.view_settings.exposure == 0.0
    lit = [light for light in scene.lights if not light.hide_render]
    assert lit and lit[0].data.energy == pytest.approx(2 * sc.lights[0].energy)


def test_distractors_leave_the_rest_of_the_scenario_alone():
    plain, busy = sample(5, 9, framing="full"), sample(5, 9, framing="full", distractors=True)
    assert plain.distractors == []
    assert dataclasses.replace(busy, distractors=[], cube_hidden=plain.cube_hidden) == plain
    counts = [len(sample(5, i, framing="full", distractors=True).distractors) for i in range(200)]
    assert 1.5 < sum(counts) / len(counts) < 2.2
    for d in (d for i in range(50) for d in sample(5, i, framing="full", distractors=True).distractors):
        # Look-alikes, never a 3×3 grid on all sides like the cube
        assert d["cells"] != [3, 3, 3]


@pytest.mark.skipif(not HAS_BPY, reason="bpy not installed")
def test_distractors_are_drawn_and_leave_the_cube_s_labels(tmp_path):
    import numpy as np
    from PIL import Image

    from cubevision.render.scene import CubeScene

    scene = CubeScene(resolution=96, samples=2, device="cpu")
    seen = dataclasses.replace(sample(1, 3, framing="full"), cube_hidden=False, offset=[0.0, 0.0], roll=0.0, aspect=16 / 9, frame_fraction=0.3)
    busy = dataclasses.replace(seen, distractors=[{
        "kind": "tiles", "size": [3, 3, 3], "rotation": [1, 0.2, 0.3, 0], "depth": 3.0, "across": [0.5, 0.2], "cells": [4, 4, 4],
        "gap": 0.08, "colours": [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0.5, 0], [1, 1, 1]], "seed": 1.0, "roughness": 0.4}])
    a, b = scene.render(seen, tmp_path / "a.jpg"), scene.render(busy, tmp_path / "b.jpg")
    assert a.box == pytest.approx(b.box) and a.visible == b.visible
    pa, pb = (np.asarray(Image.open(tmp_path / n), dtype=float) for n in ("a.jpg", "b.jpg"))
    assert np.abs(pa - pb).mean() > 2.0
    # Back to none: the same image as before
    scene.render(seen, tmp_path / "c.jpg")
    assert np.abs(pa - np.asarray(Image.open(tmp_path / "c.jpg"), dtype=float)).mean() < 1.0


def test_real_hands_leave_the_rest_of_the_scenario_alone():
    plain, real = sample(5, 9), sample(5, 9, real_hands=True)
    assert dataclasses.replace(real, hands=plain.hands) == plain
    assert len(real.hands) == len(plain.hands) and all(h.model == "capsules" and len(h.fingers) == 4 for h in real.hands)
    assert [h.side for h in real.hands] == [h.side for h in plain.hands]


@pytest.mark.skipif(not HAS_BPY, reason="bpy not installed")
def test_real_hands_are_drawn_then_hidden_again(tmp_path):
    import numpy as np
    from PIL import Image

    from cubevision.render.scene import CubeScene

    scene = CubeScene(resolution=64, samples=2, device="cpu")
    index = next(i for i in range(50) if len(sample(3, i).hands) == 2)
    blobs, capsules = sample(3, index), sample(3, index, real_hands=True)
    for name, sc in (("a", blobs), ("b", capsules), ("c", blobs)):
        scene.render(sc, tmp_path / f"{name}.jpg")
    a, b, c = (np.asarray(Image.open(tmp_path / f"{n}.jpg"), dtype=float) for n in "abc")
    assert np.abs(a - b).mean() > 2.0 and np.abs(a - c).mean() < 1.0


def test_variety_changes_only_the_cube_s_wear_and_the_light():
    for i in range(30):
        plain, varied = sample(6, i), sample(6, i, variety=True)
        same = dataclasses.replace(varied, palette=plain.palette, tile_jitter=plain.tile_jitter, background=plain.background,
                                   exposure=plain.exposure, lights=plain.lights, meta=plain.meta)
        assert same == plain
    kinds = {sample(6, i, variety=True).meta.get("light", "normal") for i in range(200)}
    assert kinds == {"normal", "backlit", "dim", "spot"}


@pytest.mark.skipif(not HAS_BPY, reason="bpy not installed")
def test_photo_backdrops_do_not_pile_up_in_memory(tmp_path):
    import bpy
    from PIL import Image

    from cubevision.render.scene import CubeScene

    photos = []
    for k in range(6):
        Image.new("RGB", (32, 24), (40 * k, 100, 200 - 30 * k)).save(tmp_path / f"p{k}.jpg")
        photos.append(tmp_path / f"p{k}.jpg")
    scene = CubeScene(resolution=32, samples=1, device="cpu", backgrounds=photos)
    for k in range(6):
        sc = sample(9, k, backgrounds=len(photos))
        scene.render(dataclasses.replace(sc, background={**sc.background, "kind": "image", "image": k}), tmp_path / f"{k}.jpg")
    assert len([im for im in bpy.data.images if im.filepath]) <= 1


def test_harsh_leaves_the_state_and_the_turn_alone():
    for i in range(60):
        plain, harsh = sample(9, i, wild=True, loose=True, stages=True), sample(9, i, wild=True, loose=True, stages=True, harsh=True)
        assert harsh.stickers == plain.stickers and harsh.turn == plain.turn and harsh.slack == plain.slack
        assert "harsh" in harsh.meta and (harsh.focus is None or harsh.focus > 0)
