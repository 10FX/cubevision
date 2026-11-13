"""The sculpted hand's grip (hand_mesh): its geometry, and on the asset when built (data/assets/hand.npz)."""
import dataclasses

import numpy as np
import pytest

from cubevision.render import hand_mesh as hm
from cubevision.render.sampler import sample
from cubevision.render.solve_video import _cuber

HAS_ASSET = hm.ASSET.exists()


def test_rotations_and_turns_agree():
    axis = np.array([0.0, 0.0, 1.0])
    assert np.allclose(hm.rotation(axis, 90) @ [1.0, 0, 0], [0, 1.0, 0])
    v = np.array([[1.0, 0.0, 0.0]])
    assert np.allclose(hm._turn(v, axis[None], np.array([90.0]))[0], [0, 1.0, 0])
    assert np.allclose(hm.swing(np.array([1.0, 0, 0]), np.array([0, 0, 2.0])) @ [1.0, 0, 0], [0, 0, 1.0])


def test_the_cube_s_distance():
    d = hm._box_distance(np.array([[0.0, 0.0, 0.0], [hm.HALF + 1, 0.0, 0.0], [0.0, 0.0, hm.HALF]]))
    assert d[0] < 0 and d[1] == pytest.approx(1.0) and d[2] == pytest.approx(0.0, abs=1e-9)


def test_a_left_hand_s_push_is_the_right_hand_s_mirrored():
    push = {"side": -1, "finger": 1, "axis": [1.0, 0.5, -0.2], "angle": 40.0, "weight": 1.0}
    m = hm.mirror_trick(push)
    # Turning a point then mirroring it is mirroring it then turning it the mirrored way
    p = np.array([0.3, -1.2, 0.7])
    a = hm.MIRROR @ hm.rotation(np.array(push["axis"]), push["angle"]) @ p
    b = hm.rotation(np.array(m["axis"]), m["angle"]) @ hm.MIRROR @ p
    assert np.allclose(a, b)


def _hands(indexes=(1, 5, 9)):
    for index in indexes:
        base = sample(22, index, framing="full", backgrounds=0, distractors=False, real_hands=True, variety=False)
        yield from _cuber(np.random.default_rng([22, index, 17]), base, True, False).hands


@pytest.mark.skipif(not HAS_ASSET, reason="the hand asset is not built (python -m cubevision.render.hand_mesh)")
def test_the_fingertips_hold_the_cube_without_going_in():
    rig = hm.Rig.load()
    for hand in _hands():
        pose = hm.hold(rig, hand, (), seed=3)
        tips = np.array([pose.transforms[3 + 3 * c][:, :3] @ rig.joints[c, 3] + pose.transforms[3 + 3 * c][:, 3] for c in range(5)])
        gap = hm._box_distance(tips)
        radii = rig.radii * hm.UNIT * hand.palm_scale
        # Thumb, index, middle and ring on the cube (the tip's centre a finger's radius off), none inside
        assert (gap[:4] < radii[:4] + 0.4).all(), gap
        assert (gap > 0).all(), gap
        # The skinned hand is whole: no vertex thrown away
        verts = rig.skin(pose.transforms)
        assert np.isfinite(verts).all() and np.ptp(verts, axis=0).max() < 15


@pytest.mark.skipif(not HAS_ASSET, reason="the hand asset is not built")
def test_grips_are_kept_and_solved_ahead_alike():
    rig = hm.Rig.load()
    hand = next(_hands((1,)))
    hm._GRIPS.clear()
    first = hm.hold(rig, hand, (), seed=1).transforms
    assert np.array_equal(first, hm.hold(rig, hand, (), seed=1).transforms)
    moved = dataclasses.replace(hand, palm_depth=hand.palm_depth + 0.1)
    hm.prime([(moved, 1)], workers=1)
    ahead = hm.hold(rig, moved, (), seed=1).transforms
    hm._GRIPS.clear()
    assert np.allclose(ahead, hm.hold(rig, moved, (), seed=1).transforms)


@pytest.mark.skipif(not HAS_ASSET, reason="the hand asset is not built")
def test_a_push_moves_only_its_finger():
    rig = hm.Rig.load()
    hand = next(_hands((1,)))
    still = hm.hold(rig, hand, (), seed=1).transforms
    pushed = hm.hold(rig, hand, [{"side": 1, "finger": 1, "axis": [1.0, 0.0, 0.0], "angle": 30.0, "weight": 1.0}], seed=1).transforms
    moved = [b for b in range(16) if not np.allclose(still[b], pushed[b])]
    assert moved == [4, 5, 6]
