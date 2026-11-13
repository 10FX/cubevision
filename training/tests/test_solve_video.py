"""Synthetic solve videos: the solve played is the truth written (no Blender needed)."""
import numpy as np
import pytest

from cubevision.cube import Cube
from cubevision.render.sampler import HOLDS
from cubevision.render.solve_video import (
    F2L_ALGS,
    OLL_ALGS,
    PLL_ALGS,
    agitated,
    in_cube_frame,
    in_cuber_frame,
    peeked,
    plan,
    poses,
    quaternion_of,
    reheld,
    timed,
    turned_hold,
)

SIDES = (1, 2, 4, 5)  # R F L B


def _face(cube: Cube, f: int) -> np.ndarray:
    return cube.colors().reshape(6, 9)[f]


def cross_done(c: Cube) -> bool:
    d = _face(c, 3)
    return all(d[i] == d[4] for i in (1, 3, 5, 7)) and all(_face(c, f)[7] == _face(c, f)[4] for f in SIDES)


def f2l_done(c: Cube) -> bool:
    return bool((_face(c, 3) == _face(c, 3)[4]).all()) and all((_face(c, f)[3:] == _face(c, f)[4]).all() for f in SIDES)


def oll_done(c: Cube) -> bool:
    return f2l_done(c) and bool((_face(c, 0) == _face(c, 0)[4]).all())


def test_a_move_written_by_the_cuber_turns_the_layer_facing_that_way():
    rng = np.random.default_rng(0)
    tokens = ["R", "U'", "F2", "D", "L'", "B", "Rw", "r'", "M", "E'", "S2", "x", "y'", "z2", "x'", "y2"]
    start = Cube().apply("R U2 F' L D2")
    for _ in range(300):
        written = [str(rng.choice(tokens)) for _ in range(12)]
        hold, in_cube, rotations = np.eye(3), [], []
        for token in written:
            if token[0] in "xyz":
                hold = turned_hold(token, hold)
                rotations.append(token)
            else:
                in_cube.append(in_cube_frame(token, hold))
        # Seen by the cuber: the cube's own moves, then the whole cube turned as the rotations turned it
        assert start.apply(" ".join(written)) == start.apply(" ".join(in_cube)).apply(" ".join(rotations) or [])


@pytest.mark.parametrize("alg", F2L_ALGS)
def test_f2l_insertions_keep_the_cross(alg):
    assert cross_done(Cube().apply(alg))


@pytest.mark.parametrize("alg", OLL_ALGS)
def test_olls_keep_the_first_two_layers(alg):
    assert f2l_done(Cube().apply(alg))


@pytest.mark.parametrize("alg", PLL_ALGS)
def test_plls_keep_the_last_layer_oriented(alg):
    assert oll_done(Cube().apply(alg))


@pytest.mark.parametrize("seed", range(40))
def test_each_stage_ends_done_and_the_solve_solves_the_scramble(seed):
    rng = np.random.default_rng(seed)
    tokens, scramble = plan(rng)
    solve = timed(rng, tokens, scramble)
    assert solve.steps[0].token == "x2" and solve.steps[0].start > 600  # inspection, then the cross on D
    assert all(a.end <= b.start for a, b in zip(solve.steps, solve.steps[1:], strict=False))
    # Played as written from the scrambled cube (Cube.apply turns the whole cube on x, y, z: the cuber's view)
    cube = Cube().apply(scramble)
    checks = {"cross": cross_done, "f2l": f2l_done, "oll": oll_done, "pll": Cube.is_solved}
    for k, step in enumerate(solve.steps):
        cube = cube.apply(step.token)
        if k + 1 == len(solve.steps) or solve.steps[k + 1].stage != step.stage:
            assert checks[step.stage](cube), (step.stage, solve.written)
            assert solve.stages[step.stage] == step.end


def test_the_frames_show_the_scramble_the_turns_and_the_solved_cube():
    rng = np.random.default_rng(3)
    tokens, scramble = plan(rng)
    solve = timed(rng, tokens, scramble)
    first = solve.steps[0]
    move = next(s for s in solve.steps if s.token[0] not in "xyz")
    at = poses(solve, [0.0, (first.start + first.end) / 2, (move.start + move.end) / 2, solve.duration])
    assert at[0].stickers == Cube().apply(scramble).stickers.tolist() and at[0].turn is None
    # Mid-x2: the cube half way between two holds, no layer turned
    assert at[1].turn is None and at[1].stickers == at[0].stickers
    # Mid-move: the layer drawn turned from the state before the move
    assert at[2].turn is not None and abs(at[2].turn.angle) > 10
    assert Cube(np.array(at[3].stickers)).is_solved() and at[3].turn is None


def test_no_two_turns_of_the_same_layer_in_a_row():
    from cubevision.cube import parse_move
    from cubevision.render.solve_video import merged

    assert merged([("U'", "f2l"), ("U", "f2l"), ("U'", "f2l"), ("R", "f2l")]) == [("U'", "f2l"), ("R", "f2l")]
    assert merged([("U", "oll"), ("U", "oll")]) == [("U2", "oll")]
    for seed in range(40):
        tokens, _ = plan(np.random.default_rng(seed))
        # Within a stage, or across two
        layers = [(m.family, m.first, m.last) for m in (parse_move(t) for t, _ in tokens)]
        assert all(a != b for a, b in zip(layers, layers[1:], strict=False)), tokens


def test_the_hands_move_smoothly_from_grip_to_grip():
    from cubevision.render.sampler import sample
    from cubevision.render.solve_video import hands_at, hands_over_time

    hands = sample(1, 4).hands
    assert hands
    times, grips = hands_over_time(np.random.default_rng(0), hands, 5000)
    assert times[-1] >= 5000 and all(300 <= b - a <= 700 for a, b in zip(times, times[1:], strict=False))
    assert hands_at(times, grips, 0.0)[0].palm_depth == pytest.approx(hands[0].palm_depth)
    # Half way between two grips: between them, and a few ms apart barely differs
    mid = hands_at(times, grips, (times[1] + times[2]) / 2)[0].fingers[0].reach
    lo, hi = sorted([grips[1][0].fingers[0].reach, grips[2][0].fingers[0].reach])
    assert lo <= mid <= hi
    a, b = hands_at(times, grips, 1234.0)[0], hands_at(times, grips, 1244.0)[0]
    assert abs(a.palm_height - b.palm_height) < 0.05


def test_corner_cutting_starts_a_turn_before_the_last_ends_and_keeps_their_order():
    rng = np.random.default_rng(5)
    tokens, scramble = plan(rng)
    solve = timed(np.random.default_rng(1), tokens, scramble, tps=10, cut=0.8)
    turns = [s for s in solve.steps if s.token[0] not in "xyz"]
    cuts = [(a, b) for a, b in zip(turns, turns[1:], strict=False) if b.start < a.end]
    assert len(cuts) > 10
    # Each ends after the one before it: the moves stay in their order
    assert all(b.end > a.end and b.start >= a.start + 0.5 * (a.end - a.start) - 1e-6 for a, b in cuts)
    # While both turn, the frame draws both layers
    a, b = cuts[0]
    pose = poses(solve, [(b.start + a.end) / 2])[0]
    assert pose.turn is not None and pose.turn_next is not None
    # Without cutting, no overlap
    plain = timed(np.random.default_rng(1), tokens, scramble, tps=10)
    assert all(b.start >= a.end for a, b in zip(plain.steps, plain.steps[1:], strict=False))


def test_a_finger_pushes_the_layer_it_turns_and_leaves_it_after():
    from cubevision.render.solve_video import TRICK_OUT, tricks_at

    tokens, scramble = [("R", "cross"), ("U", "f2l"), ("U'", "oll"), ("F", "pll")], "F' U U' R'"
    solve = timed(np.random.default_rng(0), tokens, scramble, tps=2)
    r, u, u_, _ = solve.steps
    # R: the right hand turns at the wrist, as far as the layer
    mid = tricks_at(solve, (r.start + r.end) / 2)
    assert [(t["side"], t["finger"]) for t in mid] == [(1, -1)] and -90 < mid[0]["angle"] < 0 and mid[0]["weight"] == 1
    # U with the right index on the top layer of the far face, U' with the left one
    assert [(t["side"], t["finger"], t["target"]) for t in tricks_at(solve, (u.start + u.end) / 2)] == [(1, 1, "far")]
    assert [(t["side"], t["finger"]) for t in tricks_at(solve, (u_.start + u_.end) / 2)] == [(-1, 1)]
    assert all(t["finger"] != 1 for t in tricks_at(solve, u_.end + TRICK_OUT + 1))


def test_the_camera_look_keeps_the_picture_and_changes_it_a_little():
    from cubevision.render.solve_video import CameraLook

    img = np.full((90, 160, 3), 128, np.uint8)
    img[30:60, 60:100] = (200, 40, 40)
    look = CameraLook(np.random.default_rng(2))
    out = look(img, 500.0)
    assert out.shape == img.shape and out.dtype == np.uint8
    # Still the same picture (the red patch where it was), with noise and drift
    assert abs(float(out.astype(float).mean()) - float(img.astype(float).mean())) < 25
    assert out[45, 80, 0] > out[45, 80, 1] + 60
    assert np.abs(out.astype(int) - img.astype(int)).mean() > 0.5


def _cube_moves(tokens: list[tuple[str, str]]) -> list[str]:
    hold, out = np.eye(3), []
    for token, _ in tokens:
        if token[0] in "xyz":
            hold = turned_hold(token, hold)
        else:
            out.append(in_cube_frame(token, hold))
    return out


def test_in_cuber_frame_undoes_in_cube_frame():
    hold = np.eye(3)
    holds = []
    for a in ("", "x", "x2", "x'", "z", "z'"):
        for b in ("", "y", "y2", "y'"):
            h = np.eye(3)
            for r in (a, b):
                if r:
                    h = turned_hold(r, h)
            holds.append(h)
    assert len({tuple(quaternion_of(h)) for h in holds}) == len(HOLDS)
    for hold in holds:
        for token in ("R", "U'", "F2", "D", "L'", "B2", "Rw", "r'", "M", "M'", "E2", "S", "Fw2"):
            assert in_cube_frame(in_cuber_frame(in_cube_frame(token, hold), hold), hold) == in_cube_frame(token, hold)


@pytest.mark.parametrize("seed", range(8))
def test_extra_rotations_turn_the_same_layers(seed):
    rng = np.random.default_rng(seed)
    tokens, scramble = plan(rng)
    turned = reheld(np.random.default_rng(seed + 100), tokens, 0.3)
    assert sum(t[0] in "xyz" for t, _ in turned) > sum(t[0] in "xyz" for t, _ in tokens)
    assert _cube_moves(turned) == _cube_moves(tokens)
    cube = Cube().apply(scramble)
    for token, _ in turned:
        cube = cube.apply(token)
    assert cube.is_solved()


@pytest.mark.parametrize("seed", range(4))
def test_looks_at_another_side_turn_the_cube_back_and_the_same_layers(seed):
    tokens, scramble = plan(np.random.default_rng(seed))
    looked = peeked(np.random.default_rng(seed + 200), tokens, 0.3)
    rotations = [t for t, _ in looked if t[0] in "xyz"]
    assert len(rotations) > sum(t[0] in "xyz" for t, _ in tokens)
    assert _cube_moves(looked) == _cube_moves(tokens)
    solve = timed(np.random.default_rng(seed), looked, scramble)
    cube = Cube().apply(scramble)
    for step in solve.steps:
        cube = cube.apply(step.token)
    assert cube.is_solved()


def test_an_agitated_cube_wobbles_more_and_jerks():
    rng = np.random.default_rng(5)
    tokens, scramble = plan(rng)
    solve = agitated(np.random.default_rng(6), timed(rng, tokens, scramble))
    assert solve.tilt >= 15 and solve.jerks
    assert any(j[0] < solve.steps[0].start for j in solve.jerks)  # turned this way and that during the inspection
    frames = poses(solve, [0.0, solve.duration])
    assert Cube(np.array(frames[-1].stickers)).is_solved()


def test_quaternion_of_is_right_for_every_hold_half_turns_included():
    from cubevision.render.sampler import _quaternion_multiply

    def matrix(q):
        w, x, y, z = q
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    for q in HOLDS:
        assert np.allclose(matrix(quaternion_of(np.rint(matrix(q)))), np.rint(matrix(q)), atol=1e-9)
    assert _quaternion_multiply  # the sampler's own product, same convention
