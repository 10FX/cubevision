"""The GPU decoder finds the turns of a solve from frames read (need PyTorch)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cubevision.cube import FACE_MOVES, Cube, invert_alg  # noqa: E402
from cubevision.decode import DecodeConfig, Moves, decode, frames_of, state_of  # noqa: E402

SCRAMBLE = "R U R' U' F2 D L2 B' R2 U2 F' L D2"


def _frames(scramble: str, solution: list[str], faces=(0, 1, 2), wrong: float = 0.0, seed: int = 0, unseen=()) -> list[dict]:
    """Frames of a solve read faces ``faces`` 6 times between turns (30 fps), a share ``wrong`` of them misread,
    none after the turns ``unseen`` (indexes in the solution)."""
    rng = np.random.default_rng(seed)
    cube, out, t = Cube().apply(scramble), [], 0.0
    for k, move in enumerate([None, *solution]):
        if move:
            cube = cube.apply(move)
            t += 100
        colours = cube.colors()
        for _ in range(6):
            t += 33
            if k - 1 in unseen:
                continue
            read = []
            for face in faces:
                stickers = []
                for i in range(9):
                    p = np.full(6, 0.02)
                    colour = int(colours[face * 9 + i]) if rng.random() >= wrong else int(rng.integers(6))
                    p[colour] = 0.9
                    stickers.append(p.tolist())
                read.append({"center": face, "stickers": stickers})
            out.append({"t": t, "faces": read})
    return out


@pytest.mark.parametrize("wrong", [0.0, 0.1])
def test_the_turns_come_back_from_the_frames(wrong):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solution = invert_alg(SCRAMBLE).split()
    frames = frames_of(_frames(SCRAMBLE, solution, wrong=wrong), device)
    d = decode(state_of(SCRAMBLE), frames, DecodeConfig(beam=256), Moves(device))
    assert d.solved
    assert [m["move"] for m in d.moves] == solution
    # Each turn between the frames before and after it
    assert all(a["t"] < b["t"] for a, b in zip(d.moves, d.moves[1:], strict=False))


@pytest.mark.parametrize("unseen", [range(3, 8), range(2, 10)])
def test_turns_no_frame_saw(unseen):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    solution = invert_alg(SCRAMBLE).split()
    frames = frames_of(_frames(SCRAMBLE, solution, unseen=unseen), device)
    d = decode(state_of(SCRAMBLE), frames, DecodeConfig(beam=256), Moves(device))
    found = [m["move"] for m in d.moves]
    if len(unseen) <= 8:
        # Joined across the gap: the turns found make the same cube, the turns around it are the solve's
        assert d.solved and not d.gaps
        assert found[:unseen[0]] == solution[:unseen[0]] and found[-3:] == solution[-3:]
        assert (state_of(f"{SCRAMBLE} {' '.join(found)}") == state_of("")).all()
    else:
        # Too many to join: the turns before and after the gap, the gap told (the first turn after the
        # unseen ones is in the gap too: no frame before it)
        after = solution[unseen[-1] + 2:]
        assert not d.solved and len(d.gaps) == 1
        assert found[:unseen[0]] == solution[:unseen[0]] and found[-len(after):] == after


def test_moves_are_the_app_s():
    assert FACE_MOVES[:3] == ("U", "U2", "U'")
    # Applying the gathers is applying the moves
    moves = Moves(torch.device("cpu"))
    state = torch.tensor(state_of(SCRAMBLE))
    for k, m in enumerate(FACE_MOVES):
        assert (state[moves.perms[1][k]].numpy() == state_of(f"{SCRAMBLE} {m}")).all()
