"""
The moves of a solve from the frames the reader read: a beam over cube states, on the GPU.

The scrambled state is known, and the solved one at the end; in between, every frame the reader
read scores the states it could show. Frame after frame, each state kept goes on by no turn, a
turn, or a few when the frames between could not be read; all those cubes are scored at once
(tensors of 54 colours), equal cubes merged, the best kept. At the end, the best state that a
turn or two takes to the solved cube.

    PYTHONPATH=. uv run python -m cubevision.decode runs/dumps-reader-v2 evals/synthetic/synthetic.json runs/decode-synthetic.json
        [--beam 2048] [--device cuda]

Frames: the JSON dumps of scripts/vision/eval.test.ts (DUMP=dir), one per video:
``[{"t", "faces": [{"center", "stickers": [9 × (6 probabilities | null)]}], "orientation"}]``.
Writes for scripts/vision/score.test.ts: ``{"<video>": {"moves": [{"move", "t"}], "rotations": 0, ...}}``,
moves named by the centre colour of their face (U white…), as the app's search names them.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .cube import FACE_MOVES, ROT, Cube, invert_alg

ORIENTATIONS = [f"{r} {y}".strip() for y in ("", "y", "y2", "y'") for r in ("", "x", "x2", "x'", "z", "z'")]


@dataclass
class DecodeConfig:
    #: States kept after each frame
    beam: int = 2048
    #: States that also try three turns in a row (after a gap without readings)
    deep: int = 64
    #: Cost of a turn (log-likelihood a turn must earn)
    move_cost: float = 2.2
    #: The fastest turns of a person, ms
    min_turn_ms: float = 70.0
    #: Cost of each turn beyond what fits in the time between two frames
    too_fast: float = 6.0
    #: Probability that a face read is wrong altogether (a hand across it, blur, edge-on)
    misread: float = 0.5
    #: A view counts fully once this many ms of video back it: frames close in time err alike
    correlation_ms: float = 300.0
    #: Weight of a frame that sees the cube otherwise than the one before (a new view, a turn done)
    fresh: float = 0.5
    #: Share of the stickers two frames must agree on to be the same view
    alike_share: float = 0.5
    #: Turns tried to reach the solved cube after the last frame
    finish: int = 3


# ─── The cube as tensors ──────────────────────────────────────────────────────

def _sources(alg: str) -> np.ndarray:
    """Where every sticker of a state comes from after ``alg``: state'[i] = state[src[i]]."""
    return Cube().apply(alg).stickers.copy()


def _sequences(length: int) -> list[tuple[int, ...]]:
    """Sequences of face turns without two turns of one face in a row nor both orders of opposite faces."""
    out = []
    for seq in itertools.product(range(18), repeat=length):
        faces = [m // 3 for m in seq]
        # Faces U R F D L B: opposite faces share face % 3; U D is kept, D U is the same turns
        if any(a == b or (a % 3 == b % 3 and a > b) for a, b in zip(faces, faces[1:], strict=False)):
            continue
        out.append(seq)
    return out


class Moves:
    """Every sequence of 0..4 turns as one gather of the 54 stickers, forwards (``perms``: state' = state[perm])
    and backwards (``inverse``: the state before the turns from the state after)."""

    def __init__(self, device: torch.device):
        single = [_sources(m) for m in FACE_MOVES]
        self.sequences: list[list[tuple[int, ...]]] = []
        self.perms: list[torch.Tensor] = []
        self.inverse: list[torch.Tensor] = []
        for length in range(5):
            seqs = _sequences(length)
            perms = []
            for seq in seqs:
                p = np.arange(54)
                for m in seq:
                    p = p[single[m]]
                perms.append(p)
            self.sequences.append(seqs)
            self.perms.append(torch.tensor(np.stack(perms), dtype=torch.long, device=device))
            self.inverse.append(torch.tensor(np.argsort(np.stack(perms), axis=1), dtype=torch.long, device=device))


def state_of(alg: str) -> np.ndarray:
    return (Cube().apply(alg).stickers // 9).astype(np.int64)


SOLVED = np.repeat(np.arange(6), 9)


# ─── Frames as scorers ────────────────────────────────────────────────────────

@dataclass
class Frame:
    t: float
    #: For each face read: the indexes of its stickers in the state for each of the 4 rotations
    #: (F, 4, 9), the log-probability of each colour of each cell (F, 9, 6), cells seen (F, 9)
    cells: torch.Tensor
    logs: torch.Tensor
    seen: torch.Tensor
    #: Colour read (argmax) per face and cell, -1 not seen: to tell a new view from the same one
    argmax: dict[int, np.ndarray]


_ROT = torch.tensor(ROT, dtype=torch.long)


def frames_of(dump: list[dict], device: torch.device) -> list[Frame]:
    frames = []
    for f in dump:
        faces = [face for face in f["faces"] if any(p is not None for i, p in enumerate(face["stickers"]) if i != 4)]
        if not faces:
            continue
        centres = torch.tensor([face["center"] for face in faces], dtype=torch.long)
        cells = centres[:, None, None] * 9 + _ROT[None]  # (F, 4, 9)
        logs = torch.zeros(len(faces), 9, 6)
        seen = torch.zeros(len(faces), 9, dtype=torch.bool)
        argmax = {}
        for k, face in enumerate(faces):
            am = np.full(9, -1)
            for i, p in enumerate(face["stickers"]):
                if p is None or i == 4:
                    continue
                q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1)
                logs[k, i] = torch.tensor(np.log(q / q.sum()))
                seen[k, i] = True
                am[i] = int(np.argmax(q))
            argmax[face["center"]] = am
        frames.append(Frame(float(f["t"]), cells.to(device), logs.to(device), seen.to(device), argmax))
    return frames


def alike(a: Frame, b: Frame, share: float) -> bool:
    same = compared = 0
    for centre, cb in b.argmax.items():
        ca = a.argmax.get(centre)
        if ca is None:
            continue
        both = (ca >= 0) & (cb >= 0)
        compared += int(both.sum())
        same += int((both & (ca == cb)).sum())
    return compared >= 4 and same >= share * compared


def score(states: torch.Tensor, frame: Frame, misread: float) -> torch.Tensor:
    """Log-likelihood of a frame for each state (N, 54): per face the best of its 4 rotations, or a misreading."""
    colours = states[:, frame.cells]  # (N, F, 4, 9)
    f, _, i = torch.meshgrid(torch.arange(colours.shape[1], device=states.device), torch.arange(4, device=states.device),
                             torch.arange(9, device=states.device), indexing="ij")
    ll = frame.logs[f, i, colours]  # (N, F, 4, 9)
    ll = (ll * frame.seen[None, :, None, :]).sum(-1).amax(-1)  # (N, F)
    seen = frame.seen.sum(-1)  # (F,)
    garbage = math.log(misread) + seen * math.log(1 / 6)
    return torch.logaddexp(ll + math.log(1 - misread), garbage[None]).sum(-1)


# ─── The beam ─────────────────────────────────────────────────────────────────

@dataclass
class Step:
    """The states kept at a frame: the colours, the score so far, the frame's own part of it, and how
    each came (index of the state it came from at the step before, the turns: length and sequence)."""
    states: torch.Tensor
    scores: torch.Tensor
    emitted: torch.Tensor
    parent: torch.Tensor
    length: torch.Tensor
    sequence: torch.Tensor


def _hash(states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (states.long() * weights).sum(-1)


def _merge_best(states: torch.Tensor, scores: torch.Tensor, keep: int, weights: torch.Tensor) -> torch.Tensor:
    """Indexes of the ``keep`` best distinct states (the best score of each)."""
    order = torch.argsort(scores, descending=True)
    h = _hash(states[order], weights)
    uniq, inverse = torch.unique(h, return_inverse=True)
    first = torch.full((len(uniq),), len(order), dtype=torch.long, device=states.device)
    first.scatter_reduce_(0, inverse, torch.arange(len(order), device=states.device), reduce="amin")
    first = torch.sort(first).values[:keep]
    return order[first]


def _weights(frames: list[Frame], cfg: DecodeConfig) -> list[float]:
    """How much each frame counts: a view counts fully once correlation_ms of video back it, a new one at least ``fresh``."""
    out, prev_t = [], 0.0
    for k, frame in enumerate(frames):
        gap = frame.t - prev_t
        backed = min(1.0, (gap + 1) / cfg.correlation_ms) if cfg.correlation_ms > 0 else 1.0
        out.append(backed if k and alike(frames[k - 1], frame, cfg.alike_share) else max(backed, cfg.fresh))
        prev_t = frame.t
    return out


def _pass(start: torch.Tensor, frames: list[Frame], fits: list[int | None], weights: list[float], cfg: DecodeConfig,
          perms: list[torch.Tensor], hash_weights: torch.Tensor) -> list[Step]:
    """A beam through ``frames`` from the states ``start``: before each frame, ``fits[k]`` turns fit in the time
    (None: any number up to three, at no extra cost); ``perms`` turn the states (forwards, or backwards)."""
    states, scores = start, torch.zeros(len(start), device=start.device)
    steps = []
    for frame, fit, weight in zip(frames, fits, weights, strict=True):
        cand = []
        top = 3 if fit is None else min(3, fit)
        for length in range(top + 1):
            source = torch.arange(len(states), device=start.device)
            if length == 3:
                source = source[:cfg.deep]
            p = perms[length]
            n, s = len(source), len(p)
            new = states[source][:, None, :].expand(n, s, 54).gather(2, p[None].expand(n, s, 54)).reshape(n * s, 54)
            cost = cfg.move_cost * length + (0 if fit is None else cfg.too_fast * max(0, length - fit))
            cand.append((new, (scores[source][:, None] - cost).expand(n, s).reshape(-1), source[:, None].expand(n, s).reshape(-1),
                         torch.full((n * s,), length, device=start.device), torch.arange(s, device=start.device)[None].expand(n, s).reshape(-1)))
        new_states = torch.cat([c[0] for c in cand])
        emitted = weight * score(new_states.long(), frame, cfg.misread)
        new_scores = torch.cat([c[1] for c in cand]) + emitted
        keep = _merge_best(new_states, new_scores, cfg.beam, hash_weights)
        states, scores = new_states[keep], new_scores[keep]
        steps.append(Step(states, scores, emitted[keep], torch.cat([c[2] for c in cand])[keep], torch.cat([c[3] for c in cand])[keep],
                          torch.cat([c[4] for c in cand])[keep]))
    return steps


def _chain(steps: list[Step], k: int, index: int) -> list[tuple[int, int, int]]:
    """(step, length, sequence) of the turns that led to state ``index`` at step ``k``, from the first step."""
    out = []
    for i in range(k, -1, -1):
        step = steps[i]
        out.append((i, int(step.length[index]), int(step.sequence[index])))
        index = int(step.parent[index])
    return out[::-1]


@dataclass
class Decoded:
    moves: list[dict]
    solved: bool
    score: float
    agreement: float
    #: Times (ms) where the turns could not be followed: the moves on either side are right, those between unknown
    gaps: list[float]


def decode(start: np.ndarray, frames: list[Frame], cfg: DecodeConfig, moves: Moves) -> Decoded:
    """The turns from ``start`` to the solved cube that agree best with the frames: a beam forwards from the
    scramble, one backwards from the solved cube, joined where they meet (or across a gap no frame saw)."""
    device = moves.perms[0].device
    n = len(frames)
    # Random weights: equal states, equal hashes (int64 wraps around; a collision is ~2^-60)
    hw = torch.randint(1, 2**62, (54,), generator=torch.Generator().manual_seed(0)).to(device)
    weights = _weights(frames, cfg)
    times = [f.t for f in frames]
    fits = [1 + int((times[k] - (times[k - 1] if k else 0.0)) // cfg.min_turn_ms) for k in range(n)]
    st = lambda a: torch.tensor(a, dtype=torch.uint8, device=device)[None]  # noqa: E731
    forward = _pass(st(start), frames, fits, weights, cfg, [p for p in moves.perms[:4]], hw)
    # Backwards: from the solved cube (a few turns after the last frame), through the frames in reverse, turns undone
    backward = _pass(st(SOLVED), frames[::-1], [None, *fits[:0:-1]], weights[::-1], cfg, moves.inverse[:4], hw)[::-1]

    # Where both beams hold one state at a frame: the frame's own score counted once
    best = None
    for k in range(n):
        f, b = forward[k], backward[k]
        hf, hb = _hash(f.states, hw), _hash(b.states, hw)
        common = torch.isin(hf, hb)
        if not common.any():
            continue
        fi = torch.nonzero(common).flatten()
        order_b = torch.argsort(hb)
        bi = order_b[torch.searchsorted(hb[order_b], hf[fi])]
        total = f.scores[fi] + b.scores[bi] - f.emitted[fi]
        j = int(total.argmax())
        if best is None or float(total[j]) > best[0]:
            best = (float(total[j]), k, int(fi[j]), int(bi[j]))
    if best is not None:
        _, k, i, j = best
        return _assemble(start, frames, moves, forward, backward, (k, i), (k, j), [], True, best[0])

    # The beams never meet: a stretch no frame saw. Join them across the longest gaps, within eight turns
    gaps = sorted(range(1, n), key=lambda k: times[k] - times[k - 1], reverse=True)[:6]
    joined = None
    for k in gaps:
        found = _bridge(forward[k - 1], backward[k], moves, hw, cfg.move_cost)
        if found and (joined is None or found[0] > joined[0]):
            joined = (*found, k)
    if joined:
        total, i, j, seq_a, seq_b, k = joined
        return _assemble(start, frames, moves, forward, backward, (k - 1, i), (k, j), [], True, total, bridge=(seq_a, seq_b))
    # Not even so: the turns before the longest gap and after it, the gap left unknown
    k = max(range(1, n), key=lambda k: float(forward[k - 1].scores.max() + backward[k].scores.max())) if n > 1 else 0
    i, j = int(forward[k - 1].scores.argmax()), int(backward[k].scores.argmax())
    return _assemble(start, frames, moves, forward, backward, (k - 1, i), (k, j), [times[k]], False,
                     float(forward[k - 1].scores[i] + backward[k].scores[j]), bridge=None)


def _bridge(before: Step, after: Step, moves: Moves, hw: torch.Tensor, move_cost: float, top: int = 16):
    """Turns joining one of the best states before a gap to one of the best after (up to four from each side)."""
    a, b = before.states[:top], after.states[:top]
    ball = torch.cat(moves.perms[:5])  # every sequence of 0..4 turns
    inv = torch.cat(moves.inverse[:5])
    xa = a[:, None, :].expand(len(a), len(ball), 54).gather(2, ball[None].expand(len(a), len(ball), 54))
    xb = b[:, None, :].expand(len(b), len(inv), 54).gather(2, inv[None].expand(len(b), len(inv), 54))
    ha, hb = _hash(xa, hw).flatten(), _hash(xb, hw).flatten()
    common = torch.isin(ha, hb)
    if not common.any():
        return None
    lengths = torch.cat([torch.full((len(p),), k, device=ha.device) for k, p in enumerate(moves.perms[:5])])
    ia = torch.nonzero(common).flatten()
    order = torch.argsort(hb)
    ib = order[torch.searchsorted(hb[order], ha[ia])]
    sa, sb = ia // len(ball), ib // len(inv)
    la, lb = lengths[ia % len(ball)], lengths[ib % len(inv)]
    total = before.scores[sa] + after.scores[sb] - move_cost * (la + lb)
    w = int(total.argmax())
    offsets = np.cumsum([0] + [len(p) for p in moves.perms[:5]])

    def split(x: int) -> tuple[int, int]:
        length = int(np.searchsorted(offsets, x, side="right") - 1)
        return length, x - int(offsets[length])

    return float(total[w]), int(sa[w]), int(sb[w]), split(int(ia[w] % len(ball))), split(int(ib[w] % len(inv)))


def _assemble(start, frames, moves, forward, backward, fwd, bwd, gaps, solved, total, bridge=None) -> Decoded:
    """The moves of the path: the forward beam's to ``fwd`` (step, state), the bridge if any, the backward beam's from ``bwd``."""
    out = []

    def add(seq_moves: list[int], after: float, before: float):
        for j, m in enumerate(seq_moves):
            out.append({"move": FACE_MOVES[m], "t": round(after + (before - after) * (j + 1) / len(seq_moves))})

    times = [f.t for f in frames]
    for i, length, seq in _chain(forward, *fwd):
        if length:
            add(list(moves.sequences[length][seq]), times[i - 1] if i else 0.0, times[i])
    if bridge:
        (la, sa), (lb, sb) = bridge
        # The second half was found backwards from the state after the gap: its turns, in the order made
        add([*moves.sequences[la][sa], *moves.sequences[lb][sb]], times[fwd[0]], times[bwd[0]])
    # Backwards, the step of frame k holds the turns made between frame k and the next (the end after the last)
    n = len(frames)
    for i, length, seq in reversed(_chain(backward[::-1], n - 1 - bwd[0], bwd[1])):
        if not length:
            continue
        k = n - 1 - i
        add(list(moves.sequences[length][seq]), times[k], times[k + 1] if k + 1 < n else times[-1] + 150 * length)
    return Decoded(out, solved, total, agreement(start, frames, out), gaps)


def agreement(start: np.ndarray, frames: list[Frame], moves: list[dict]) -> float:
    """Share of the stickers read that the reconstruction explains (best rotation per face)."""
    k, agree, total = 0, 0, 0
    state = start.copy()
    single = {m: _sources(m) for m in FACE_MOVES}
    for frame in frames:
        while k < len(moves) and moves[k]["t"] <= frame.t:
            state = state[single[moves[k]["move"]]]
            k += 1
        cells = frame.cells.cpu().numpy()
        logs = frame.logs.cpu().numpy()
        seen = frame.seen.cpu().numpy()
        for f in range(len(cells)):
            best = max(range(4), key=lambda r: sum(logs[f, i, state[cells[f, r, i]]] for i in range(9) if seen[f, i]))
            for i in range(9):
                if seen[f, i]:
                    total += 1
                    agree += int(np.argmax(logs[f, i]) == state[cells[f, best, i]])
    return agree / total if total else 0.0


def holds_ranked(scramble: str, frames: list[Frame], misread: float, early: int = 6) -> list[tuple[str, np.ndarray]]:
    """The 24 ways the scrambler may have held the cube, best first by the first frames read."""
    out = []
    for hold in ORIENTATIONS:
        start = state_of(f"{hold} {scramble} {invert_alg(hold) if hold else ''}".strip())
        st = torch.tensor(start, device=frames[0].cells.device)[None]
        s = sum(float(score(st, f, misread)[0]) for f in frames[:early])
        out.append((s, hold, start))
    out.sort(key=lambda x: -x[0])
    return [(h, s) for _, h, s in out]


def solve(scramble: str, dump: list[dict], cfg: DecodeConfig, moves: Moves, tries: int = 3) -> tuple[str, Decoded]:
    frames = frames_of(dump, moves.perms[0].device)
    if not frames:
        return "", Decoded([], False, -math.inf, 0.0, [])
    best = None
    for hold, start in holds_ranked(scramble, frames, cfg.misread)[:tries]:
        d = decode(start, frames, cfg, moves)
        if best is None or (d.solved, d.agreement) > (best[1].solved, best[1].agreement):
            best = (hold, d)
    return best


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dumps", type=Path)
    p.add_argument("cases", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--set", nargs="*", default=[], help="DecodeConfig fields, key=value")
    args = p.parse_args(argv)
    cfg = DecodeConfig()
    for kv in args.set:
        key, value = kv.split("=")
        setattr(cfg, key, type(getattr(cfg, key))(value))
    device = torch.device(args.device)
    moves = Moves(device)
    results = {}
    for case in json.loads(args.cases.read_text()):
        name = Path(case["video"]).stem
        dump = args.dumps / f"{name}.json"
        if not dump.exists():
            continue
        t0 = time.time()
        hold, d = solve(case["scramble"], json.loads(dump.read_text()), cfg, moves)
        results[case["video"]] = {"moves": d.moves, "rotations": 0, "hold": hold, "solved": d.solved, "agreement": round(d.agreement, 4), "gaps": d.gaps}
        print(f"{case['video']}: {len(d.moves)} moves, solved {d.solved}{f', gap at {d.gaps}' if d.gaps else ''}, "
              f"agreement {d.agreement:.1%}, hold {hold!r}, {time.time() - t0:.1f} s", flush=True)
    args.out.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
