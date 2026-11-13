"""
Label a rendered dataset again with today's rule of what is visible, without drawing it again: each
scenario is set up as when it was rendered (same seed, same settings: dataset.json) and only measured.

    python -m cubevision.render.relabel --data data/synth/train --workers 6

The scenes must be the ones drawn: each image's stickers measured by the rule it was labelled with
(3 of the 5 inner points of a tile, 2 for --loose, 3 stickers a face) must give back its labels, else the run stops (--tolerance). At the
end the old labels move to orig-labels/ and the new ones take their place.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

#: Exit status of a worker that stopped after --recycle scenes, its share not done (as generate.py)
RECYCLED = 75


def _args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--backgrounds", type=Path, default=Path("data/backgrounds"))
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--recycle", type=int, default=3000)
    p.add_argument("--limit", type=int, default=0, help="only this many images per worker (a check), labels left as they are")
    p.add_argument("--tolerance", type=float, default=0.02, help="share of the faces measured by the old rule allowed to differ")
    p.add_argument("--worker", type=int, default=-1, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def _labels(data: Path):
    for path in sorted(data.glob("labels-*.jsonl")):
        with path.open() as f:
            for line in f:
                if line.strip():
                    yield line


def _part(args: argparse.Namespace) -> Path:
    return args.data / "relabel" / f"part-{args.worker:02d}.jsonl"


def _work(args: argparse.Namespace) -> int:
    import dataclasses

    import numpy as np

    from cubevision.labels import FaceLabel, ImageLabel
    from cubevision.render import hand_mesh
    from cubevision.render.annotate import annotate
    from cubevision.render.sampler import sample
    from cubevision.render.scene import CubeScene

    settings = json.loads((args.data / "dataset.json").read_text())
    backgrounds = sorted(p for p in args.backgrounds.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    part = _part(args)
    part.parent.mkdir(exist_ok=True)
    done = set()
    if part.exists():
        with part.open() as f:
            done = {json.loads(line)["meta"]["index"] for line in f if line.strip()}
    scene = CubeScene(settings["resolution"], 1, "cpu", backgrounds)
    wild = bool(settings.get("wild"))
    old_hits = 2 if settings.get("loose") else 3  # loose datasets (before relabel.py) counted half-hidden tiles
    t0, count, same, faces_seen, read_before, read_now = time.time(), 0, 0, 0, 0, 0
    with part.open("a") as out:
        for line in _labels(args.data):
            label = ImageLabel.from_json(line)
            index = label.meta["index"]
            if index % args.workers != args.worker or index in done:
                continue
            sc = sample(settings["seed"], index, backgrounds=len(backgrounds), framing=settings["framing"],
                        distractors=bool(settings.get("distractors")), real_hands=bool(settings.get("real_hands")),
                        variety=bool(settings.get("variety")), wild=wild, loose=bool(settings.get("loose")),
                        stages=bool(settings.get("stages")), harsh=bool(settings.get("harsh")))
            mesh = settings.get("mesh_hands", 0.0)
            if mesh and sc.hands and hand_mesh.ASSET.exists() and np.random.default_rng([settings["seed"], index, 23]).random() < mesh:
                sc = dataclasses.replace(sc, hands=[dataclasses.replace(h, model="mesh") for h in sc.hands])
            r = scene.render(sc, None)
            if r.inner is None:  # no cube drawn: nothing to read, before or now
                faces = label.faces
                same += 6
            else:
                before, _ = annotate(sc.stickers, sc.turn, [n >= old_hits for n in r.inner], r.facing, r.area, wild, 3 if wild else None)
                faces, _ = annotate(sc.stickers, sc.turn, r.visible, r.facing, r.area, wild)
                same += sum(a == b for a, b in zip(before, label.faces, strict=True))
                read_before += sum(s != 6 for f in label.faces if f.present for s in f.stickers)
                read_now += sum(s != 6 for f in faces if f.present for s in f.stickers)
            faces_seen += 6
            if not args.limit:
                out.write(dataclasses.replace(label, faces=[FaceLabel(f.present, list(f.stickers)) for f in faces]).to_json() + "\n")
                out.flush()
            count += 1
            if count % 500 == 0 or count == args.limit:
                print(f"[relabel {args.worker}] {count} scenes, {count / (time.time() - t0):.1f}/s, faces as labelled "
                      f"{same / faces_seen:.4f}, stickers read {read_before} -> {read_now}", flush=True)
            if 1 - same / faces_seen > args.tolerance and count >= 100:
                print(f"[relabel {args.worker}] the scenes differ from the ones drawn ({same / faces_seen:.3f} of the faces as labelled)", flush=True)
                return 2
            if count == args.limit:
                return 0
            if count >= args.recycle:
                return RECYCLED
    print(f"[relabel {args.worker}] done: {count} scenes, faces as labelled {same / max(faces_seen, 1):.4f}, "
          f"stickers read {read_before} -> {read_now}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    if args.worker >= 0:
        return _work(args)
    cmds = [[sys.executable, "-m", "cubevision.render.relabel", "--data", str(args.data), "--backgrounds", str(args.backgrounds),
             "--workers", str(args.workers), "--recycle", str(args.recycle), "--limit", str(args.limit),
             "--tolerance", str(args.tolerance), "--worker", str(k)] for k in range(args.workers)]
    cwd = Path(__file__).resolve().parents[2]
    procs = [subprocess.Popen(cmd, cwd=cwd) for cmd in cmds]
    codes: list[int | None] = [None] * len(procs)
    while None in codes:
        for i, proc in enumerate(procs):
            if codes[i] is not None or proc.poll() is None:
                continue
            if proc.returncode == RECYCLED:
                procs[i] = subprocess.Popen(cmds[i], cwd=cwd)
            else:
                codes[i] = proc.returncode
                if proc.returncode:
                    for other in procs:
                        other.terminate()
        time.sleep(1)
    if any(codes) or args.limit:
        return max(c or 0 for c in codes)
    total = sum(1 for _ in _labels(args.data))
    parts = sorted((args.data / "relabel").glob("part-*.jsonl"))
    relabelled = sum(1 for p in parts for line in p.open() if line.strip())
    if relabelled != total:
        print(f"[relabel] {relabelled} of {total} labels: run it again to finish", flush=True)
        return 1
    orig = args.data / "orig-labels"
    orig.mkdir(exist_ok=True)
    for path in args.data.glob("labels-*.jsonl"):
        path.rename(orig / path.name)
    for p in parts:
        p.rename(args.data / f"labels-relabel-{p.stem.split('-')[1]}.jsonl")
    print(f"[relabel] {total} labels replaced (the old ones in {orig})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
