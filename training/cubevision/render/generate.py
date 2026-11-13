"""
Render a synthetic dataset.

    # Blender as a Python module (pip install bpy), from ml2/
    python -m cubevision.render.generate --out data/synth/train --seed 1 --count 100000 --workers 3
    # Whole frames for the detector
    python -m cubevision.render.generate --out data/detect/train --framing full --resolution 416 --seed 11 --count 60000

    # Or with the Blender program
    blender -b --factory-startup --python cubevision/render/generate.py -- --out data/synth/train --count 1000

Writes ``images/0000000.jpg …``, one ``labels-<start>.jsonl`` per worker
(cubevision/labels.py) and ``dataset.json`` (seed, resolution, samples, code
commit). Interrupted runs resume where they stopped; a folder is never filled
with two different settings. Image i of seed s is always the same scenario,
whatever the number of workers.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _args(argv: list[str] | None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=1, help="dataset seed: use different seeds for train / val / test")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=1000)
    p.add_argument("--framing", choices=["crop", "full"], default="crop",
                   help="crop: the square crop the reader reads; full: whole frames for the detector")
    p.add_argument("--resolution", type=int, default=256, help="side of the crops, or longer side of the full frames")
    p.add_argument("--samples", type=int, default=32, help="Cycles samples per pixel (denoised)")
    p.add_argument("--device", default="auto", help="auto | optix | cuda | cpu")
    p.add_argument("--backgrounds", type=Path, help="folder of photos used as backdrops (jpg/png)")
    p.add_argument("--workers", type=int, default=1, help="processes sharing the GPU (2-3 keep it busy)")
    p.add_argument("--recycle", type=int, default=3000,
                   help="a worker hands over to a fresh one after this many images: Blender slows down as it runs")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--distractors", action="store_true", help="look-alikes and things of a room around the cube, more frames without it")
    p.add_argument("--real-hands", action="store_true", help="hands as filmed: round phalanges, knuckles, nails, fingertips near the edges")
    p.add_argument("--variety", action="store_true", help="faded and worn cubes, backlight, dim rooms, hard spot lights")
    p.add_argument("--wild", action="store_true", help="footage as it comes: tilted and moving cubes, whole-cube rotations, glare, "
                   "glossy, printed and fluo stickers, faces read almost edge-on")
    p.add_argument("--loose", action="store_true", help="real cubes as solved: layers a few degrees off (settled), faster turns "
                   "and smeared whole-cube rotations")
    p.add_argument("--stages", action="store_true", help="the states of real solves: scrambled, the cross, F2L pairs, OLL, PLL, "
                   "Roux, partway through the site's algorithms, the step's colour on any face")
    p.add_argument("--harsh", action="store_true", help="the worst footage: smeared glances and shakes, cameras out of focus, "
                   "burnt, oversaturated or dark colours")
    p.add_argument("--mesh-hands", type=float, default=0.0,
                   help="share of the hands drawn as the sculpted hand (hand_mesh; its asset built), the others as before")
    return p.parse_args(argv)


def _manifest(args: argparse.Namespace) -> None:
    """dataset.json: what the folder holds. Refuses to add images rendered with other settings."""
    settings = {"seed": args.seed, "framing": args.framing, "resolution": args.resolution, "samples": args.samples,
                **({"distractors": True} if args.distractors else {}), **({"real_hands": True} if args.real_hands else {}),
                **({"variety": True} if args.variety else {}), **({"mesh_hands": args.mesh_hands} if args.mesh_hands else {}),
                **({"wild": True} if args.wild else {}), **({"loose": True} if args.loose else {}),
                **({"stages": True} if args.stages else {}), **({"harsh": True} if args.harsh else {})}
    path = args.out / "dataset.json"
    if path.exists():
        existing = json.loads(path.read_text())
        different = {k: (existing.get(k), v) for k, v in settings.items() if existing.get(k) != v}
        if different:
            raise SystemExit(f"{args.out} holds another dataset ({different}): use another --out")
        return
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL, cwd=Path(__file__).parent).strip()
    except Exception:
        commit = "unknown"
    args.out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**settings, "commit": commit, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))


def done_indices(out: Path) -> set[int]:
    """Images already rendered in a folder, by any worker: a run resumed with another --count or
    --workers splits the range differently, and must not render (and label) an image twice."""
    done: set[int] = set()
    for path in out.glob("labels-*.jsonl"):
        with path.open() as f:
            done |= {json.loads(line)["meta"]["index"] for line in f if line.strip()}
    return done


# Exit status of a worker that stopped after --recycle images, its range not done
RECYCLED = 75


def _spawn(args: argparse.Namespace) -> int:
    """Split the range between worker processes and wait for them, each replaced by a fresh one when
    it has rendered --recycle images (a long-lived Blender grows and slows: 26k then 7k images an hour)."""
    per = -(-args.count // args.workers)
    cmds = []
    for w in range(args.workers):
        start = args.start + w * per
        count = min(per, args.start + args.count - start)
        if count <= 0:
            break
        cmd = [sys.executable, "-m", "cubevision.render.generate", "--out", str(args.out), "--seed", str(args.seed),
               "--start", str(start), "--count", str(count), "--framing", args.framing, "--resolution", str(args.resolution),
               "--samples", str(args.samples), "--device", args.device, "--recycle", str(args.recycle), "--child"]
        if args.backgrounds:
            cmd += ["--backgrounds", str(args.backgrounds)]
        if args.distractors:
            cmd += ["--distractors"]
        if args.real_hands:
            cmd += ["--real-hands"]
        if args.variety:
            cmd += ["--variety"]
        if args.mesh_hands:
            cmd += ["--mesh-hands", str(args.mesh_hands)]
        if args.wild:
            cmd += ["--wild"]
        if args.loose:
            cmd += ["--loose"]
        if args.stages:
            cmd += ["--stages"]
        if args.harsh:
            cmd += ["--harsh"]
        cmds.append(cmd)
    cwd = Path(__file__).resolve().parents[2]
    procs = [subprocess.Popen(cmd, cwd=cwd) for cmd in cmds]
    codes = [None] * len(procs)
    while None in codes:
        for i, proc in enumerate(procs):
            if codes[i] is not None or proc.poll() is None:
                continue
            if proc.returncode == RECYCLED:
                procs[i] = subprocess.Popen(cmds[i], cwd=cwd)
            else:
                codes[i] = proc.returncode
        time.sleep(1)
    return max(codes)


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    _manifest(args)
    if not args.child:
        return _spawn(args)

    import dataclasses

    import numpy as np

    from cubevision.labels import ImageLabel
    from cubevision.render import hand_mesh
    from cubevision.render.annotate import annotate
    from cubevision.render.sampler import sample
    from cubevision.render.scene import CubeScene

    backgrounds = sorted(p for p in args.backgrounds.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")) if args.backgrounds else []
    images = args.out / "images"
    images.mkdir(parents=True, exist_ok=True)
    labels_path = args.out / f"labels-{args.start:07d}.jsonl"
    done = done_indices(args.out) & set(range(args.start, args.start + args.count))

    scene = CubeScene(args.resolution, args.samples, args.device, backgrounds)
    print(f"[generate] seed {args.seed}, images {args.start}..{args.start + args.count - 1}, device {scene.device}, {len(done)} already done", flush=True)
    t0, rendered = time.time(), 0
    with labels_path.open("a") as out:
        for index in range(args.start, args.start + args.count):
            if index in done:
                continue
            sc = sample(args.seed, index, backgrounds=len(backgrounds), framing=args.framing, distractors=args.distractors,
                        real_hands=args.real_hands, variety=args.variety, wild=args.wild, loose=args.loose, stages=args.stages,
                        harsh=args.harsh)
            if args.mesh_hands and sc.hands and hand_mesh.ASSET.exists() and np.random.default_rng([args.seed, index, 23]).random() < args.mesh_hands:
                sc = dataclasses.replace(sc, hands=[dataclasses.replace(h, model="mesh") for h in sc.hands])
            name = f"{index:07d}.jpg"
            r = scene.render(sc, images / name)
            faces, turning = annotate(sc.stickers, sc.turn, r.visible, r.facing, r.area, args.wild)
            meta = {
                "index": index, "seed": args.seed, "recipe": sc.recipe, "style": sc.style,
                "turn": sc.turn.__dict__ if sc.turn else None, "hands": len(sc.hands),
                "azimuth": round(sc.azimuth, 1), "elevation": round(sc.elevation, 1), "background": sc.background["kind"],
                "centres": "plain" if not sc.caps else "+".join(k for k, on in (("caps", sc.caps["all"]), ("logo", sc.caps["logo"] is not None)) if on),
                "framing": sc.framing, "aspect": round(sc.aspect, 3), "body": sc.body is not None, "cube": not sc.cube_hidden,
                "lighting": sc.meta.get("lighting", "white"), "distractors": len(sc.distractors),
                "variety": sc.meta.get("variety", "none"), "light": sc.meta.get("light", "normal"),
                "hand_model": sc.hands[0].model if sc.hands else None, "wild": sc.meta.get("wild"),
                **({"loose": sc.meta["loose"]} if "loose" in sc.meta else {}),
                **({"harsh": sc.meta["harsh"]} if "harsh" in sc.meta else {}),
            }
            box = [round(v, 5) for v in r.box] if r.box else None
            out.write(ImageLabel(f"images/{name}", faces, turning, meta, box, r.orientation).to_json() + "\n")
            out.flush()
            rendered += 1
            if rendered >= args.recycle and index < args.start + args.count - 1:
                print(f"[generate] {rendered} images: a fresh worker takes over", flush=True)
                return RECYCLED
            if rendered % 50 == 0:
                rate = rendered / (time.time() - t0)
                left = (args.count - len(done) - rendered) / rate
                print(f"[generate] {rendered} images, {rate:.1f}/s, ~{left / 60:.0f} min left", flush=True)
    print(f"[generate] done: {rendered} images in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    # Run as a script (by the Blender program): make the package importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
