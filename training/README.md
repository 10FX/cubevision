# Training the models

How the two models cubevision runs are made: images rendered with Blender, a detector and a reader trained
on them with PyTorch, exported to ONNX. This is ChronoCube's training code, as it stood when the released
models were made, without its data and without the runs that used real footage.

Nothing here is needed to run cubevision: the trained models are in the
[releases](https://github.com/10FX/cubevision/releases). This folder is for reading how they were made, and
for training them again.

## What is here

| Path | What it does |
|---|---|
| `cubevision/render/` | scenes drawn at random (cube, hands, light, room, camera) and rendered with Blender as a Python module; `solve_video.py` renders whole solves with their truth |
| `cubevision/data.py`, `augment.py`, `labels.py` | the renders as training examples, and what a phone or a webcam does to a picture |
| `cubevision/model.py`, `losses.py`, `train.py`, `evaluate.py` | the reader: which faces a crop shows, their stickers, whether a layer is mid-turn, the cube's orientation |
| `cubevision/detector.py`, `train_detector.py` | the detector: SSDLite320-MobileNetV3 from torchvision's COCO weights, a new two-class head |
| `cubevision/export.py`, `export_detector.py` | ONNX files in fp16 and fp32 with the JSON that describes them, checked against the PyTorch model |
| `cubevision/cube.py`, `decode.py`, `geometry.py` | the cube and the decoder in Python, for the evaluation |
| `configs/` | one file per training run; `v7.yaml` is the released reader, `detector.yaml` the released detector |
| `tools/` | diagnostics: recall by cube size, where frames are lost, false alarms on photos without a cube |
| `tests/` | unit tests of all of the above; they need PyTorch but neither Blender nor a GPU |

## Running it

Python 3.13 and [uv](https://docs.astral.sh/uv/); a GPU for anything but the tests. The versions are pinned
in `uv.lock`; PyTorch comes from the CUDA 12.8 index (needed by RTX 50xx cards).

```sh
uv sync --extra train --extra dev                # the tests (they need PyTorch, not Blender) and the linter
uv run pytest -q && uv run ruff check .

uv sync --extra render --extra train --extra dev   # everything (Blender as a module too)
make smoke        # the whole chain on a few hundred images, about 15 minutes: it checks that every step runs
make data-v7 train-v7 evaluate RUN=reader-v7 export FULL=runs/reader-v7/best.pt
make data-detect train-detect evaluate-detect export-detect
make synthetic    # 40 solves rendered as filmed, with their truth, to measure cubevision against
```

The head of the `Makefile` lists the targets. The released reader was trained on 646,053 renders
(`configs/v7.yaml` lists the folders, one of them twice), on an RTX 5060 Ti.

## What is not here, and why

- **No data and no weights.** Renders, runs and checkpoints are large and can be made again.
- **No real footage, and no run that used any.** The owner's own videos and the frames harvested from them
  are private. Three later configurations fine-tune the reader on them; they are not in `configs/`, and the
  web app's current reader, which comes from them, is not published. The released models never saw a real
  image of a cube.
- **Backgrounds.** Renders use photos as backdrops (`BACKGROUNDS`, by default `data/backgrounds`): the
  released models used COCO val2017, whose photos carry Flickr licences of several kinds and are not
  redistributed here. Download them from [cocodataset.org](https://cocodataset.org/#download), or point
  `BACKGROUNDS` at photos of your own; without any, the scenes get plain backdrops.
- **The hand.** The hands that hold the cube come from Blender Studio's Human Base Meshes (CC0). Download the
  bundle and build the asset once: `uv run python -m cubevision.render.hand_mesh <bundle.blend>`.

Comments in the code refer to ChronoCube's sources (`src/lib/vision/…`): the Rust in `../crates` is the port
of those files.
