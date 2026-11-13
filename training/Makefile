# The cube detector and reader, from synthetic images to the ONNX files cubevision loads.
# Run from training/ (see README.md). Every step can be run on its own.
#
#   make setup            # Python env (uv), once
#   make check            # unit tests + lint
#   make smoke            # whole pipeline on a few hundred images (~15 min)
#   make data             # reader crops: train / val / test renders (hours: see README)
#   make data-detect      # whole frames for the detector
#   make train            # full reader at 192 px (`make train-256`: at 256 px, `make train-lite`: small)
#   make train-detect     # the detector (COCO-pretrained SSDLite, fine-tuned)
#   make evaluate         # reader: test report + temperature calibration on val
#   make evaluate-detect  # detector: test report
#   make export           # reader ONNX into models/ (FULL=… LITE=… to choose)
#   make export-detect    # detector ONNX into models/
#   make synthetic        # 40 synthetic solve videos as filmed in the wild, the truth known (evals/synthetic/)

PY        ?= uv run python
WORKERS   ?= 3
DEVICE    ?= auto
TRAIN     ?= 250000
VAL       ?= 10000
TEST      ?= 10000
DETECT_TRAIN ?= 27000
DETECT_VAL   ?= 1500
# Optional folder of photos used as backdrops (rooms, desks, walls)
BACKGROUNDS ?= data/backgrounds

BG := $(if $(BACKGROUNDS),--backgrounds $(BACKGROUNDS),)
GEN := $(PY) -m cubevision.render.generate --device $(DEVICE) --workers $(WORKERS) $(BG)
# Footage as it comes: every kind of scene, hand and light, the cube tilted and moving, faces read almost edge-on
WILD := --distractors --real-hands --variety --mesh-hands 0.8 --wild

.PHONY: setup check smoke data data-loose data-v7 data-detect preview train train-wild train-wild-more train-wild-v6 train-v7 data-more train-256 train-lite train-detect evaluate evaluate-detect export export-detect synthetic

setup:
	uv sync --extra render --extra train --extra dev
	$(PY) -c "import torch, bpy; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'); print('bpy', bpy.app.version_string)"

check:
	uv run pytest -q
	uv run ruff check .

# Seeds: 1 train, 2 val, 3 test (never shared: train.py refuses a val folder rendered with a train seed).
# Crops at 256 px: the 192 px reader shrinks them, the 256 px one reads them as they are
data:
	$(GEN) --out data/synth/val --seed 2 --count $(VAL) --resolution 256 --samples 16 $(WILD)
	$(GEN) --out data/synth/test --seed 3 --count $(TEST) --resolution 256 --samples 16 $(WILD)
	$(GEN) --out data/synth/train --seed 1 --count $(TRAIN) --resolution 256 --samples 16 $(WILD)

# 150k more training renders, seed 4 (make data-more), for the reader v6
data-more:
	$(GEN) --out data/synth/train2 --seed 4 --count 150000 --resolution 256 --samples 16 $(WILD)

# Real cubes as solved (--loose): layers a few degrees off, faster turns, smeared rotations, half-hidden stickers read
data-loose:
	$(GEN) --out data/synth/val-loose --seed 6 --count 4000 --resolution 256 --samples 16 $(WILD) --loose
	$(GEN) --out data/synth/train3 --seed 5 --count 120000 --resolution 256 --samples 16 $(WILD) --loose

# The reader v7's renders: the states of real solves (--stages) on real cubes (--loose)
data-v7:
	$(GEN) --out data/synth/val-v7 --seed 8 --count 6000 --resolution 256 --samples 16 $(WILD) --loose --stages --harsh
	$(GEN) --out data/synth/train-v7 --seed 7 --count 120000 --resolution 256 --samples 16 $(WILD) --loose --stages --harsh

data-detect:
	$(GEN) --out data/detect/train --framing full --resolution 416 --seed 11 --count $(DETECT_TRAIN) --samples 16 $(WILD)
	$(GEN) --out data/detect/val --framing full --resolution 416 --seed 12 --count $(DETECT_VAL) --samples 16 $(WILD)
	$(GEN) --out data/detect/test --framing full --resolution 416 --seed 13 --count $(DETECT_VAL) --samples 16 $(WILD)

preview:
	$(PY) -m cubevision.render.preview data/synth/train --count 48 --columns 8 --out runs/preview-train.png

# A 2-epoch model is not accurate: the smoke run checks that every step runs, not the numbers
smoke:
	$(GEN) --out data/smoke/train --seed 101 --count 512
	$(GEN) --out data/smoke/val --seed 102 --count 128
	$(PY) -m cubevision.render.preview data/smoke/train --count 32 --columns 8 --out runs/preview-smoke.png
	$(PY) -m cubevision.train --config configs/full.yaml --set train.name=smoke train.epochs=2 train.batch_size=32 \
		"data.train=[data/smoke/train]" "data.val=[data/smoke/val]" data.workers=4
	$(PY) -m cubevision.evaluate --checkpoint runs/smoke/best.pt --data data/smoke/val --calibrate
	$(PY) -m cubevision.export --checkpoint runs/smoke/best.pt --data data/smoke/val --out runs/smoke/models --min-agreement 0
	$(GEN) --out data/smoke/detect-train --framing full --resolution 416 --seed 103 --count 256
	$(GEN) --out data/smoke/detect-val --framing full --resolution 416 --seed 104 --count 64
	$(PY) -m cubevision.train_detector --config configs/detector.yaml --set train.name=smoke-detector train.epochs=2 train.batch_size=16 \
		"data.train=[data/smoke/detect-train]" "data.val=[data/smoke/detect-val]" data.workers=4
	$(PY) -m cubevision.export_detector --checkpoint runs/smoke-detector/best.pt --data data/smoke/detect-val --out runs/smoke/models --min-agreement 0

train:
	$(PY) -m cubevision.train --config configs/full.yaml

train-256:
	$(PY) -m cubevision.train --config configs/full-256.yaml

train-lite:
	$(PY) -m cubevision.train --config configs/lite.yaml

# The reader v4 on the wild renders, from v3's weights
train-wild:
	$(PY) -m cubevision.train --config configs/wild.yaml --init runs/reader-v3/best.pt

# The reader v5: v4 trained on at a lower rate
train-wild-more:
	$(PY) -m cubevision.train --config configs/wild-more.yaml --init runs/reader-v4/best.pt

# The reader v6: v5 trained on, with the 150k more renders
train-wild-v6:
	$(PY) -m cubevision.train --config configs/wild-v6.yaml --init runs/reader-v5/best.pt

# The reader v7: MobileNetV4-medium from ImageNet weights, on the relabelled renders and the solve stages (data-v7)
train-v7:
	$(PY) -m cubevision.train --config configs/v7.yaml

train-detect:
	$(PY) -m cubevision.train_detector --config configs/detector.yaml

# RUN=full-256 to evaluate another run
RUN ?= full
evaluate:
	$(PY) -m cubevision.evaluate --checkpoint runs/$(RUN)/best.pt --data data/synth/val --calibrate
	$(PY) -m cubevision.evaluate --checkpoint runs/$(RUN)/best.pt --data data/synth/test --out runs/$(RUN)/test-report.json

# The score threshold is chosen on val (precision >= 0.98), then the test report uses it, and so does the export
evaluate-detect:
	$(PY) -m cubevision.train_detector --evaluate runs/detector/best.pt --data data/detect/val --calibrate
	$(PY) -m cubevision.train_detector --evaluate runs/detector/best.pt --data data/detect/test --out runs/detector/test-report.json

# FULL: the reader exported in fp16 for GPUs (e.g. runs/reader-v7/best.pt); LITE: the one for processors (default: from FULL)
FULL ?= runs/full/best.pt
LITE ?=
export:
	$(PY) -m cubevision.export --checkpoint $(FULL) $(if $(LITE),--lite-checkpoint $(LITE),) --data data/synth/val --out models

export-detect:
	$(PY) -m cubevision.export_detector --checkpoint runs/detector/best.pt --data data/detect/val --out models

# Synthetic solves: the whole system measured against the truth (cubevision video … on each, see ../README.md). As filmed: the cube
# agitated, held anew, every hand, room and light, 480p to 1080p, 30 or 60 fps, VP8, VP9 or H.264 squeezed hard,
# a fifth filmed from above (the PLL's top uniform: its U turns seen on the side rows only)
SYNTHETIC ?= evals/synthetic
synthetic:
	$(PY) -m cubevision.render.solve_video --out $(SYNTHETIC) --count 40 --seed 23 --workers 2 $(BG) --wild --real-hands --variety \
		--distractors --camera-look --corner-cutting 0.3 --tps 3 10 --codecs vp8 vp9 h264 --bitrates 800k 1500k 2500k 5000k --overhead 0.2

