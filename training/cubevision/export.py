"""
Export checkpoints for the browser (onnxruntime-web).

    python -m cubevision.export --checkpoint runs/full/best.pt --data data/synth/val --out ../public/models
    python -m cubevision.export --checkpoint runs/full/best.pt --lite-checkpoint runs/lite/best.pt --data data/synth/val

Writes, for ml2/spec.json:

* ``cube-reader.fp32.onnx``: reference, kept in each run folder (checked against PyTorch);
* ``cube-reader.fp16.onnx``: the "full" reader for WebGPU (``--checkpoint``,
  half the size, same answers);
* ``cube-reader.int8.onnx``: the "lite" reader for the WASM fallback
  (``--lite-checkpoint``, by default the same model; static QDQ quantization
  calibrated on renders, a quarter of the size);
* ``cube-reader.json``: metadata read by the app (for each variant its file
  and input size, the temperature, validation metrics and agreement with fp32).

Each variant must agree with its fp32 model on the validation images (argmax
of every sticker and presence) or the export fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .data import SyntheticDataset
from .evaluate import load_model
from .model import ExportWrapper

OUTPUTS = ["present", "stickers", "turning", "orientation"]


def _images(data: list[Path], size: int, count: int) -> np.ndarray:
    ds = SyntheticDataset(data, size, train=False)
    step = max(1, len(ds) // count)
    return np.stack([ds[i][0].numpy() for i in range(0, len(ds), step)][:count]).astype(np.float32) / 255.0


def _run(path: Path, images: np.ndarray) -> list[np.ndarray]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = [session.run(OUTPUTS, {"image": images[i : i + 32]}) for i in range(0, len(images), 32)]
    return [np.concatenate([o[k] for o in outs]) for k in range(len(OUTPUTS))]


def _agreement(ref: list[np.ndarray], other: list[np.ndarray]) -> dict[str, float]:
    # Angle between the two orientations of each image, degrees
    cos = (np.einsum("nji,nji->n", ref[3], other[3]) - 1) / 2
    angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return {
        "present": float(((ref[0] > 0.5) == (other[0] > 0.5)).mean()),
        "stickers": float((ref[1].argmax(-1) == other[1].argmax(-1)).mean()),
        "turning": float(((ref[2] > 0.5) == (other[2] > 0.5)).mean()),
        "orientation_max_deg": float(angle.max()),
        "max_abs_diff": float(max(np.abs(a - b).max() for a, b in zip(ref, other, strict=True))),
    }


class _Calibration:
    """Calibration reader for static quantization."""

    def __init__(self, images: np.ndarray):
        self.batches = iter([{"image": images[i : i + 1]} for i in range(len(images))])

    def get_next(self):
        return next(self.batches, None)


def metadata_json(meta: dict) -> str:
    """The metadata the app fetches: undefined metrics (NaN) become null, as JSON.parse needs."""
    def finite(x):
        if isinstance(x, float) and not np.isfinite(x):
            return None
        if isinstance(x, dict):
            return {k: finite(v) for k, v in x.items()}
        if isinstance(x, list | tuple):
            return [finite(v) for v in x]
        return x

    return json.dumps(finite(meta), indent=2, allow_nan=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_onnx(checkpoint: Path, name: str, opset: int, images_from: list[Path], count: int) -> dict:
    """Checkpoint → fp32 ONNX in its run folder, checked against PyTorch."""
    model, cfg = load_model(checkpoint, torch.device("cpu"))
    calib_path = checkpoint.parent / "calibration.json"
    temperature = json.loads(calib_path.read_text())["temperature"] if calib_path.exists() else 1.0
    wrapper = ExportWrapper(model, temperature).eval()
    size = cfg.model.image_size
    fp32 = checkpoint.parent / f"{name}.fp32.onnx"

    # Plain attention kernels: the fused fast path is not exportable
    torch.backends.mha.set_fastpath_enabled(False)
    dummy = torch.rand(1, 3, size, size)
    export_kwargs = dict(input_names=["image"], output_names=OUTPUTS, opset_version=opset,
                         dynamic_axes={"image": {0: "batch"}, **{o: {0: "batch"} for o in OUTPUTS}})
    try:
        torch.onnx.export(wrapper, (dummy,), str(fp32), dynamo=False, **export_kwargs)
    except Exception as error:  # a PyTorch without the TorchScript exporter: use the dynamo one
        print(f"[export] TorchScript exporter unavailable ({error}), using dynamo")
        torch.onnx.export(wrapper, (dummy,), str(fp32), dynamo=True, **export_kwargs)
    onnx.checker.check_model(onnx.load(str(fp32)))

    images = _images(images_from, size, count)
    with torch.no_grad():
        ref_torch = [t.numpy() for t in wrapper(torch.from_numpy(images[:64]))]
    ref = _run(fp32, images)
    parity = _agreement(ref_torch, [r[:64] for r in ref])
    print(f"[export] {checkpoint}: fp32 vs PyTorch {parity}")
    if parity["max_abs_diff"] > 1e-3:
        raise SystemExit("ONNX fp32 does not match PyTorch")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return {"fp32": fp32, "images": images, "ref": ref, "size": size, "backbone": cfg.model.backbone,
            "temperature": temperature, "validation": ckpt.get("val"), "checkpoint": str(checkpoint)}


def to_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    """The fp16 reader for WebGPU, inputs and outputs kept in float32."""
    from onnxconverter_common import float16

    # Casts to float of shape values (the attention's 1/sqrt(head size)) would keep producing float
    # where the converter declares float16: left in fp32, the converter casts their outputs. And GridSample:
    # onnxruntime-web's WebGPU shader for it in fp16 does not compile (f32 coefficients times f16 values,
    # 1.30, Chrome 155), the stickers' sampling then skipped without an error thrown
    to_float = lambda n: n.op_type == "Cast" and any(a.name == "to" and a.i == onnx.TensorProto.FLOAT for a in n.attribute)  # noqa: E731
    keep = [n.name for n in model.graph.node if n.op_type == "GridSample" or to_float(n)]
    return float16.convert_float_to_float16(model, keep_io_types=True, node_block_list=keep)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True, help="the full reader (fp16, WebGPU)")
    p.add_argument("--lite-checkpoint", type=Path, help="the lite reader (int8, WASM); default: --checkpoint")
    p.add_argument("--data", type=Path, nargs="+", required=True, help="validation images (parity checks and int8 calibration)")
    p.add_argument("--out", type=Path, default=Path("../public/models"))
    p.add_argument("--name", default="cube-reader")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--check-images", type=int, default=512)
    p.add_argument("--min-agreement", type=float, default=0.99)
    # Percentile calibration keeps every activation of every image in memory: 128 detector frames took
    # over 7 GB and the kernel killed the export; 32 stay near 2 GB
    p.add_argument("--calibration-images", type=int, default=32)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    full = _to_onnx(args.checkpoint, args.name, args.opset, args.data, args.check_images)
    lite_checkpoint = args.lite_checkpoint or args.checkpoint
    lite = full if lite_checkpoint == args.checkpoint else _to_onnx(lite_checkpoint, args.name, args.opset, args.data, args.check_images)

    fp16 = args.out / f"{args.name}.fp16.onnx"
    onnx.save(to_fp16(onnx.load(str(full["fp32"]))), str(fp16))

    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    int8 = args.out / f"{args.name}.int8.onnx"
    pre = args.out / f"{args.name}.pre.onnx"
    # The symbolic shape inference fails on the decoder's attention reshapes; ONNX's own shape
    # inference and onnxruntime's graph optimization (what quantization needs) still run
    quant_pre_process(str(lite["fp32"]), str(pre), skip_symbolic_shape=True)
    try:
        quantize_static(str(pre), str(int8), _Calibration(lite["images"][:args.calibration_images]), quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.Percentile)
    except ValueError as error:  # a constant tensor (the geometric reader's tables) breaks the histograms
        print(f"[export] int8 quantization failed ({error})")
    pre.unlink()

    variants = {}
    for kind, path, source in (("full", fp16, full), ("lite", int8, lite)):
        agreement = _agreement(source["ref"], _run(path, source["images"])) if path.exists() else None
        if kind == "lite" and (agreement is None or agreement["stickers"] < args.min_agreement or agreement["present"] < args.min_agreement):
            # As the detector's: rather a bigger WASM model than one that reads differently
            print(f"[export] {path.name} {'missing' if agreement is None else 'disagrees with fp32'}: the lite variant is the fp32 model")
            path.unlink(missing_ok=True)
            path = args.out / f"{args.name}.fp32.onnx"
            shutil.copyfile(source["fp32"], path)
            agreement = _agreement(source["ref"], _run(path, source["images"]))
        print(f"[export] {path.name}: {path.stat().st_size / 1e6:.1f} MB, agreement {agreement}")
        if agreement["stickers"] < args.min_agreement or agreement["present"] < args.min_agreement:
            raise SystemExit(f"{path.name} disagrees with fp32")
        variants[kind] = {
            "file": path.name, "size": source["size"], "backbone": source["backbone"], "checkpoint": source["checkpoint"],
            "temperature": source["temperature"], "validation": source["validation"],
            "bytes": path.stat().st_size, "sha256": _sha256(path), "agreement_with_fp32": agreement,
        }

    meta = {
        "name": args.name,
        "spec": json.loads((Path(__file__).resolve().parents[1] / "spec.json").read_text())["version"],
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": {"name": "image", "layout": "NCHW", "range": [0, 1]},
        "outputs": {"present": [6], "stickers": [6, 9, 7], "turning": [1], "orientation": [3, 3]},
        # Each face's stickers in StickerCube order as the centres seen turn the cube (model.by_colour)
        "cells": "cube",
        "variants": variants,
    }
    (args.out / f"{args.name}.json").write_text(metadata_json(meta))
    print(f"[export] wrote {args.out / (args.name + '.json')}")


if __name__ == "__main__":
    main()
