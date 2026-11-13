"""
Export the cube detector for the browser (onnxruntime-web), next to the reader.

    python -m cubevision.export_detector --checkpoint runs/detector/best.pt --data data/detect/val --out ../public/models

Writes ``cube-detector.fp32.onnx`` (reference, in the run folder),
``cube-detector.fp16.onnx`` (WebGPU), ``cube-detector.int8.onnx`` (WASM) and
``cube-detector.json`` (read by src/lib/vision/detector.ts). Each variant must
find the same cube as fp32 (same decision, box IoU ≥ 0.9) on 99 % of the
validation frames, or the export fails.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from .detector import CROP_MARGIN, PAD, DetectionDataset, ExportWrapper, box_iou
from .export import _Calibration, _sha256, metadata_json
from .train_detector import calibrated_threshold, load_detector

OUTPUTS = ["box", "score"]


def _images(data: list[Path], size: int, count: int) -> np.ndarray:
    ds = DetectionDataset(data, size, train=False)
    step = max(1, len(ds) // count)
    return np.stack([ds[i][0].numpy() for i in range(0, len(ds), step)][:count]).astype(np.float32) / 255.0


def _run(path: Path, images: np.ndarray) -> list[np.ndarray]:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outs = [session.run(OUTPUTS, {"image": images[i : i + 32]}) for i in range(0, len(images), 32)]
    return [np.concatenate([o[k] for o in outs]) for k in range(len(OUTPUTS))]


def _agreement(ref: list[np.ndarray], other: list[np.ndarray], threshold: float) -> dict[str, float]:
    found_ref, found = ref[1][:, 0] >= threshold, other[1][:, 0] >= threshold
    iou = box_iou(torch.from_numpy(ref[0]), torch.from_numpy(other[0])).numpy()
    same = (found_ref == found) & (~found_ref | (iou >= 0.9))
    return {"same": float(same.mean()), "iou_median": float(np.median(iou[found_ref])) if found_ref.any() else float("nan"),
            "score_max_abs_diff": float(np.abs(ref[1] - other[1]).max())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, nargs="+", required=True, help="validation frames (parity checks and int8 calibration)")
    p.add_argument("--out", type=Path, default=Path("../public/models"))
    p.add_argument("--name", default="cube-detector")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--check-images", type=int, default=512)
    p.add_argument("--min-agreement", type=float, default=0.99)
    # Percentile calibration keeps every activation of every image in memory: 128 detector frames took
    # over 7 GB and the kernel killed the export; 32 stay near 2 GB
    p.add_argument("--calibration-images", type=int, default=32)
    args = p.parse_args()

    model, cfg = load_detector(args.checkpoint, torch.device("cpu"))
    # Chosen on validation frames by train_detector --evaluate --calibrate, else 0.5
    threshold = calibrated_threshold(args.checkpoint)
    size = cfg.model.size
    wrapper = ExportWrapper(model, size).eval()
    fp32 = args.checkpoint.parent / f"{args.name}.fp32.onnx"
    dummy = torch.rand(1, 3, size, size)
    export_kwargs = dict(input_names=["image"], output_names=OUTPUTS, opset_version=args.opset,
                         dynamic_axes={"image": {0: "batch"}, **{o: {0: "batch"} for o in OUTPUTS}})
    try:
        torch.onnx.export(wrapper, (dummy,), str(fp32), dynamo=False, **export_kwargs)
    except Exception as error:  # a PyTorch without the TorchScript exporter: use the dynamo one
        print(f"[export] TorchScript exporter unavailable ({error}), using dynamo")
        torch.onnx.export(wrapper, (dummy,), str(fp32), dynamo=True, **export_kwargs)
    onnx.checker.check_model(onnx.load(str(fp32)))

    images = _images(args.data, size, args.check_images)
    with torch.no_grad():
        ref_torch = [t.numpy() for t in wrapper(torch.from_numpy(images[:64]))]
    ref = _run(fp32, images)
    parity = _agreement(ref_torch, [r[:64] for r in ref], threshold)
    print(f"[export] fp32 vs PyTorch: {parity}")
    if parity["same"] < 1.0 or parity["score_max_abs_diff"] > 1e-3:
        raise SystemExit("ONNX fp32 does not match PyTorch")

    args.out.mkdir(parents=True, exist_ok=True)
    from onnxconverter_common import float16

    fp16 = args.out / f"{args.name}.fp16.onnx"
    onnx.save(float16.convert_float_to_float16(onnx.load(str(fp32)), keep_io_types=True), str(fp16))

    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    int8 = args.out / f"{args.name}.int8.onnx"
    pre = args.out / f"{args.name}.pre.onnx"
    quant_pre_process(str(fp32), str(pre))
    # Convolutions of the backbone only: the heads (scores and box offsets) and the box decoding stay in
    # float (quantized heads moved scores by up to 0.48: 78 % of the frames kept the same answer)
    heads = [n.name for n in onnx.load(str(pre)).graph.node if "/head/" in n.name]
    quantize_static(str(pre), str(int8), _Calibration(images[:args.calibration_images]), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.Percentile,
                    op_types_to_quantize=["Conv"], nodes_to_exclude=heads)
    pre.unlink()

    variants = {}
    for kind, path in (("full", fp16), ("lite", int8)):
        agreement = _agreement(ref, _run(path, images), threshold)
        print(f"[export] {path.name}: {path.stat().st_size / 1e6:.1f} MB, agreement {agreement}")
        if agreement["same"] < args.min_agreement and kind == "lite":
            # MobileNetV3's backbone quantizes poorly (hard-swish, squeeze-excite): the WASM variant
            # stays in float rather than answer differently
            print(f"[export] {path.name} disagrees with fp32: the lite variant is the fp32 model")
            path.unlink()
            path = args.out / f"{args.name}.fp32.onnx"
            shutil.copyfile(fp32, path)
            agreement = _agreement(ref, _run(path, images), threshold)
        if agreement["same"] < args.min_agreement:
            raise SystemExit(f"{path.name} disagrees with fp32")
        variants[kind] = {"file": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path), "agreement_with_fp32": agreement}

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    meta = {
        "name": args.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": {"name": "image", "size": size, "layout": "NCHW", "range": [0, 1], "letterbox": PAD},
        "outputs": {"box": [4], "score": [1]},
        "threshold": threshold,
        "cropMargin": CROP_MARGIN,
        "validation": ckpt.get("val"),
        "variants": variants,
    }
    (args.out / f"{args.name}.json").write_text(metadata_json(meta))
    print(f"[export] wrote {args.out / (args.name + '.json')}")


if __name__ == "__main__":
    main()
