"""Outputs of both fp32 models on a fixed input, from ONNX Runtime's CPU: what the Rust side must match."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODELS = Path(os.environ.get("CUBE_MODELS", "/home/thomas/TIMER/ml2/runs/reader-v7/web"))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "tests" / "reference.json")


def fixed_input(size: int) -> np.ndarray:
    c, y, i = np.meshgrid(np.arange(3), np.arange(size), np.arange(size), indexing="ij")
    return (((i + y + c) % size) / (size - 1)).astype(np.float32)[None]


def run(name: str, size: int) -> dict[str, list[float]]:
    session = ort.InferenceSession(str(MODELS / name), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"image": fixed_input(size)})
    return {meta.name: np.asarray(value, dtype=np.float32).ravel().tolist() for meta, value in zip(session.get_outputs(), outputs)}


reference = {"onnxruntime": ort.__version__, **run("cube-detector.fp32.onnx", 320), **run("cube-reader.fp32.onnx", 224)}
OUT.write_text(json.dumps(reference) + "\n")
print({key: len(value) if isinstance(value, list) else value for key, value in reference.items()})
