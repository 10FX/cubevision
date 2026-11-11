#!/bin/sh
# ONNX Runtime 1.31.1 for CUDA (with its CUDA 13 and cuDNN libraries) in ort/venv; ort/env points at it
set -eu
cd "$(dirname "$0")"
[ -d ort/venv ] || uv venv -q --python 3.13 ort/venv
uv pip install -q --python ort/venv/bin/python "onnxruntime-gpu[cuda,cudnn]==1.31.1"
site=$(ort/venv/bin/python -I -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
lib=$(ls "$site"/onnxruntime/capi/libonnxruntime.so.* | head -1)
libs=$(find "$site/nvidia" -maxdepth 3 -type d -name lib | sort | tr '\n' ':')
cat > ort/env <<ENV
export ORT_DYLIB_PATH="$lib"
export LD_LIBRARY_PATH="$site/onnxruntime/capi:${libs}/usr/lib/wsl/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
ENV
