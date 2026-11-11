#!/bin/sh
# The spike built and run on Windows from WSL, sources copied to C:\Users\Thomas\cubevision-spike:
#   ./run-windows.sh infer --device cuda --seconds 20 > results/infer-windows-cuda.json
#   ./run-windows.sh test <filter>
# DirectML runs on ONNX Runtime 1.24.4 (ort\dml), OpenVINO on 1.24.1 (ort\ov), everything else on 1.31.1 (ort\cuda).
set -eu
cd "$(dirname "$0")"
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
WIN='C:\Users\Thomas\cubevision-spike'
src=$(wslpath -w "$PWD")
models=$(wslpath -w "${CUBE_MODELS:-/home/thomas/TIMER/ml2/runs/reader-v7/web}")
lib=cuda
case " $* " in *" directml "*) lib=dml ;; *openvino*) lib=ov ;; esac
if [ "${1:-}" = test ]; then shift; run="cargo test --release -- --include-ignored $*"; else run="cargo run --release -q -- $*"; fi
"$PS" -NoProfile -ExecutionPolicy Bypass -Command "
robocopy '$src' '$WIN' /E /XD target ort results /NFL /NDL /NJH /NJS /NP | Out-Null
robocopy '$models' '$WIN\\models' *.onnx /NFL /NDL /NJH /NJS /NP | Out-Null
Set-Location '$WIN'
if (-not (Test-Path ort\\ov.ps1)) { & .\\setup-ort.ps1 | Out-Null }
. .\\ort\\$lib.ps1
\$env:CUBE_MODELS = '$WIN\\models'
$run
exit \$LASTEXITCODE
" | tr -d '\r'
