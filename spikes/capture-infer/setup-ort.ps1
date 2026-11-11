# ONNX Runtime on Windows: 1.31.1 for CUDA (with its CUDA 13 and cuDNN DLLs) in ort\venv, 1.24.4 for DirectML
# (its last version) in ort\dml. ort\cuda.ps1 and ort\dml.ps1 point the spike at one of them.
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed ($LASTEXITCODE)" } }
Set-Location $PSScriptRoot
cmd /c "rustup toolchain install 1.92.0 --profile minimal >nul 2>&1"; Check 'rustup'
foreach ($pair in @(@('venv', 'onnxruntime-gpu[cuda,cudnn]==1.31.1', 'cuda'), @('dml', 'onnxruntime-directml==1.24.4', 'dml'))) {
    $dir, $package, $name = $pair
    if (-not (Test-Path "ort\$dir")) { uv venv -q --python 3.13 "ort\$dir"; Check 'uv venv' }
    uv pip install -q --python "ort\$dir\Scripts\python.exe" $package; Check 'uv pip'
    $site = (Resolve-Path "ort\$dir\Lib\site-packages").Path
    $dirs = @("$site\onnxruntime\capi")
    if (Test-Path "$site\nvidia") {
        $dirs += Get-ChildItem "$site\nvidia" -Recurse -Filter *.dll | ForEach-Object { $_.DirectoryName } | Sort-Object -Unique
    }
    @(
        "`$env:ORT_DYLIB_PATH = '$site\onnxruntime\capi\onnxruntime.dll'",
        "`$env:PATH = '$($dirs -join ';');' + `$env:PATH"
    ) | Set-Content "ort\$name.ps1"
}
