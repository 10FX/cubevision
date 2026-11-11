# cubevision

Rubik's Cube vision, native on the PC: find the cube, read its stickers, and turn a camera feed or a
video into the moves on a 3D cube. It runs the same algorithms as the web app
[ChronoCube](https://chronocube.vercel.app), rewritten in Rust to run faster on a PC.

**Status: early.** The first step, a throwaway prototype, checked that Rust can carry the whole thing:
the two ONNX models on CPU, CUDA and DirectML with the same outputs as Python, a video read faster than
real time, and a webcam on Windows. The numbers are in
[`spikes/capture-infer/FINDINGS.md`](spikes/capture-infer/FINDINGS.md) (in French), and the design in
[`docs/superpowers/specs`](docs/superpowers/specs).

| Measure (RTX 5060 Ti) | Result |
|---|---|
| Detector + reader, one frame, CUDA fp16 (p95) | 13.8 ms on Windows, 15.8 ms on WSL |
| 1080p60 video, decode to reading | 120 fps, cube found on every frame |
| Portrait 480×854 video | 46 fps, cube found on 99.2 % of frames |

The apps themselves (live mode, video mode, 3D cube) come next.

## Licence

Apache-2.0, see [LICENSE](LICENSE).
