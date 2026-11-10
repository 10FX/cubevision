# Prototype capture et inférence : plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** mesurer si Rust tient la caméra, la vidéo et les deux modèles sur la RTX, pour confirmer ou rediscuter le choix de Rust (étape 1 de §10 de la spec).

**Architecture:** un seul binaire jetable, `spike`, dans `spikes/capture-infer/`, hors du futur workspace. La vidéo passe par `ffmpeg` en sous-processus, la webcam par `nokhwa`, les modèles par `ort` en chargement dynamique. Chaque commande imprime un rapport JSON. Le code n'est pas repris tel quel dans les crates ; seul `FINDINGS.md` compte.

**Tech Stack:** Rust 1.92, `ort =2.0.0-rc.15`, ONNX Runtime 1.31, `nokhwa 0.10.11`, `ffmpeg`, `clap`, `serde_json`, `anyhow`, `ndarray`.

**Spec:** `docs/superpowers/specs/2026-11-10-cubevision-design.md`

**Les plans suivants** (`cube-core` et `cube-decode`, puis le reste de §10) s'écrivent une fois le verdict connu : ils en dépendent.

## Global Constraints

- Modèles : `/home/thomas/TIMER/ml2/runs/reader-v7/web/` (`CUBE_MODELS`), lecteur v7. fp16 sur GPU, fp32 sur CPU.
- Détecteur : entrée `image` `[batch,3,320,320]`, sorties `box [batch,4]` et `score [batch,1]`, seuil 0,5, letterbox avec remplissage 0,5, marge de recadrage 1,5.
- Lecteur : entrée `image` `[batch,3,224,224]` RGB 0..1, sorties `present`, `stickers`, `turning`, `orientation`.
- `ort` : `default-features = false`, features `std`, `ndarray`, `api-31`, `load-dynamic`, `cuda`, `directml`. Version exacte, jamais de plage.
- ONNX Runtime 1.23 est exclu.
- Vidéos de test : uniquement `ml2/evals/synthetic-v3/synth-23-001.mp4` (h264 1920×1080 60 i/s) et `synth-23-000.webm` (vp8 480×854 30 i/s).
- **Aucune photo ni vidéo privée du propriétaire** n'est lue, copiée ou commitée. `.gitignore` exclut `*.onnx`, `*.mp4`, `*.webm`, `*.mov`, `*.MOV`, `target/`, `ort/`.
- Un seul travail GPU à la fois. Dans WSL, préfixer les commandes lourdes de `echo 1000 > /proc/self/oom_score_adj`.
- Le dépôt reste local : aucun `git remote`, aucun push.
- Tout commit se termine par :
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_017yaT8RqevodceVQooFtM8D
  ```

**Critères du verdict :**

| Porte | Mesure | Réussite |
|---|---|---|
| P1 Parité | sorties Rust contre Python, même entrée | CPU fp32 : écart max ≤ 1e-4. CUDA fp16 : ≤ 1e-2 |
| P2 GPU | détecteur + lecteur par image, CUDA, WSL et Windows | p95 ≤ 33 ms, session créée sans repli CPU |
| P3 Vidéo | décodage + détecteur + recadrage + lecteur sur les 2 vidéos | i/s ≥ i/s du fichier, cube trouvé sur plus de 50 % des images |
| P4 Webcam | Windows, 1280×720 à 30 i/s, 30 s, inférence comprise | au moins 90 % de la cadence réelle annoncée par la caméra |

**Replis à essayer avant de rediscuter Rust :** P2 : DirectML sous Windows. P4 : `ffmpeg -f dshow` en sous-processus, puis OpenCV.

## Review Focus

1. CUDA demandé mais indisponible : le repli silencieux sur CPU fausserait toutes les mesures ; la commande doit échouer avec un message. (Tâche 2)
2. Webcam absente ou occupée par une autre application : message lisible et code de sortie non nul, pas de panique. (Tâche 4)
3. Vidéo avec rotation dans les métadonnées (téléphone) : `ffmpeg` la tourne, donc la taille d'image attendue doit être inversée, sinon les images sont cisaillées. (Tâche 3)
4. Vidéo tronquée ou `ffmpeg` introuvable : la dernière image incomplète est ignorée, `ffmpeg` absent donne un message. (Tâche 3)
5. Webcam qui ne fournit pas le format demandé : le rapport donne la taille, la cadence et le format réellement obtenus, et P4 se calcule sur ceux-là. (Tâche 4)

---

### Task 1 : modèles sur CPU, parité avec Python

**Files:**
- Create: `.gitignore`, `spikes/capture-infer/Cargo.toml`, `rust-toolchain.toml`, `src/main.rs`, `src/lib.rs`, `src/models.rs`, `src/preprocess.rs`, `src/stats.rs`, `tools/make_reference.py`, `tests/reference.json`, `tests/parity.rs`

**Interfaces:**
- Produces, dans `models.rs` :
  - `enum Device { Cpu, Cuda, DirectMl }`, avec `FromStr` sur `cpu`, `cuda`, `directml`
  - `Models::load(dir: &Path, device: Device, precision: Precision) -> anyhow::Result<Models>`, avec `enum Precision { Fp16, Fp32 }`
  - `Models::detect(&mut self, tensor: &[f32]) -> Result<Detection>`, `struct Detection { boxed: [f32; 4], score: f32 }`
  - `Models::read(&mut self, tensors: &[f32], batch: usize) -> Result<Vec<Reading>>`, `struct Reading { present: Vec<f32>, stickers: Vec<f32>, turning: f32, orientation: Vec<f32> }`
- Produces, dans `preprocess.rs` :
  - `struct Frame { data: Vec<u8>, width: u32, height: u32 }` (RGB)
  - `struct BoxF { x: f32, y: f32, w: f32, h: f32 }`, `struct Letterbox { scale: f32, ox: i32, oy: i32 }`
  - `letterbox(&Frame, size: u32, pad: f32) -> (Vec<f32>, Letterbox)` (CHW)
  - `unletterbox(boxed: [f32; 4], size: u32, &Letterbox) -> BoxF`
  - `reader_crop(&BoxF, margin: f32) -> BoxF`
  - `square_crop(&Frame, &BoxF, size: u32) -> Vec<f32>` (CHW 0..1)
- Produces, dans `stats.rs` : `percentile(values: &mut [f64], p: f64) -> f64`

- [ ] **Step 1 : générer la référence.** `tools/make_reference.py` construit l'entrée déterministe `x[c][y][i] = ((i + y + c) % size) / (size - 1)`, batch 1, l'envoie aux deux modèles fp32 avec le CPU d'ONNX Runtime 1.31.0, et écrit toutes les sorties aplaties dans `tests/reference.json`.

Run: `/home/thomas/TIMER/ml2/.venv/bin/python -I tools/make_reference.py`
Expected: le fichier contient les clés `box`, `score`, `present`, `stickers`, `turning`, `orientation`.

- [ ] **Step 2 : écrire les tests qui échouent.**

```rust
// tests/parity.rs (CUBE_MODELS et ORT_DYLIB_PATH viennent de l'environnement)
#[test] fn cpu_fp32_matches_python() { /* écart absolu max ≤ 1e-4 sur chaque sortie de reference.json */ }

// src/preprocess.rs
#[test] fn letterbox_landscape() { // 1920×1080 dans 320
    assert_eq!(geometry(1920, 1080, 320), (70, 0, 1.0 / 6.0)) } // (oy, ox, scale)
#[test] fn letterbox_portrait() { // 480×854 dans 320 : largeur arrondie à 180
    assert_eq!(ox(480, 854, 320), 70); assert_eq!(oy(480, 854, 320), 0) }
#[test] fn reader_crop_is_square_around_the_box() {
    let c = reader_crop(&BoxF { x: 100., y: 100., w: 40., h: 80. }, 1.5);
    assert_eq!((c.x, c.y, c.w, c.h), (60., 80., 120., 120.)) }
#[test] fn unletterbox_undoes_letterbox() { /* boîte [0, 0.21875, 1, 0.78125] sur 1920×1080 → (0, 0, 1920, 1080) à 0,5 px près */ }
#[test] fn percentile_of_1_to_100() { /* p95 = 95 à 1 près */ }
```

- [ ] **Step 3 : vérifier qu'ils échouent.**

Run: `ORT_DYLIB_PATH=/home/thomas/TIMER/ml2/.venv/lib/python3.13/site-packages/onnxruntime/capi/libonnxruntime.so.1.31.0 CUBE_MODELS=/home/thomas/TIMER/ml2/runs/reader-v7/web cargo test`
Expected: échec de compilation, fonctions absentes. Installer d'abord Rust dans WSL si besoin (`rustup`, version fixée par `rust-toolchain.toml` à `1.92.0`).

- [ ] **Step 4 : implémenter `models.rs`, `preprocess.rs`, `stats.rs`.** Reprendre `letterbox`, `unletterbox`, `readerCrop` et `squareCrop` de `src/lib/vision/detector.ts` et `roi.ts` (moyenne de `span` pixels par axe), sur du RGB au lieu du RGBA. Sessions créées avec `error_on_failure()` sur le moteur demandé.

- [ ] **Step 5 : vérifier que les tests passent** avec la même commande. Expected : tous PASS.

- [ ] **Step 6 : commit** de `.gitignore` et `spikes/capture-infer/` (`spike: models on the CPU, same outputs as Python`).

### Task 2 : modèles sur la RTX, WSL et Windows

**Files:**
- Create: `spikes/capture-infer/src/bench.rs`, `setup-ort.sh`, `setup-ort.ps1`, `run-windows.sh`, `results/`
- Modify: `src/main.rs`, `tests/parity.rs`

**Interfaces:**
- Consumes: `Models`, `percentile`.
- Produces: `spike infer --models DIR --device D [--precision P] --seconds N`, qui imprime `{device, precision, detector_ms: {p50, p95}, reader_ms: {p50, p95}, total_p95_ms, reader_batch8_ms_per_image}`.

- [ ] **Step 1 : écrire les tests qui échouent.**

```rust
#[test] #[ignore = "needs a CUDA runtime"] fn cuda_fp16_is_close_to_python() { /* écart max ≤ 1e-2 */ }
#[test] #[ignore = "needs a CPU-only runtime"]
fn cuda_refused_without_cuda() { // ORT_DYLIB_PATH pointe la lib CPU de ml2/.venv
    let error = Models::load(dir, Device::Cuda, Precision::Fp16).unwrap_err();
    assert!(error.to_string().to_lowercase().contains("cuda")) }
```

- [ ] **Step 2 : préparer ONNX Runtime GPU.** `setup-ort.sh` et `setup-ort.ps1` créent `ort/venv` avec `uv` et `onnxruntime-gpu[cuda,cudnn]==1.31.1`, puis écrivent `ort/env` avec `ORT_DYLIB_PATH` vers la lib de `onnxruntime/capi` et les dossiers `nvidia/*` nécessaires (`LD_LIBRARY_PATH`, ou `PATH` sous Windows).

Run: `./setup-ort.sh && . ort/env && ls "$ORT_DYLIB_PATH"`
Expected: le fichier existe.

- [ ] **Step 3 : lancer `cargo test -- --ignored`, implémenter `bench.rs` et la commande `infer`, relancer jusqu'à PASS.** Les mesures tournent sur une entrée fixe, après 20 inférences de chauffe.

- [ ] **Step 4 : mesurer dans WSL.**

Run: `cargo run --release -- infer --models "$CUBE_MODELS" --device cuda --seconds 20 > results/infer-wsl-cuda.json`, puis `--device cpu --precision fp32`.
Expected: `total_p95_ms` ≤ 33 en CUDA pour P2. Sinon, noter la valeur et continuer.

- [ ] **Step 5 : mesurer sous Windows.** `run-windows.sh` copie les sources sans `target/` ni `ort/` vers `C:\Users\Thomas\cubevision-spike`, lance `setup-ort.ps1` puis `cargo run --release` via `powershell.exe`, et copie le JSON dans `results/`. `rustup` y installera 1.92.0 via `rust-toolchain.toml`.

Run: `./run-windows.sh infer --device cuda --seconds 20`, puis `--device directml`, puis `--device cpu --precision fp32`.
Expected: trois JSON dans `results/`. Un échec CUDA s'y note, et DirectML sert de repli.

- [ ] **Step 6 : commit** (`spike: both models on the RTX, WSL and Windows`).

### Task 3 : vidéo de bout en bout

**Files:**
- Create: `spikes/capture-infer/src/video.rs`, `src/pipeline.rs`, `tests/video.rs`
- Modify: `src/main.rs`

**Interfaces:**
- Consumes: `Models`, `Frame`, `letterbox`, `unletterbox`, `reader_crop`, `square_crop`.
- Produces:
  - `VideoReader::open(path: &Path, ffmpeg: &str) -> Result<VideoReader>`, `.size() -> (u32, u32)`, `.fps() -> f64`, `.next_frame() -> Result<Option<Frame>>`
  - `process_frame(&mut Models, &Frame, last: &mut Option<BoxF>) -> Result<FrameTimes>`, `struct FrameTimes { letterbox_ms, detector_ms, crop_ms, reader_ms: f64, found: bool }`. Sans détection valide, le lecteur lit la dernière boîte, ou un carré central au départ.
  - `spike video FILE --models DIR --device D`, qui imprime `{file, codec, width, height, file_fps, frames, processed_fps, found_share, ms: {decode, letterbox, detector, crop, reader}}`, avec le nom du fichier seul, sans chemin.

- [ ] **Step 1 : écrire les tests qui échouent.** Les vidéos sont générées dans un dossier temporaire avec `ffmpeg -f lavfi -i testsrc=size=320x240:rate=30:duration=1`.

```rust
#[test] fn reads_every_frame() { /* 30 images de 320×240×3 octets */ }
#[test] fn rotated_video_swaps_size() { // ffmpeg -display_rotation 90 -i in.mp4 -c copy out.mp4
    assert_eq!(VideoReader::open(&out, "ffmpeg")?.size(), (240, 320)) }
#[test] fn truncated_video_ends_cleanly() { /* mp4 coupé à 60 % : next_frame finit par Ok(None), chaque image est complète */ }
#[test] fn missing_ffmpeg_is_explained() { /* open(.., "ffmpeg-absent") → l'erreur contient "ffmpeg" */ }
```

- [ ] **Step 2 : vérifier qu'ils échouent.** Run: `cargo test --test video`. Expected : échec de compilation.

- [ ] **Step 3 : implémenter `video.rs` et `pipeline.rs`.** `ffprobe` donne la taille, la cadence et la rotation ; `ffmpeg -i FILE -f rawvideo -pix_fmt rgb24 -` est lu avec `read_exact`, et une lecture incomplète à la fin termine le flux.

- [ ] **Step 4 : vérifier que les tests passent.** Run: `cargo test --test video`. Expected : PASS.

- [ ] **Step 5 : mesurer P3.**

Run: `cargo run --release -- video /home/thomas/TIMER/ml2/evals/synthetic-v3/synth-23-001.mp4 --models "$CUBE_MODELS" --device cuda > results/video-mp4.json`, puis la même commande avec `synth-23-000.webm` vers `results/video-webm.json`.
Expected: `processed_fps` ≥ `file_fps` et `found_share` > 0,5 sur les deux.

- [ ] **Step 6 : commit** (`spike: a video read end to end`).

### Task 4 : webcam sous Windows

**Files:**
- Create: `spikes/capture-infer/src/cam.rs`
- Modify: `src/main.rs`, `Cargo.toml` (`nokhwa = { version = "=0.10.11", features = ["input-native"] }`)

**Interfaces:**
- Consumes: `process_frame`, `Models`, `percentile`.
- Produces: `spike cam --index I --width W --height H --fps F --seconds N [--models DIR --device D]`, qui imprime `{camera, requested: {w, h, fps}, actual: {w, h, fps, format}, frames, delivered_fps, delivered_share, frame_gap_ms: {p50, p95, max}, inference: {found_share, total_p95_ms} | null}`.

La capture tourne dans son thread et ne garde que la dernière image, comme le direct de la spec ; `delivered_share` compare les images obtenues à `actual.fps × seconds`.

- [ ] **Step 1 : écrire le test qui échoue**, dans `src/cam.rs` : `missing_camera_is_explained`, qui ouvre l'index 99 et vérifie que `run` renvoie une erreur contenant `camera`, sans paniquer.

- [ ] **Step 2 : implémenter `cam.rs` et vérifier sous Windows.** Run: `./run-windows.sh test cam`. Expected : PASS.

- [ ] **Step 3 : mesurer la capture seule, puis avec inférence.** La webcam (SKILLKORP SKP_W-10) doit être libre ; un cube devant elle donne un `found_share` utile. Aucune image n'est enregistrée.

Run: `./run-windows.sh cam --index 0 --width 1280 --height 720 --fps 30 --seconds 30`, puis la même commande avec `--models … --device cuda`.
Expected: `delivered_share` ≥ 0,9 dans les deux cas pour P4. La caméra occupée doit donner un message d'erreur.

- [ ] **Step 4 : commit** (`spike: the webcam on Windows`).

### Task 5 : verdict

**Files:**
- Create: `spikes/capture-infer/FINDINGS.md`

- [ ] **Step 1 : écrire `FINDINGS.md`.** Une ligne par porte (P1 à P4) avec la valeur mesurée, le fichier de `results/` qui la prouve et réussi ou échoué ; les replis essayés ; ce que les crates `cube-infer` et `cube-capture` devront reprendre (versions, chemin ONNX Runtime, `ffmpeg` ou non, format webcam) ; et le verdict : Rust confirmé, ou à rediscuter.

- [ ] **Step 2 : contrôler qu'aucun média n'est suivi.** Run: `git ls-files | grep -iE '\.(mp4|webm|mov|jpg|png|onnx)$'`. Expected : aucune ligne.

- [ ] **Step 3 : commit** (`spike: findings`) et présenter le verdict au propriétaire, avec les chiffres, avant d'écrire le plan suivant.
