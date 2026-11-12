# cube-infer et cube-capture : plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** donner au workspace les deux crates qui manquent entre une image et des lectures : `cube-infer` (les deux
modèles ONNX, le matériel, le cadrage du cube) et `cube-capture` (les images d'un fichier vidéo ou d'une caméra).

**Architecture :** `cube-infer` dépend de `cube-decode` (il en rend les `FaceReading`, et lit ses `Frame` RGBA) et de
`ort` en chargement dynamique. Tout ce qui ne touche pas ONNX (géométrie, suivi de la boîte, sorties du lecteur) est
porté de `detector.ts`, `roi.ts` (`squareCrop`) et `model.ts`, et comparé au TypeScript par des fixtures. Le détecteur
est un trait : le suivi se teste avec un détecteur scripté. `cube-capture` ne dépend d'aucune autre crate du
workspace : ffmpeg en sous-processus pour les fichiers, nokhwa (Windows) ou ffmpeg v4l2 (Linux) pour la caméra.
La boucle image par image (miroir, mouvement, suivi) appartient à l'exécutable, étape 4.

**Tech Stack :** Rust 1.92, édition 2024 ; `ort =2.0.0-rc.15` (`load-dynamic`, `api-24`, `cuda`, `directml`,
`openvino`), `anyhow`, `serde`/`serde_json`, `rayon` ; `nokhwa =0.10.11` sous Windows. ffmpeg et ffprobe dans le PATH.

**Spec :** `docs/superpowers/specs/2026-11-10-cubevision-design.md` (§3, §4, §5, §7, §8, §10 étape 3) ; constats du
prototype : `spikes/capture-infer/FINDINGS.md`.

## Global Constraints

- Une crate ne connaît pas les suivantes : `cube-infer` peut dépendre de `cube-core` et `cube-decode`, jamais de
  `cube-capture` ; `cube-capture` ne dépend d'aucune crate du workspace.
- `ort =2.0.0-rc.15`, `load-dynamic` ; un seul `Environment` par processus ; `error_on_failure()` sur le moteur
  demandé, jamais de repli CPU muet. ONNX Runtime 1.23 exclu.
- Sessions GPU : 1 fil intra-op, sans attente active. Une forme d'entrée fixe par session : lots toujours complétés
  à 8 en vidéo, une session de forme 1 pour l'unité.
- fp16 sur CUDA et DirectML ; fp32 sur CPU et OpenVINO (le même réseau).
- Images RGBA, 4 octets par pixel, comme le site et comme `cube_decode::motion::Frame`.
- Fidélité numérique (parties portées) : `number` → `f64`, `Float32Array` → `f32` stocké et calcul en f64,
  `Math.round` → `js_round`, écriture dans un `Uint8ClampedArray` → arrondi au pair le plus proche, borné 0..255.
- Les tests qui demandent les modèles ou ONNX Runtime sont `#[ignore]` et lisent `CUBE_MODELS` et `ORT_DYLIB_PATH` :
  `cargo test --workspace` reste vert sans eux.
- **Aucune image réelle du propriétaire** dans le dépôt ni dans les fixtures. Les vidéos de test sont fabriquées
  par ffmpeg (`testsrc`) ou sont des vidéos de synthèse, jamais commitées (`*.mp4`, `*.webm` ignorés).
- Commits de cubevision avec l'adresse noreply et la remorque de session. Le générateur de fixtures se commite dans
  ChronoCube sur `feat/cube-reader`, ce fichier seul. Rien n'est poussé sans l'accord du propriétaire.

## Review Focus

- Une image dont `data` est plus courte que `largeur × hauteur × 4`, ou de taille nulle : erreur claire, pas de
  lecture hors bornes ni de panic obscur. Test en tâche 1.
- Une région ou une boîte entièrement hors de l'image, de taille nulle ou NaN (sortie du détecteur) : pas de panic,
  le carré garde le gris (letterbox) ou répète le bord (lecteur). Tests en tâches 1 et 2.
- Des sorties du modèle NaN ou hors de 0..1 : aucune face inventée, pas de panic. Test en tâche 3.
- Modèles absents, bibliothèque ONNX Runtime absente, moteur demandé indisponible : message qui dit quoi faire
  (`cubevision fetch-models`, `ORT_DYLIB_PATH`, `--device`), jamais de repli muet. Tests en tâche 4.
- Vidéo tronquée, sans piste vidéo, tournée de 90°, ffmpeg absent, caméra absente : erreur claire ou fin propre.
  Tests en tâches 5 et 6.

---

## Fichiers

```
crates/cube-infer/src/lib.rs          ré-exports
crates/cube-infer/src/geometry.rs     BoxF, letterbox, unletterbox, reader_crop, square_crop, tensor   (detector.ts, roi.ts, model.ts)
crates/cube-infer/src/follow.rs       Detection, Detector, detect_with_zoom, BoxTracker, CubeFollower  (detector.ts)
crates/cube-infer/src/output.rs       ReaderOutput, readings_from_output, confidence                    (model.ts)
crates/cube-infer/src/device.rs       Device, Precision, ordre du choix automatique
crates/cube-infer/src/models.rs       métadonnées JSON, sessions, OnnxDetector, OnnxReader, Models
crates/cube-infer/tests/…             geometry.rs, follow.rs, output.rs (fixtures), models.rs (#[ignore])
crates/cube-capture/src/lib.rs        Image, ré-exports
crates/cube-capture/src/video.rs      VideoReader (ffprobe + ffmpeg)
crates/cube-capture/src/latest.rs     Latest<T> : la dernière image seulement
crates/cube-capture/src/camera.rs     Camera (nokhwa sous Windows, ffmpeg v4l2 sous Linux)
crates/cube-capture/tests/video.rs    vidéos testsrc fabriquées par ffmpeg
```

Générateur : `~/TIMER/scripts/vision/fixtures.test.ts`, cas `cube-infer: …`, écrit dans
`crates/cube-infer/tests/fixtures/`. Commande, depuis ChronoCube :
`timeout 900 npx vitest run --config vitest.vision.config.mts scripts/vision/fixtures.test.ts -t "cube-infer"`.

Vérification de chaque tâche, depuis la racine : `cargo test -q -p <crate> --release`, puis à la fin de la tâche
`cargo clippy --workspace --all-targets -q -- -D warnings` et `cargo fmt --all --check`.

---

### Task 1 : cube-infer, la géométrie

**Files :** Create `crates/cube-infer/{Cargo.toml,src/lib.rs,src/geometry.rs,tests/geometry.rs,tests/fixtures/geometry.json}` ;
Modify `~/TIMER/scripts/vision/fixtures.test.ts`.

**Interfaces :**
- Consumes : `cube_decode::motion::Frame<'a> { data: &[u8], width: u32, height: u32 }` (RGBA).
- Produces :
  - `pub struct BoxF { pub x: f64, pub y: f64, pub w: f64, pub h: f64 }` (`Clone, Copy, Debug, PartialEq`)
  - `pub struct Letterbox { pub scale: f64, pub ox: i64, pub oy: i64, pub rx: f64, pub ry: f64 }`
  - `pub fn check(frame: &Frame) -> Result<(), FrameError>` : `data.len() >= w*h*4`, `w > 0`, `h > 0`
  - `pub fn letterbox(frame: &Frame, size: u32, pad: f32, region: Option<&BoxF>) -> (Vec<f32>, Letterbox)` (NCHW 0..1)
  - `pub fn unletterbox(boxed: [f32; 4], size: u32, lb: &Letterbox) -> BoxF`
  - `pub fn reader_crop(boxed: &BoxF, margin: f64) -> BoxF`
  - `pub fn square_crop(frame: &Frame, boxed: &BoxF, size: u32) -> Vec<u8>` (RGBA `size×size`, alpha 255)
  - `pub fn tensor(rgba: &[u8], out: &mut [f32])` (`toTensorData` : plans R, G, B en 0..1)

- [ ] **Step 1 :** générateur `inferGeometry()` : 6 images pseudo-aléatoires (mulberry32, tailles 64×48, 48×64, 37×53,
  200×120, 16×16, 1×1), pour chacune 8 régions (dont entière, débordante, hors image, plus petite qu'un pixel) :
  `letterbox` (taille 32, pad 0.5) → `scale, ox, oy` et somme de contrôle du tenseur (somme f64 des valeurs f32 et
  16 valeurs échantillonnées) ; `unletterbox` de 4 boîtes ; `readerCrop` (marges 1.5 et 1) ; `squareCrop`
  (tailles 24 et 8) → octets en hexadécimal. Écrit `geometry.json`.
- [ ] **Step 2 :** test `geometry_matches_typescript` (égalité exacte des octets, `scale` et boîtes à 1e-12, tenseur
  exact en f32) et `a_short_or_empty_frame_is_refused`, `a_nan_box_does_not_panic`. Run : échec, crate vide.
- [ ] **Step 3 :** écrire `geometry.rs`. Les fonctions qui lisent des pixels paniquent avec le message de `check`
  si l'image est invalide (documenté) ; une boîte NaN ou nulle rend un carré de bord répété, une région NaN un carré gris.
- [ ] **Step 4 :** Run : PASS. Clippy, fmt.
- [ ] **Step 5 :** Commit `cube-infer: the frame into the models' squares` ; commit du générateur dans ChronoCube.

### Task 2 : cube-infer, suivre le cube

**Files :** Create `src/follow.rs`, `tests/follow.rs`, `tests/fixtures/follow.json` ; Modify le générateur.

**Interfaces :**
- Consumes : tâche 1.
- Produces :
  - `pub struct Detection { pub boxed: BoxF, pub score: f32 }`
  - `pub struct Look<'a> { pub frame: &'a Frame<'a>, pub region: Option<BoxF> }`
  - `pub trait Detector { fn crop_margin(&self) -> f64; fn look(&mut self, looks: &[Look]) -> anyhow::Result<Vec<Option<Detection>>>; }`
    (`look` : un regard par entrée, sans regard rapproché)
  - `pub const SMALL_CUBE: f64 = 0.25; pub const ZOOM: f64 = 4.0;`
  - `pub fn is_small(frame: &Frame, boxed: &BoxF) -> bool`, `pub fn zoom_region(near: &BoxF) -> BoxF`
  - `pub fn detect_with_zoom(detector: &mut impl Detector, frame: &Frame, hint: Option<&BoxF>) -> Result<Option<Detection>>`
  - `pub struct BoxTracker` : `new(keep: u32, follow: f64)`, `Default` (8, 0.6), `update(Option<BoxF>) -> Option<BoxF>`,
    `current() -> Option<BoxF>`, `reset()`
  - `pub struct CubeFollower<D: Detector>` : `new(detector, every: u32)`, `crop(&mut self, frame) -> Result<Option<BoxF>>`,
    `crop_batch(&mut self, frames: &[Frame]) -> Result<Vec<Option<BoxF>>>`, `read(&mut self, cube: bool)`,
    `pub boxes: BoxTracker`, `detector() -> &D`

- [ ] **Step 1 :** générateur `inferFollow()` : un détecteur scripté (réponse tirée d'un hachage de l'indice de
  l'image et de la région : absent, petit cube, grand cube, saut) ; 6 scénarios de 40 images (`every` 3 avec
  `read` vrai/faux tiré, `every` 1 par `crop`, `every` 1 par `cropBatch` en lots de 8, petit cube perdu puis
  retrouvé de près, tout absent, sauts) ; sorties : la boîte rendue à chaque image, et le journal des regards
  demandés (image, région). Aussi 30 suites `BoxTracker.update`.
- [ ] **Step 2 :** tests `the_tracker_matches_typescript`, `the_follower_matches_typescript` (boîtes à 1e-9, journal
  des regards identique), `a_batch_crops_as_frame_by_frame` (propriété : `crop_batch` = `crop` un à un quand `every`
  vaut 1, sur les scénarios). Run : échec.
- [ ] **Step 3 :** écrire `follow.rs`. `crop_batch` suit `cropBatch` du site (regards groupés, puis les petits de
  près, puis les perdus un par un autour de la boîte suivie).
- [ ] **Step 4 :** Run : PASS. Clippy, fmt.
- [ ] **Step 5 :** Commits (cubevision, ChronoCube).

### Task 3 : cube-infer, des sorties du lecteur aux lectures

**Files :** Create `src/output.rs`, `tests/output.rs`, `tests/fixtures/outputs.json` ; Modify le générateur.

**Interfaces :**
- Produces :
  - `pub struct ReaderOutput { pub present: [f32; 6], pub stickers: Vec<f32> /* 6×9×7 */, pub turning: f32, pub orientation: Option<[f32; 9]> }`
  - `pub const PRESENT: f32 = 0.5; pub const MIN_CONFIDENCE: f64 = 0.6; pub const TURNING: f32 = 0.5;`
  - `pub fn readings_from_output(out: &ReaderOutput, ordered: bool, still: bool) -> Vec<FaceReading>`
  - `pub fn confidence(stickers: &[Option<[f32; 6]>; 9]) -> f64`
  - `impl ReaderOutput { pub fn orientation64(&self) -> Option<[f64; 9]> }`

- [ ] **Step 1 :** générateur `inferOutputs()` : 200 sorties aléatoires (softmax de logits tirés, faces présentes ou
  non, cases cachées, `turning` de part et d'autre de 0.5, quelques valeurs exactement aux seuils), chacune avec
  `ordered` et `still` tirés ; sortie : les lectures (centres, cases, rotation).
- [ ] **Step 2 :** tests `readings_match_typescript` (f32 exacts) et `nan_outputs_invent_no_face`. Run : échec.
- [ ] **Step 3 :** écrire `output.rs` ; les faces passent par `FaceReading::sanitized`.
- [ ] **Step 4 :** Run : PASS. Clippy, fmt.
- [ ] **Step 5 :** Commits.

### Task 4 : cube-infer, les modèles et le matériel

**Files :** Create `src/device.rs`, `src/models.rs`, `tests/models.rs`, `tests/reference.json` (copie de
`spikes/capture-infer/tests/reference.json`, sorties d'ONNX Runtime 1.31 en Python sur une entrée fixe).

**Interfaces :**
- Consumes : tâches 1 à 3.
- Produces :
  - `pub enum Device { Cpu, Cuda, DirectMl, OpenVinoCpu, OpenVinoGpu, OpenVinoNpu }` : `name()`, `FromStr`,
    `precision() -> Precision` ; `pub enum Precision { Fp16, Fp32 }`
  - `pub const AUTO: [Device; 4] = [Cuda, DirectMl, OpenVinoGpu, Cpu]` : l'ordre essayé sans `--device`
  - `pub const BATCH: usize = 8`
  - `pub struct Models { pub detector: OnnxDetector, pub reader: OnnxReader, pub device: Device, pub fallback: Option<String> }`
  - `Models::load(dir: &Path, device: Option<Device>) -> anyhow::Result<Models>` : `Some` : ce moteur ou une erreur ;
    `None` : le premier de `AUTO` qui ouvre les deux sessions et passe un essai à blanc ; arrivé au CPU après un
    échec GPU, `fallback` porte l'avertissement
  - `OnnxDetector` : `impl Detector` (lots complétés à `BATCH`, session de forme 1 pour un seul regard), `size()`
  - `OnnxReader` : `size() -> u32`, `ordered() -> bool`, `read(&mut self, crops: &[&[u8]]) -> Result<Vec<ReaderOutput>>`
    (RGBA `size×size` ; 1 crop : forme 1 ; sinon lots complétés à `BATCH`)
  - `pub fn library() -> anyhow::Result<PathBuf>` : `ORT_DYLIB_PATH`, sinon la bibliothèque à côté de l'exécutable

- [ ] **Step 1 :** tests sans modèles : `every_device_is_named_as_it_is_asked_for`, `fp16_only_on_cuda_and_directml`,
  `missing_models_point_to_fetch_models` (dossier vide → message contenant `cubevision fetch-models`),
  `metadata_is_read` (un `cube-detector.json` et un `cube-reader.json` écrits dans un dossier temporaire : taille,
  seuil, marge, fichiers, `ordered`). Tests `#[ignore]` avec modèles : `cpu_matches_python` (1e-4),
  `cuda_is_close_to_python` (1e-2), `a_batch_reads_as_one_by_one` (9 crops : 1e-4 sur CPU),
  `a_black_frame_shows_no_cube`, `an_unavailable_device_is_an_error` (DirectML sous Linux), `auto_picks_a_device`.
  Run : échec.
- [ ] **Step 2 :** écrire `device.rs` et `models.rs` d'après `spikes/capture-infer/src/models.rs`, en lisant les
  métadonnées JSON au lieu de constantes. Sessions chauffées au chargement (un passage à blanc par forme).
- [ ] **Step 3 :** Run sans modèles : PASS. Run avec (`. spikes/capture-infer/ort/env`,
  `CUBE_MODELS=<dossier des modèles publics>`, `--include-ignored`) : PASS sur CPU et CUDA. Clippy, fmt.
- [ ] **Step 4 :** Commit `cube-infer: the two models on the device asked for`.

### Task 5 : cube-capture, les fichiers vidéo

**Files :** Create `crates/cube-capture/{Cargo.toml,src/lib.rs,src/video.rs,tests/video.rs}`.

**Interfaces :**
- Produces :
  - `pub struct Image { pub data: Vec<u8> /* RGBA */, pub width: u32, pub height: u32, pub t: f64 /* ms */ }`
  - `pub struct VideoOptions { pub ffmpeg: String, pub max_width: Option<u32> }` ; `Default` : `"ffmpeg"`, `Some(1280)`
    (le site lit en 1280 de large au plus : `scale=w:h`, `h = round(H·w/W/2)·2`)
  - `pub struct VideoInfo { pub width: u32, pub height: u32, pub fps: f64, pub codec: String, pub rotation: i32, pub duration: Option<f64> }`
  - `VideoReader::open(path: &Path, options: &VideoOptions) -> anyhow::Result<VideoReader>`, `info() -> &VideoInfo`
    (taille des images rendues), `next_image() -> Result<Option<Image>>`
  - Temps de chaque image : `pts_time` des paquets (ffprobe, triés), arrondis à la ms ; au-delà : `round(i·1000/fps)`.

- [ ] **Step 1 :** tests (vidéos `testsrc` 320×240, 30 i/s, 1 s, fabriquées dans un dossier temporaire) :
  `reads_every_frame` (30 images RGBA, temps 0, 33, 67…), `a_wide_video_is_scaled_to_1280` (1920×1080 → 1280×720 ;
  `max_width: None` → 1920×1080), `rotated_video_swaps_size`, `truncated_video_ends_cleanly`,
  `missing_ffmpeg_is_explained`, `a_file_that_is_no_video_is_explained`, `frame_times_follow_the_timestamps`
  (vidéo à cadence variable faite avec `setpts`). Run : échec.
- [ ] **Step 2 :** écrire `video.rs` d'après `spikes/capture-infer/src/video.rs` (`-pix_fmt rgba`).
- [ ] **Step 3 :** Run : PASS. Clippy, fmt.
- [ ] **Step 4 :** Commit `cube-capture: frames out of a video file`.

### Task 6 : cube-capture, la caméra

**Files :** Create `src/latest.rs`, `src/camera.rs` ; Modify `Cargo.toml` (nokhwa sous `cfg(windows)`), `src/lib.rs`.

**Interfaces :**
- Produces :
  - `pub struct Latest<T>` (`Clone`, partagé) : `put(T)`, `take() -> Option<T>`, `dropped() -> u64`, `fail(String)`,
    `error() -> Option<String>`
  - `pub struct CameraRequest { pub index: u32, pub width: u32, pub height: u32, pub fps: u32 }` ; `Default` 0, 1280×720, 30
  - `pub struct CameraFormat { pub name: String, pub width: u32, pub height: u32, pub fps: u32 }`
  - `Camera::open(request: &CameraRequest) -> anyhow::Result<Camera>` (15 s au plus), `format() -> &CameraFormat`,
    `latest() -> anyhow::Result<Option<Image>>` (l'image la plus récente, les autres jetées ; `t` en ms depuis
    l'ouverture ; `Err` quand la caméra s'arrête), `dropped() -> u64` ; `Drop` arrête le fil.
  - Windows : nokhwa MJPEG, cadence prise du format demandé. Linux : `ffmpeg -f v4l2 -i /dev/video<index>` en
    RGBA. Autres systèmes : erreur claire.

- [ ] **Step 1 :** tests : `only_the_latest_frame_is_kept` (3 `put`, 1 `take`, `dropped() == 2`),
  `a_failure_reaches_the_reader`, `missing_camera_is_explained` (index 99 → erreur contenant « camera »). Run : échec.
- [ ] **Step 2 :** écrire `latest.rs` et `camera.rs` d'après `spikes/capture-infer/src/cam.rs`.
- [ ] **Step 3 :** Run : PASS sous Linux. Compiler et tester sous Windows (`cargo.exe test -p cube-capture`, sources
  copiées comme le fait `spikes/capture-infer/run-windows.sh`). Clippy, fmt.
- [ ] **Step 4 :** Commit `cube-capture: the camera's latest frame`.

### Task 7 : de bout en bout, et le README

**Files :** Create `crates/cube-infer/examples/read_video.rs` ; Modify `README.md`.

- [ ] **Step 1 :** exemple `read_video <vidéo> [--device d]` (dev-dependency `cube-capture`) : lit une vidéo par
  lots de 8 avec `CubeFollower` (`every` 1), `square_crop`, le lecteur, `readings_from_output` ; imprime les images
  lues, la part où le cube est trouvé, les images par seconde, le moteur.
- [ ] **Step 2 :** le lancer sur une vidéo de synthèse, CPU puis CUDA : le cube est trouvé sur plus de 90 % des
  images et lu sur une majorité ; noter les chiffres mesurés.
- [ ] **Step 3 :** README : état (étape 3 faite), les deux crates dans le tableau des modules, comment lancer les
  tests avec modèles, les limites dites (caméra Linux non essayée sur une vraie caméra, matériel non testé).
- [ ] **Step 4 :** suite complète, clippy, fmt. Commit `cube-infer and cube-capture in the README`.
