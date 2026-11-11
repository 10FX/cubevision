# cube-core et cube-decode : plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** porter en Rust le modèle du cube (`cube-core`) et le passage des lectures aux mouvements (`cube-decode`) de
ChronoCube, avec les mêmes états, mouvements et scores que le TypeScript sur des fixtures qu'il génère.

**Architecture :** un workspace Cargo à la racine, `crates/cube-core` (aucune dépendance) et `crates/cube-decode`
(`cube-core` et `rayon`). Chaque fichier Rust porte un fichier TypeScript, mêmes noms en snake_case. Un script de
ChronoCube, `scripts/vision/fixtures.test.ts`, écrit des fixtures JSON (entrées et sorties du TypeScript) dans
`crates/*/tests/fixtures/` ; les tests Rust les relisent et comparent.

**Tech Stack :** Rust 1.92, édition 2024, `rayon` ; `serde_json` en dépendance de test seulement. Côté ChronoCube :
vitest (`vitest.vision.config.mts`), le mode `DUMP` de `scripts/vision/eval.test.ts`.

**Spec :** `docs/superpowers/specs/2026-11-10-cubevision-design.md` (§3 architecture, §8 tests, §9 dépôt publié,
§10 étape 2).

## Global Constraints

- `cube-core` et `cube-decode` ne dépendent ni d'ONNX, ni de la caméra, ni du GPU : ils se testent seuls.
- `cube-core` : aucune dépendance (`[dependencies]` vide).
- Non porté : `searchSolve` et le détecteur classique (`stickers`, `color`, `roi`), `plan`, `warm`, `passes`, `passWorker`.
- Parité : le script de ChronoCube génère des fixtures à partir des lectures de vidéos de synthèse
  (`ml2/evals/synthetic-v3`, mode `DUMP`) et de cas aléatoires ; les tests Rust retrouvent les mêmes états, mouvements
  et scores.
- **Aucune image réelle du propriétaire** dans le dépôt ni dans les fixtures : jamais `ml2/data/real-*`, jamais
  `ml2/runs/dump-v7` (lectures de ses vidéos réelles, nommées par leur mélange).
- Commits de cubevision avec `108695987+10FX@users.noreply.github.com` (vérifier `git log --format=%ae -1`) et la
  remorque de session. Le script de fixtures se commite dans ChronoCube sur `feat/cube-reader`, jamais sur `master`.
- Fidélité numérique : chaque `Float32Array` du TypeScript est `f32` en Rust (valeur stockée arrondie en f32, calculs
  en f64 comme JS) ; chaque `number` est `f64` ; `Uint8Array` `u8`, `Int32Array` `i32`. Les nombres des fixtures se
  lisent en `f64` puis `as f32` (comme `Float32Array.from`), jamais directement en `f32`. Tris stables
  (`sort_by`, comme `Array.prototype.sort`), même comparateur.
- Tolérance des scores comparés : 1e-9 relatif ; mouvements, états, booléens et entiers exacts.

## Review Focus

- Des probabilités NaN ou nulles venues du modèle : aucun panic (les tris de Rust ≥ 1.81 paniquent sur un ordre non
  total) ; une case NaN compte comme cachée, une probabilité 0 donne `-inf` comme en JS. Test dans la tâche 4.
- Un mélange avec des jetons inconnus ou des couches hors du 3×3 (`"R foo U"`, `"4R"`, `"2-5Rw"`) : ignorés ou sans
  effet, comme le TypeScript, jamais de panic. Tests dans les tâches 1 et 2.
- Des images sans aucune face lue, ou toutes vides : `analyze_readings` rend `None`, le décodeur `explain` sans
  mouvement. Tests dans les tâches 6 et 10.
- Deux images au même instant, ou des temps qui reculent : pas de panic, même résultat que le TypeScript. Test dans
  la tâche 10.
- Une longue vidéo (3 000 lectures) : le décodage finit, sans panic ni dépassement de pile (pas de récursion sur les
  lectures). Test dans la tâche 6.

---

## Fichiers

```
Cargo.toml                                  workspace : crates/*, spikes/ exclu
crates/cube-core/src/lib.rs                 ré-exports
crates/cube-core/src/notation.rs            lib/cube/notation.ts (parse, format, invert)
crates/cube-core/src/cube.rs                lib/cube/stickerCube.ts, 3×3 seulement
crates/cube-core/src/goals.rs               lib/cube/goals.ts : ORIENTATIONS, cross, solved_slots, cfop_progress
crates/cube-core/tests/parity.rs            contre tests/fixtures/core.json
crates/cube-decode/src/lib.rs
crates/cube-decode/src/reading.rs           types : FaceReading, Reading, FrameReadings, Motion, TimedMove
crates/cube-decode/src/state.rs             vision/search.ts sans searchSolve
crates/cube-decode/src/decode/{mod,sequences,prepare,pass,bridge}.rs   vision/decode.ts
crates/cube-decode/src/help.rs              vision/help.ts
crates/cube-decode/src/reconstruct.rs       smartcube/analysis.ts
crates/cube-decode/src/present.rs           smartcube/present.ts
crates/cube-decode/src/orientation.rs       vision/orientation.ts
crates/cube-decode/src/pipeline.rs          vision/pipeline.ts : analyzeReadings et ce qu'il appelle
crates/cube-decode/src/live.rs              vision/live.ts
crates/cube-decode/src/motion.rs            vision/motion.ts
crates/cube-decode/src/mirror.rs            vision/mirror.ts
crates/cube-decode/src/truth.rs             vision/truth.ts
crates/cube-decode/tests/common/mod.rs      lecture des fixtures (serde_json → types)
crates/cube-decode/tests/{state,decode,help,solves,orientation,pipeline,live,pixels,truth}.rs
~/TIMER/scripts/vision/fixtures.test.ts     générateur (ChronoCube)
```

Le générateur, lancé par `OUT=~/cubevision/crates READINGS=<dossier DUMP> npx vitest run --config
vitest.vision.config.mts scripts/vision/fixtures.test.ts` depuis `~/TIMER`, écrit :

| Fichier | Contenu |
|---|---|
| `cube-core/tests/fixtures/core.json` | `moves`, `inverts`, `cubes`, `orientations`, `goals` |
| `cube-decode/tests/fixtures/state.json` | `applyMove`, `scores`, `hashes`, `alike` (lectures aléatoires) |
| `cube-decode/tests/fixtures/random.json` | décodages de lectures simulées (avec trous et `moved: false`) |
| `cube-decode/tests/fixtures/solves.json` | depuis les `truth.json` de synthèse : reconstruction, présentation, aide, quarts |
| `cube-decode/tests/fixtures/pixels.json` | images RGBA tirées au hasard : vignettes, mouvement, miroir |
| `cube-decode/tests/fixtures/videos/<nom>.json` | images lues (`frames`), `scramble`, `truth`, et les sorties : `decode`, `result`, `score`, `live` |

Toute valeur aléatoire vient de mulberry32 avec une graine écrite dans le fichier. Les probabilités sont arrondies à
4 décimales avant de passer au TypeScript : l'entrée écrite est exactement celle qu'il a lue. Vidéos : 4 de
`synthetic-v3` (2 webm, 2 mp4, les plus courtes), < 4 Mo de fixtures au total.

---

### Task 1 : workspace et notation

**Files :**
- Create : `Cargo.toml`, `crates/cube-core/{Cargo.toml,src/lib.rs,src/notation.rs,tests/parity.rs}`
- Create : `~/TIMER/scripts/vision/fixtures.test.ts` (section `core.moves`, `core.inverts`)

**Interfaces :**
- Produces :
  - `enum Family { U, D, F, B, R, L, M, E, S, X, Y, Z }`, `enum Style { Face, Wide, Lower, Slice, Range }`
  - `struct Move { family: Family, from: u8, to: u8, turns: u8 /* 1..=3 */, style: Style }` (Clone, Copy, PartialEq, Debug)
  - `fn parse_move(token: &str) -> Option<Move>`, `fn format_move(m: &Move) -> String`
  - `fn tokenize(alg: &str) -> Vec<&str>`, `struct ParsedAlg { moves: Vec<Move>, invalid: Vec<String> }`,
    `fn parse_alg(alg: &str) -> ParsedAlg`, `fn format_alg(moves: &[Move]) -> String`, `fn invert_alg(alg: &str) -> String`

- [ ] **Step 1 : workspace.** `Cargo.toml` racine : `[workspace] members = ["crates/*"]`, `exclude = ["spikes"]`,
  `resolver = "3"`, `[workspace.package] edition = "2024"`, `license = "Apache-2.0"`. `cube-core` sans dépendance,
  `serde_json = "1"` en `[dev-dependencies]`.
- [ ] **Step 2 : générateur, section notation.** `fixtures.test.ts` écrit `core.json` avec `moves` : pour chaque jeton
  de la liste ci-dessous et de 300 jetons tirés au hasard (lettres, préfixes, `w`, nombres, `'`), `{ token, move:
  parseMove(token), formatted: move && formatMove(move) }` ; et `inverts` : `{ alg, inverse: invertAlg(alg) }` sur 50
  algorithmes aléatoires de 1 à 20 jetons. Jetons imposés : `R`, `U2'`, `R'2`, `R’`, `3Rw`, `2-3R`, `2-3Rw`, `r`, `M'`,
  `y2`, `x0`, `R4`, `R0`, `0R`, `3-2R`, `2M`, `Rw3`, `foo`, `(R`.
- [ ] **Step 3 : test qui échoue.** `tests/parity.rs::notation_matches_typescript` : pour chaque cas de `moves`,
  `parse_move(token)` égale `move` (null = `None`) et `format_move` égale `formatted` ; pour `inverts`, `invert_alg`.
  `tokenize("(R U) R'") == ["R", "U", "R'"]`.
  Run : `cargo test -p cube-core` → FAIL (fonctions absentes).
- [ ] **Step 4 : implémenter `notation.rs`.** Analyseur écrit à la main équivalent à la regex
  `^(?:(\d+)(?:-(\d+))?)?([UDFBRL]w?|[udfbrl]|[MESxyz])(\d*)('?)$`, après le remplacement des guillemets
  typographiques et de `R'2` en `R2'`, et les mêmes règles de rejet.
- [ ] **Step 5 :** `cargo test -p cube-core` → PASS. Commit « cube-core: notation, as ChronoCube parses it » (cubevision) et
  « Vision: fixtures for cubevision's Rust port » (ChronoCube, `scripts/vision/fixtures.test.ts` seul).

### Task 2 : le cube 3×3

**Files :** Create `crates/cube-core/src/cube.rs` ; Modify `tests/parity.rs`, `fixtures.test.ts` (`core.cubes`).

**Interfaces :**
- Consumes : `Move`, `parse_alg` (tâche 1).
- Produces : `const FACES: [char; 6] = ['U','R','F','D','L','B']` ; `struct Cube` (Clone, PartialEq, Eq, Hash, Debug) ;
  `Cube::solved()`, `apply(&self, alg: &str) -> Cube` (jetons inconnus ignorés), `apply_moves(&self, &[Move]) -> Cube`,
  `color(&self, face: usize, row: usize, col: usize) -> u8`, `cubie_colors(&self, face, row, col) -> Vec<u8>`,
  `stickers(&self) -> &[u8; 54]` (l'autocollant d'origine à chaque place), `colors(&self) -> [u8; 54]` (= `stateOf`).

- [ ] **Step 1 : générateur.** `cubes` : 300 algorithmes aléatoires (faces, `w`, minuscules, M E S, x y z, tranches
  `2R`, plages, et `4R`, `2-5Rw`) → `{ alg, stickers: Array.from(new StickerCube(3).apply(alg).stickers), cubie: [face,
  row, col, cubieColors] }`.
- [ ] **Step 2 : tests qui échouent.** `cube_matches_typescript` (stickers et `cubie_colors` de chaque cas) ;
  `layers_beyond_the_cube_turn_nothing` : `Cube::solved().apply("4R") == Cube::solved()` ;
  `unknown_tokens_are_skipped` : `apply("R foo U") == apply("R U")`. Run → FAIL.
- [ ] **Step 3 : implémenter.** Géométrie de `stickerCube.ts` pour n = 3 (positions doublées, rotations d'axe,
  permutations par mouvement calculées une fois dans un `OnceLock` par `(family, from, to, turns)`).
- [ ] **Step 4 :** `cargo test -p cube-core` → PASS. Commit « cube-core: the 3×3 sticker cube ».

### Task 3 : tenues et objectifs CFOP

**Files :** Create `crates/cube-core/src/goals.rs` ; Modify `tests/parity.rs`, `fixtures.test.ts` (`core.orientations`, `core.goals`).

**Interfaces :**
- Produces : `const ORIENTATIONS: [&str; 24]` (même ordre que `goals.ts`) ; `fn cross(cube: &Cube, neutral: bool) -> bool` ;
  `fn solved_slots(cube: &Cube, neutral: bool) -> u32` ; `fn cfop_progress(cube: &Cube) -> u32` (0 à 7).

- [ ] **Step 1 : générateur.** `goals` : pour chaque `truth.json` de `synthetic-v3`, le cube après le mélange puis après
  chaque jeton de la solution → `{ stickers, cross, crossNeutral, slots, slotsNeutral, progress }`.
- [ ] **Step 2 : test qui échoue.** `goals_match_typescript` ; `ORIENTATIONS == orientations`. Run → FAIL.
- [ ] **Step 3 : implémenter** (`Cube::from_stickers([u8; 54])` ajouté pour relire les fixtures, `pub`).
- [ ] **Step 4 :** PASS. Commit « cube-core: the 24 holds and CFOP progress ».

### Task 4 : lectures et score

**Files :** Create `crates/cube-decode/{Cargo.toml,src/lib.rs,src/reading.rs,src/state.rs,tests/common/mod.rs,tests/state.rs}` ;
Modify `fixtures.test.ts` (`state.json`).

**Interfaces :**
- Produces (`reading.rs`) :
  - `type State = [u8; 54]` ; `struct FaceReading { center: u8, stickers: [Option<[f32; 6]>; 9], rotation: Option<[f32; 4]> }`
  - `struct Reading { from: f64, to: f64, faces: Vec<FaceReading>, moved: Option<bool> }`
  - `enum Motion { NotMeasured, Unknown, Change(f64) }` (TS : `undefined`, `null`, nombre)
  - `struct FrameReadings { t: f64, faces: Vec<FaceReading>, orientation: Option<[f64; 9]>, turning: Option<f64>, motion: Motion }`
  - `struct TimedMove { token: String, t: f64 }`
  - `fn FaceReading::sanitized(self) -> FaceReading` : une case contenant un NaN devient `None`, une rotation avec NaN `None`.
- Produces (`state.rs`) : `const MOVES: [&str; 18]` ; `fn state_of(cube: &Cube) -> State` ; `fn apply_move(state: &State, mv: usize) -> State` ;
  `const ROT: [[usize; 9]; 4]` ; `struct Scorer` ; `const GARBAGE: f64 = 0.5` ; `fn scorer(reading: &Reading, misread: f64) -> Scorer` ;
  `fn score(state: &State, s: &Scorer, rotations: Option<&mut Vec<usize>>) -> f64` ;
  `fn reading_score(state: &State, reading: &Reading) -> (f64, Vec<usize>)` ; `fn hash_of(s: &State) -> u64` (la valeur
  JS, < 2^53) ; `fn alike(a: &Reading, b: &Reading, share: f64) -> bool` ;
  `struct SearchResult { moves: Vec<TimedMove>, path: Vec<(State, Vec<usize>)>, agreement: f64, solved: bool }`.
- Produces (`tests/common/mod.rs`) : `fn fixture(path: &str) -> serde_json::Value`, `fn reading(v: &Value) -> Reading`,
  `fn frame(v: &Value) -> FrameReadings`, `fn state(v: &Value) -> State`, `fn close(a: f64, b: f64) -> bool` (1e-9 relatif,
  `-inf` égal à `-inf`).

- [ ] **Step 1 : générateur.** `state.json`, graine écrite : 200 états (mélanges aléatoires) ; pour chacun une lecture de
  1 à 3 faces, cases cachées au hasard, probabilités bruitées autour de la vraie couleur, quelques 0 ; → `scores`
  (`score`, `rotations`), `hashes` (`hashOf`), `applyMove` (état, mouvement, résultat), `alike` (paires, `share` 0.5).
- [ ] **Step 2 : tests qui échouent.** `scores_match_typescript`, `hashes_match_typescript`, `moves_match_typescript`,
  `alike_matches_typescript` ; `nan_probabilities_count_as_hidden` : une case `[NaN; 6]` passée par `sanitized` donne le
  même score que la case `None`, sans panic ; `zero_probability_scores_minus_infinity_like_javascript`. Run → FAIL.
- [ ] **Step 3 : implémenter.** `scorer` garde les logs en `f32` (NaN = non vu), comme le `Float32Array` ; `hash_of`
  reproduit les deux passes FNV-1a avec `Math.imul` (`wrapping_mul` sur `i32`) et la combinaison du TypeScript.
- [ ] **Step 4 :** `cargo test -p cube-decode` → PASS. Commit « cube-decode: readings and their scores ».

### Task 5 : fixtures des vidéos de synthèse

**Files :** Modify `fixtures.test.ts` (`videos/`, `random.json`) ; Create `crates/cube-decode/tests/fixtures/videos/*.json`,
`crates/cube-decode/tests/fixtures/random.json` ; Modify `tests/common/mod.rs`.

**Interfaces :**
- Produces : `fn video_fixtures() -> Vec<(String, Value)>` (nom, contenu) dans `tests/common`.

- [ ] **Step 1 : lectures.** Liste `[{ video, scramble }]` des 4 vidéos (mélange lu dans leur `truth.json`) écrite dans le
  bloc-notes ; `DUMP=<bloc-notes>/dumps EVAL=<liste> npm run vision:eval` en arrière-plan (`setsid nohup … < /dev/null &`,
  `oom_score_adj` 1000, modèle `public/models`, CPU seulement). Expected : un `<vidéo>.json` par vidéo.
- [ ] **Step 2 : générateur.** Pour chaque dump : les probabilités arrondies à 4 décimales puis remises en
  `Float32Array`, et sur ces images-là : `decode` = `decodeSolve(holding.start, mergeReadings(frames).readings)`
  (`moves`, `agreement`, `solved`, `gaps`, `path[].state`), `result` = `analyzeReadings(frames, scramble)` entier,
  `score` = `scoreAgainstTruth(result, truth)`, `live` = l'état (`hashOf`), les mouvements et la confiance de
  `LiveTracker` après chaque image (départ : `holding.start`). `random.json` : 20 solves simulés (mélange aléatoire,
  solution de 15 à 25 quarts, 2 à 4 lectures par état, 100 ms entre lectures), dont 5 avec un trou de 4 à 8 mouvements
  sans lecture et 5 avec `moved: false` sur les lectures entre deux mouvements → entrées et `decodeSolve`.
- [ ] **Step 3 : test.** `tests/decode.rs::fixtures_load` : chaque vidéo a ≥ 100 images, un `scramble`, et
  `frame(...)` lit chaque image sans panic. Run → PASS (lecture seule).
- [ ] **Step 4 :** vérifier qu'aucun chemin ni nom de fichier des fixtures ne vient de `ml2/data` ou de `dump-v7`
  (`grep -rl "real-\|dump-v7" crates/` vide). Commit des fixtures (cubevision) et du générateur (ChronoCube).

### Task 6 : le décodeur

**Files :** Create `crates/cube-decode/src/decode/{mod,sequences,prepare,pass,bridge}.rs`, `tests/decode.rs`.

**Interfaces :**
- Consumes : `State`, `Reading`, `ROT`, `alike`, `SearchResult` (tâche 4).
- Produces :
  - `struct DecodeOptions { beam: usize, deep: usize, deeper: usize, move_cost: f64, slices: bool, slice_cost: f64,
    misread: f64, correlation: f64, fresh: f64, alike_share: f64, still: bool }`, `Default` = `{ 1024, 64, 8, 1.8,
    true, 2.2, 0.5, 300, 0.5, 0.5, true }`
  - `struct DecodeResult { search: SearchResult, gaps: Vec<f64> }`
  - `fn decode_solve(start: &State, readings: &[Reading], o: &DecodeOptions) -> DecodeResult`
  - `fn explain(start: &State, readings: &[Reading], moves: &[TimedMove], solved: bool, gaps: Vec<f64>) -> DecodeResult`
- Répartition : `sequences.rs` (suites de tours, `ZOBRIST` par mulberry32 de graine `0x9e3779b9`, la table `Table`),
  `prepare.rs` (`prepare`, `likelihood`, `bound`, `log_add_exp`), `pass.rs` (`Step`, le faisceau), `bridge.rs` (la
  jonction des trous jusqu'à huit tours), `mod.rs` (plan, poids, `join_passes`, `pick`, `decode_solve`).

- [ ] **Step 1 : tests qui échouent.** `random_decodes_match_typescript` (`random.json` : mouvements et temps exacts,
  `agreement` proche, `solved`, `gaps`) ; `video_decodes_match_typescript` (`decode` de chaque vidéo) ;
  `no_reading_gives_no_move` (`decode_solve` sur `[]` et sur des lectures sans case vue : aucun mouvement,
  `solved == (start == résolu)`) ; `a_long_video_decodes` (les lectures d'une vidéo répétées jusqu'à 3 000, temps
  décalés : finit, en release sous 60 s). Run : `cargo test -p cube-decode --test decode` → FAIL.
- [ ] **Step 2 : implémenter**, fichier par fichier, dans l'ordre du TypeScript. `Table` locale à chaque appel (le TS la
  partage entre appels : même résultat, rien de global entre fils). Les hashes de Zobrist en `i64` (sommes exactes,
  comme les doubles du TS). `decode_solve` lance le décodage filtré et le libre, et dans chacun l'aller et le
  retour, avec `rayon::join` : même résultat que `decodeSolve`.
- [ ] **Step 3 :** `cargo test -p cube-decode --release --test decode` → PASS. Un écart de mouvements sur une vidéo :
  chercher l'arrondi (`f32`, `ln_1p`) avant tout changement d'algorithme, et le noter dans le ledger.
- [ ] **Step 4 :** Commit « cube-decode: the decoder, forwards and backwards, gaps bridged ».

### Task 7 : les mouvements donnés par le cubeur

**Files :** Create `src/help.rs`, `tests/help.rs` ; Modify `fixtures.test.ts` (`solves.json` : `help`).

**Interfaces :**
- Produces : `fn help_moves(text: &str, hold: &str) -> Option<Vec<usize>>` ;
  `fn helped_solve(start: &State, readings: &[Reading], help: &[usize], decode: impl Fn(&State, &[Reading]) -> DecodeResult) -> DecodeResult`.

- [ ] **Step 1 : générateur.** `help` : pour chaque `truth.json`, les 1 à 8 premiers jetons de la solution, dans la tenue
  `''` et une autre → `helpMoves` ; plus `helpedSolve` sur la première vidéo avec ses 5 premiers jetons.
- [ ] **Step 2 : tests qui échouent** `help_moves_match_typescript`, `helped_solve_matches_typescript`. Run → FAIL.
- [ ] **Step 3 : implémenter.** **Step 4 :** PASS. Commit « cube-decode: moves the cuber vouches for ».

### Task 8 : reconstruction et écriture

**Files :** Create `src/reconstruct.rs`, `src/present.rs`, `tests/solves.rs` ; Modify `fixtures.test.ts` (`solves.json`).

**Interfaces :**
- Produces (`reconstruct.rs`) : `fn facelets(cube: &Cube) -> String`, `fn collapse(moves: &[String]) -> Vec<String>`,
  `enum Phase { Cross, F2l1, F2l2, F2l3, F2l4, Oll, Pll }`, `struct Split { phase, end, duration, recognition: f64, moves: u32, alg: String }`,
  `struct Reconstruction { moves: u32, tps: f64, time: f64, splits: Vec<Split>, solution: String, pause: f64, start: f64 }`,
  `const PAUSE: f64 = 500.0`, `fn reconstruct(start: &Cube, solve: &[TimedMove], written: Option<&[Vec<String>]>) -> Reconstruction`.
- Produces (`present.rs`) : `fn face_map(rotation: &str) -> [char; 6]` (lettre de U R F D L B), `fn cross_down(start: &Cube, moves: &[String], colour: Option<u8>) -> String`,
  `fn present_solve(start: &Cube, moves: &[String], cross: Option<u8>) -> Vec<Vec<String>>`, `fn leading_rotation(solution: &str) -> (String, String)`,
  `fn replay_of(scramble: &str, solution: &str, hold: &str) -> (String, String)` (setup, algorithme).

- [ ] **Step 1 : générateur.** Pour chaque `truth.json` : les tours de la solution en tours de faces (sans rotation)
  datés par `truth.moves[].end` → `reconstruct`, `presentSolve`, `crossDown`, `replayOf`, `collapse`.
- [ ] **Step 2 : tests qui échouent** `reconstructions_match_typescript`, `presentations_match_typescript`. Run → FAIL.
- [ ] **Step 3 : implémenter.** **Step 4 :** PASS. Commit « cube-decode: the reconstruction, written cross down ».

### Task 9 : orientation du cube

**Files :** Create `src/orientation.rs`, `tests/orientation.rs`.

**Interfaces :**
- Produces : `type Mat3 = [f64; 9]`, `type Quat = [f64; 4]` (w x y z) ; `fn snap(o: &Mat3) -> (usize, f64)` ;
  `struct OrientedFrame { t: f64, orientation: Mat3 }`, `struct HoldSpan { from: f64, hold: usize }` ;
  `fn hold_spans`, `fn cuber_notation`, `fn unflip`, `fn quat_of`, `fn slerp` aux signatures du TS ;
  `struct OrientationTrack { hz: f64, start: f64, q: String }` (base64 écrit à la main, sans dépendance) ;
  `fn encode_track(frames: &[OrientedFrame], reference: &Mat3) -> Option<OrientationTrack>` ;
  `struct OrientationResult { notation: Vec<Vec<String>>, track: Option<OrientationTrack>, rotations: u32 }` ;
  `fn cuber_view(frames: &[OrientedFrame], moves: &[TimedMove], scramble_hold: &str, camera: Option<&Mat3>) -> Option<OrientationResult>`.

- [ ] **Step 1 : test qui échoue.** `cuber_view_matches_typescript` : `result.orientation` de chaque vidéo, à partir de
  ses images orientées et de `result.timed` ramenés dans la tenue du mélange (comme `finish`), `q` identique.
  Run → FAIL.
- [ ] **Step 2 : implémenter.** **Step 3 :** PASS. Commit « cube-decode: the cuber's rotations from the cube's orientation ».

### Task 10 : de la vidéo lue au résultat

**Files :** Create `src/pipeline.rs`, `tests/pipeline.rs`.

**Interfaces :**
- Produces : `const MOTION: f64 = 0.02`, `const STILL_TURNING: f64 = 0.15` ; `fn merge_readings(frames: &[FrameReadings]) -> (Vec<Reading>, Vec<i32>)` ;
  `fn cancel_turns(moves: &[TimedMove]) -> Vec<TimedMove>` (itératif, pas récursif) ;
  `fn frame_agreement(start: &Cube, moves: &[TimedMove], frames: &[FrameReadings]) -> Vec<(f64, Option<f64>)>` ;
  `struct AnalyzeOptions { hold: Option<String>, camera: Option<Mat3>, decode: DecodeOptions, help: Option<String> }` ;
  `struct VisionResult { timed: Vec<TimedMove>, reconstruction: Reconstruction, agreement: f64, solved: bool,
  read_frames: usize, frames: usize, hold: String, orientation: Option<OrientationResult>, notation: Vec<Vec<String>>,
  gaps: Vec<f64>, frame_agreement: Vec<(f64, Option<f64>)> }` ;
  `fn analyze_readings(frames: &[FrameReadings], scramble: &str, o: &AnalyzeOptions) -> Option<VisionResult>`.

- [ ] **Step 1 : tests qui échouent.** `results_match_typescript` (tout `result` de chaque vidéo) ; `no_face_read_gives_none` ;
  `frames_at_one_time_or_going_back_do_not_panic` (une vidéo avec deux images au même `t` et une qui recule, comparée
  au cas ajouté par le générateur `videos/<nom>-shuffled.json`). Run → FAIL.
- [ ] **Step 2 : implémenter.** **Step 3 :** PASS. Commit « cube-decode: frames read to a reconstruction ».

### Task 11 : le suivi en direct

**Files :** Create `src/live.rs`, `tests/live.rs` ; Modify `fixtures.test.ts` (`ScrambleFollower` sur les mélanges).

**Interfaces :**
- Produces : `fn append_move(moves: &[usize], mv: usize) -> Vec<usize>`, `fn undo_moves(moves: &[usize]) -> Vec<usize>` ;
  `struct TrackerOptions { move_cost: f64 /* 4 */, beam: usize /* 24 */, jump_cost: f64 /* 12 */, bonus: Option<Box<dyn Fn(&State) -> f64 + Send>>, anchors: Option<Box<dyn Fn() -> Vec<(State, Vec<usize>)> + Send>> }` ;
  `struct LiveTracker` : `new(start: State, o: TrackerOptions)`, `reset(&mut self, State)`, `update(&mut self, &[FaceReading])`,
  `state() -> &State`, `moves() -> &[usize]`, `confidence() -> f64` ;
  `struct ScrambleProgress { done: f64, total: f64, holds: Vec<String> }`, `const SCRAMBLE_HOLD: &str = ""` ;
  `struct ScrambleFollower` : `new(scramble: &str, holds: &[&str])`, `way_back(&self, &State, after: f64) -> Option<(Vec<usize>, f64)>`,
  `locate(&self, &State) -> Option<ScrambleProgress>`, `written(&self, mv: usize, hold: &str) -> String`.

- [ ] **Step 1 : générateur.** `follower` : pour 5 mélanges, les états après chaque demi-jeton → `locate`, `wayBack`.
- [ ] **Step 2 : tests qui échouent** `tracker_matches_typescript` (`live` de chaque vidéo, image par image : hash de l'état,
  mouvements, confiance), `follower_matches_typescript`. Run → FAIL.
- [ ] **Step 3 : implémenter.** **Step 4 :** PASS. Commit « cube-decode: the live tracker and the scramble follower ».

### Task 12 : mouvement de l'image et miroir

**Files :** Create `src/motion.rs`, `src/mirror.rs`, `tests/pixels.rs` ; Modify `fixtures.test.ts` (`pixels.json`).

**Interfaces :**
- Produces : `struct Frame<'a> { data: &'a [u8] /* RGBA */, width: usize, height: usize }` ; `fn thumbnail(f: &Frame) -> [f32; 576]` ;
  `struct MotionMeter` : `measure(&mut self, crop: Option<&Frame>, t: f64) -> Option<f64>` ;
  `fn flip_frame(f: &Frame) -> Vec<u8>` ; `struct MirrorCheck` : `new(scramble: &str, hold: Option<&str>)`, `lead() -> f64`,
  `add(&mut self, normal: &[FaceReading], flipped: &[FaceReading]) -> Option<bool>`, `mirrored: Option<bool>`.

- [ ] **Step 1 : générateur.** 10 images RGBA de 37×53 et 64×64 tirées au hasard, mesurées en suite → `thumbnail`,
  `measure`, `flipFrame` (somme de contrôle) ; `MirrorCheck` sur les 30 premières images de la première vidéo, faces
  telles quelles et retournées.
- [ ] **Step 2 : tests qui échouent** `motion_matches_typescript`, `mirror_matches_typescript`. Run → FAIL.
- [ ] **Step 3 : implémenter.** **Step 4 :** PASS. Commit « cube-decode: crop motion and mirrored videos ».

### Task 13 : comparaison à la vérité et bilan

**Files :** Create `src/truth.rs`, `tests/truth.rs` ; Modify `README.md` (feuille de route, structure).

**Interfaces :**
- Produces : `enum Stage { Cross, F2l, Oll, Pll }` ; `struct Truth { scramble, solution: String, moves: Vec<(String, f64, f64, String)>, stages: [f64; 4] }` ;
  `struct Quarter { face: char, dir: i8, t: f64 }` ; `fn quarters(moves: &[TimedMove], hold: &str) -> Vec<Quarter>` ;
  `fn align(a: &[Quarter], b: &[Quarter]) -> (usize, Vec<(Quarter, Quarter)>)` ; `fn rotations_in(solution: &str) -> usize` ;
  `struct TruthScore { quarters: usize, distance: usize, exact: f64, time_median: Option<f64>, stages: [Option<f64>; 4], rotations: (usize, Option<usize>) }` ;
  `fn score_against_truth(result: &VisionResult, truth: &Truth) -> TruthScore`.

- [ ] **Step 1 : test qui échoue** `scores_match_typescript` (`score` de chaque vidéo, sur le `VisionResult` calculé en
  Rust). Run → FAIL.
- [ ] **Step 2 : implémenter.** **Step 3 :** `cargo test --workspace --release` → tout PASS ; `cargo clippy --workspace
  --all-targets -- -D warnings` → aucun avertissement.
- [ ] **Step 4 : README.** Feuille de route : étape 2 faite ; structure : `crates/cube-core`, `crates/cube-decode` ;
  comment régénérer les fixtures. Commit « cube-decode: scored against the truth of synthetic solves », puis push
  (dépôt public, adresse vérifiée).
