# cubevision : design

Date : 2026-11-10. Statut : à relire par le propriétaire.

## 1. Objectif

Un dépôt GitHub open source, séparé de ChronoCube, qui montre toute la vision du Rubik's Cube : détection,
lecture des faces, déduction des mouvements, reconstruction, cube 3D. Mêmes algorithmes que le site, réécrits
en natif pour tourner plus vite sur PC.

C'est une **vitrine technique** : on la clone, on la compile, on voit ce qu'elle fait et comment. Pour un vrai
usage au quotidien, le README renvoie vers ChronoCube.

**Réussi si :**
- le mode direct écrit les mouvements d'un vrai cube filmé à la webcam, à la cadence de la caméra ;
- le mode vidéo redonne au moins la qualité du site, plus vite que lui, sur les mêmes vidéos ;
- les tests prouvent que les algorithmes portés donnent les mêmes résultats que le TypeScript ;
- le README montre tout cela par des démos et des chiffres mesurés.

**Hors périmètre :** autres cubes que le 3x3, chrono, comptes, cloud, lien de code ou de données avec
ChronoCube, installeur, inspecteur image par image et graphiques d'analyse du site (ajouts possibles plus
tard). ChronoCube ne change pas.

## 2. Les deux modes

Ils sont indépendants : le direct ne prépare pas la vidéo, et inversement.

### Direct : `cubevision live`

Pour tester et voir que ça marche.

- Départ : cube résolu, blanc en haut, vert devant.
- Chaque mouvement s'écrit avec **10 images traitées de décalage** (0,33 s à 30 images/s, 0,17 s à 60). Seuls
  les mouvements confirmés sont affichés : le suivi garde ses hypothèses, celle en tête au bout de 10 images
  est figée, et les hypothèses qui la contredisent sont retirées. Une liste affichée ne change plus.
- Les faces sont nommées par la couleur de leur centre, donc les mouvements sont écrits blanc en haut, vert
  devant, quelle que soit la tenue du cube.
- Limite héritée du suivi : un M s'écrit R L', et un Rw s'écrit L, car une tranche ou un large sont vus comme
  des mouvements de faces.
- Le cube 3D joue les mouvements confirmés, fixe, blanc en haut, vert devant.
- Une touche remet à zéro (nouveau départ d'un cube résolu).
- Pas de mélange, pas de chrono, pas de reconstruction.

### Vidéo : `cubevision video solve.mp4 --scramble "…"`

Pour ceux qui veulent aller plus loin. Les mêmes champs existent dans la fenêtre.

| Entrée | Valeur |
|---|---|
| Vidéo | fichier, obligatoire |
| Mélange | notation WCA, obligatoire |
| Couleurs du mélange | `--top`, `--front` ; blanc et vert par défaut |
| Couleur de la croix | `--cross` ; détectée si absente, comme `crossDown` sur le site |

Sorties :

- tous les mouvements, appliqués au cube 3D : couleurs WCA, croix en bas dès le premier mouvement,
  gyroscopique (suit l'orientation mesurée dans la vidéo), déplaçable à la souris ;
- une barre de lecture synchronise cube, vidéo et mouvements (lecture, mouvement précédent et suivant,
  vitesses 0,25 à 1) ;
- les informations : rotations, mouvements larges et tranches, étapes CFOP et leurs temps, TPS, part des
  autocollants lus qui collent à la solution, images lues, passages non suivis ;
- `--json` : le même résultat sans ouvrir de fenêtre.

Les vidéos filmées en miroir sont détectées (`mirror.ts`).

### Autres commandes

- `cubevision bench` : mesure chaque étape et imprime un tableau reproductible.
- `cubevision eval` : compare une reconstruction à la vérité (vidéos de synthèse, `truth.ts`).
- `cubevision fetch-models` : télécharge les modèles depuis les GitHub Releases.

## 3. Architecture

Un workspace Rust. Chaque crate a un seul rôle et ne connaît pas les suivantes.

| Crate | Rôle | Portage depuis `src/` de ChronoCube |
|---|---|---|
| `cube-core` | État du cube, notation, 24 tenues, objectifs CFOP. Aucune dépendance | `lib/cube/stickerCube`, `notation`, `goals` |
| `cube-decode` | Des lectures aux mouvements, écriture de la solution | `lib/vision/search`, `decode`, `pipeline` (partie `analyzeReadings`), `live`, `orientation`, `motion`, `mirror`, `truth` ; `lib/smartcube/analysis`, `present` |
| `cube-infer` | Modèles ONNX, choix du matériel, suivi de la boîte, recadrage | `lib/vision/detector`, `model` |
| `cube-capture` | Images depuis la caméra ou un fichier vidéo | remplace le navigateur |
| `cube-view` | Cube 3D en wgpu, interface | `components/cube/RubiksCube3D`, `Speedcube`, `reconstruct/ReplayBar` |
| `cubevision` | L'exécutable et ses commandes | `lib/vision/liveWorker`, `analyzeVideo`, `scripts/vision/eval.test` |

`cube-core` et `cube-decode` ne dépendent ni d'ONNX, ni de la caméra, ni du GPU : ils se testent seuls.

`training/` reprend `ml2/` en Python : rendu Blender, entraînement, export, tests. Sans `data/` ni `runs/`.

**Non porté :** le détecteur classique sans IA (`stickers`, `color`, `roi`, `searchSolve` côté classique), qui
ne servait que sans modèles ; le choix de backend du navigateur (`plan`, `warm`, `passes`, `passWorker`), remplacé par
§5 et par des threads.

## 4. Chemin des données

```
capture → décodage → détecteur → recadrage → lecteur → lectures
                                                          │
                     direct : suivi, 10 images de décalage┤
                     vidéo  : tout lu, puis décodage ─────┘→ résultat → affichage
```

Chaque étape tourne dans son thread, reliée à la suivante par un canal borné.

- **Direct** : le canal garde la dernière image. Si le GPU prend du retard, les plus anciennes sont jetées.
- **Vidéo** : le canal bloque. Aucune image n'est perdue, la lecture ralentit.
- Les images du lecteur sont envoyées au GPU par lots en vidéo, à l'unité en direct.
- Le décodage vidéo répartit ses passes (aller et retour, avec et sans filtre) sur tous les cœurs, avec rayon.
- Le détecteur ne regarde pas chaque image quand le cube est suivi, comme sur le site (une image sur 3).

## 5. Modèles et matériel

**Livraison en deux temps :**
1. les modèles actuels du site, inchangés : détecteur SSDLite320-MobileNetV3 (320 px) et lecteur MobileNetV4
   conv medium (224 px), en variantes `full` (GPU) et `lite` (CPU). Ils donnent la référence face au site ;
2. un lecteur PC plus gros, réentraîné sur les données existantes et publié à côté, avec son gain mesuré. Il ne
   commence qu'une fois le premier temps livré.

On entraîne une fois, sur la RTX. Le même fichier ONNX tourne partout : seul le moteur d'exécution change.

| Matériel | Windows | Linux | Testé par nous |
|---|---|---|---|
| NVIDIA RTX | CUDA | CUDA | oui (RTX 5060 Ti) |
| Processeur Intel | OpenVINO ou CPU | OpenVINO ou CPU | oui (i5-12400F) |
| GPU Intel | OpenVINO | OpenVINO | à confirmer selon la machine disponible |
| AMD | DirectML | MIGraphX | non |

Le moteur est choisi automatiquement, affiché dans l'application, et forcé avec `--device`. Le README dit
explicitement ce qui n'a pas été testé.

`ort` (crate Rust, chargement dynamique d'ONNX Runtime) et ONNX Runtime sont figés sur des versions testées.
ONNX Runtime 1.23 est exclu (bug connu sur les RTX 50xx).

## 6. Performance

Objectifs, à vérifier par la mesure ; aucun chiffre n'est annoncé avant :

- direct : tenir la cadence de la caméra sur la RTX 5060 Ti ;
- vidéo : plus rapide que le temps réel, et plus rapide que le site sur les mêmes vidéos.

`cubevision bench` mesure capture, détecteur, lecteur, suivi et décodage, par matériel. La référence web est
mesurée une fois dans Chrome sous Windows, sur les mêmes fichiers.

## 7. Cas d'erreur

| Situation | Comportement |
|---|---|
| Pas de cube visible | le cadre disparaît, le suivi attend |
| Suivi perdu en direct | message et touche de remise à zéro |
| Reconstruction non résolue | mouvements gardés, passages manquants signalés |
| Le mélange ne colle pas à la vidéo | message explicite, pas de résultat inventé |
| Modèles absents | indique `cubevision fetch-models` |
| Pas de GPU utilisable | CPU avec le modèle `lite`, et un avertissement |
| Vidéo illisible, caméra occupée | message d'erreur clair, code de sortie non nul |

## 8. Tests

- **Parité avec le TypeScript :** un script dans ChronoCube génère des fixtures à partir de vraies lectures
  (`ml2/runs/dump-v7`, 3 vidéos, 1,8 Mo, lectures seulement, sans images) et de cas aléatoires. Les tests Rust
  retrouvent les mêmes états, mouvements et scores.
- **Unitaires** dans chaque crate.
- **Décalage du direct :** un mouvement confirmé n'est jamais modifié ensuite.
- **Bout en bout :** une courte vidéo de synthèse donne les mouvements attendus (`solve_video.py`).
- **CI GitHub :** Linux et Windows, CPU uniquement, plus `training/` (pytest, ruff).

## 9. Le dépôt publié

- Licence Apache-2.0 pour le code et les modèles.
- README en anglais pour toucher le public open source : GIF, schéma, tableau web contre natif, explication de
  l'entraînement, lien vers ChronoCube.
- Modèles dans les GitHub Releases, pas dans git.
- **Entraînement, à décrire honnêtement :** le lecteur actuel (v7c) est entraîné sur des rendus de synthèse **et** sur
  environ 260 images issues des vraies vidéos du propriétaire, labellisées automatiquement (`data/real-harvest`).
  Il n'est donc pas « 100 % synthétique ».
- **Ne sont pas publiés :** les vidéos réelles et `data/real-harvest`, sauf choix explicite du propriétaire ; les
  images COCO (licences Flickr variées), qu'un script télécharge pour qui veut réentraîner. Sans `real-harvest`, un
  réentraînement ne reproduit pas exactement les poids publiés, et le README le dit.
- Les démos utilisent des vidéos de synthèse.
- Rien n'est publié sur GitHub sans l'accord explicite du propriétaire au moment de le faire.

## 10. Ordre de livraison

1. **Prototype jetable (porte d'entrée) :** capture webcam et décodage vidéo en Rust, les deux modèles sur la
   RTX. Il mesure et lève les deux risques ci-dessous. Compilé aussi sous Windows, car WSL2 ne voit pas la webcam par
   défaut.
2. `cube-core`, puis `cube-decode`, validés par les fixtures.
3. `cube-infer` et `cube-capture`.
4. Mode vidéo en JSON, et comparaison au site.
5. `cube-view` : cube 3D, mode direct, replay.
6. `training/`, `bench`, CI, README, publication.
7. Lecteur PC plus gros.

**Risques traités par l'étape 1 :**
- capture caméra et vidéo en Rust moins mûre qu'en C++ ; repli envisagé : FFmpeg ou OpenCV liés au binaire ;
- ONNX Runtime CUDA sur RTX 50xx ; replis envisagés : autre version figée, DirectML sous Windows.

Si l'un des deux échoue sans repli acceptable, le choix de Rust est rediscuté avant d'aller plus loin.
