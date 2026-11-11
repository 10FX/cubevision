# Prototype capture + inférence : constats

Prototype jetable (§10, étape 1 de la spec). Machine : i5-12400F, RTX 5060 Ti 16 Go (pilote 621.01),
Windows 11 et WSL2. Modèles : `reader-v7/web` (détecteur 320, lecteur 224), fp16 sur GPU, fp32 sur CPU.
Vidéos : uniquement les synthétiques `synthetic-v3` ; aucune image de la webcam n'est enregistrée.

## Portes

| Porte | Mesure | Preuve | Résultat |
|---|---|---|---|
| P1 Parité | CPU fp32 ≤ 1e-4 et CUDA fp16 ≤ 1e-2 sur toutes les sorties, contre ONNX Runtime 1.31 en Python | `tests/parity.rs`, `tests/reference.json` | réussi |
| P2 GPU | total p95 par image : CUDA 15,8 ms (WSL), 13,8 ms (Windows), DirectML 12,7 ms ; seuil 33 ms | `results/infer-*.json` | réussi |
| P3 Vidéo | 1080p60 h264 : 120 i/s, cube trouvé sur 100 % ; 480×854 vp8 : 46 i/s pour 30, 99,2 % | `results/video-mp4.json`, `results/video-webm.json` | réussi |
| P4 Webcam | 1280×720 MJPEG à 30 i/s annoncés : 15,1 i/s reçus (0,50), autant que ffmpeg DirectShow (15 i/s) | `results/cam-windows.json`, `results/cam-windows-cuda.json`, `results/cam-windows-ffmpeg-dshow.txt` | échoué selon la règle, limité par la caméra ; non concluant à 30 i/s |

P4 : la SKILLKORP SKP_W-10 ne livre que 15 i/s, à 1280×720 comme à 640×480, à nokhwa comme à
ffmpeg DirectShow. Rust attend la caméra 60 ms par image et décode en 3,8 ms : il ne perd aucune image
par rapport au lecteur natif. Les formats MJPEG de la caméra sont à cadence fixe (5 à 30 i/s, chacun
annoncé seul). La cause probable, une pose allongée en faible lumière, n'a pas été vérifiée : les
30 i/s restent à mesurer en bonne lumière (même commande).

Lecture en direct sur CUDA : 20 ms p50, 27 ms p95, mais sans cube devant la caméra (`found_share` 0).
Ce chiffre ne comprend donc pas la seconde vue zoomée (un passage du détecteur et un letterbox de
plus) qu'un petit cube déclenche ; avec elle, le p95 approcherait le budget de 33 ms (estimation non
mesurée). À remesurer avec un cube.

Le CPU seul tient aussi P2 (15,6 ms Windows, 18,0 ms WSL), mais pas les lots : 8,4 ms par image en
lot de 8, contre 1,1 ms sur CUDA.

## Seconde machine : portable Intel

Intel Core Ultra 5 125H, Intel Arc intégré, 15,5 Go, Windows 11, avec le zip de la release `v0.1.0-prototype`
(`results/laptop-*.json`).

- DirectML sur l'Arc : total p95 44,3 ms (détecteur 13,1, lecteur 32,1), au-dessus des 33 ms de P2. En lot de 8,
  le lecteur passe à 13,8 ms par image : le mode vidéo y tient, le direct image par image non.
- CPU : 100,8 ms p95.
- Webcam intégrée « FHD Camera », 1280×720 MJPEG à 30 i/s annoncés : 10,5 i/s reçus, une image toutes les 96 ms,
  très régulières (pose longue probable, mesure vers 2 h du matin, lumière non vérifiée).
- Direct avec un cube et la seconde vue : 45,7 ms p50, 49,1 ms p95, cube trouvé sur 79 % des images, aucune
  image reçue perdue. Le portable suivrait ~20 i/s.
- À essayer : OpenVINO (`--device openvino-gpu`, `openvino-npu`), le moteur prévu pour Intel par la spec.

## Replis essayés

- P2 : DirectML sous Windows marche (12,7 ms p95) mais seulement sur ONNX Runtime 1.24.4, sa dernière
  version ; en lot de 8 il coûte 9,4 ms par image (rien gagné), à creuser avant d'en faire un moteur.
- P4 : `ffmpeg -f dshow` ne fait pas mieux (15 i/s) : la limite est la caméra, pas la capture.
- P3 : image par image, le 1080p60 plafonnait à 40 i/s ; les lots de 8 l'ont mené à 120 i/s.
  Ce gain vaut pour un cube trouvé dès la première vue. Sur la webm portrait, où le cube est petit,
  la première vue le manque sur ~89 % des images : chaque série d'échecs est coupée au premier zoom
  réussi et refaite, si bien que le détecteur tourne plus de fois en lots (101 lots de 8 pour
  96 images, contre 189 passages unitaires). Là, les 46 i/s viennent du lecteur en lot et des
  letterbox parallèles, pas du détecteur. Une vidéo portrait 1080×1920 n'a pas été mesurée.

## À reprendre dans `cube-infer`

- `ort =2.0.0-rc.15`, `load-dynamic`, feature `api-24` (pour charger aussi DirectML 1.24.4).
  Un seul `Environment` par processus ; `error_on_failure()` sur le moteur demandé, jamais de repli CPU muet.
- ONNX Runtime : CUDA 1.31.1 (`onnxruntime-gpu[cuda,cudnn]`, CUDA 13) sous WSL et Windows,
  `ORT_DYLIB_PATH` vers la bibliothèque ; DirectML 1.24.4 à part (`setup-ort.sh`, `setup-ort.ps1`).
- Sessions GPU : 1 fil intra-op sans attente active (5 cœurs occupés tombent à 1, même vitesse).
- Une forme d'entrée fixe par session CUDA : changer de taille de lot d'un appel à l'autre coûte
  ~100 ms ; les lots vidéo sont donc toujours complétés à 8.
- La seconde vue zoomée du site (`detectWithZoom` : cube < 1/4 de l'image, zoom ×4) est
  indispensable en portrait : 11 % de cubes trouvés sans elle (version intermédiaire, mesure non conservée), 99,2 % avec.
- Mode vidéo par lots de 8 (même résultat qu'image par image, test à l'appui) ; image par image en direct.
  Pour les petits cubes, prévoir une session détecteur par forme fixe (1 et 8) afin que les vues
  isolées ne paient pas le changement de forme, et un chemin séquentiel des échecs qui ne refait pas
  le reste de la série.
- Chauffer les sessions avant toute mesure : les modes `video` et `cam` du prototype ne le font pas,
  leurs premiers passages CUDA pèsent sur les moyennes (prudent, mais trompeur sur une vidéo courte).
- OpenVINO (`--device openvino-cpu|gpu|npu`) : ONNX Runtime 1.24.1, avec OpenVINO 2025.4.1 à installer à part
  sous Windows (`ort\ov`). `openvino-cpu` donne les sorties de Python à 1e-3 près, 16,0 ms p95 sur l'i5-12400F
  (15,6 ms pour le moteur CPU). Sur la carte NVIDIA, que son plugin GPU voit par OpenCL, le lecteur fp16 ne
  compile pas (conversion fp16 sans noyau) : OpenVINO part donc des modèles fp32 et calcule lui-même en fp16 sur
  GPU et NPU. Pas encore mesuré sur un vrai GPU Intel ni sur un NPU.

## À reprendre dans `cube-capture`

- Vidéo : `ffmpeg`/`ffprobe` en sous-processus, rawvideo rgb24, `-fps_mode passthrough`, rotation lue
  dans les side data (la taille est inversée à 90°/270°), dernière image incomplète ignorée.
- Webcam : `nokhwa =0.10.11` (Media Foundation) en MJPEG, dans son propre fil, seule la dernière
  image gardée. Sa cadence rapportée est fausse (dénominateur lu : 30/1 donne 1) : la prendre du
  format demandé. Windows partage la caméra : une application déjà dessus n'empêche pas l'ouverture.

## Verdict

Rust confirmé pour l'inférence et la vidéo (P1 à P3). Pour la webcam, Rust reçoit tout ce que la
caméra produit, autant que ffmpeg ; mais P4 n'est pas tranché à 30 i/s, faute d'une caméra qui les
livre ici. Avant `cube-capture` : refaire `cam` en bonne lumière, un cube devant la caméra, ce qui
mesure aussi le coût réel du direct avec le zoom.
