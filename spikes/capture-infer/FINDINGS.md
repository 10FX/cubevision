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
| P4 Webcam | 1280×720 MJPEG à 30 i/s annoncés : 15,1 i/s reçus (0,50), autant que ffmpeg DirectShow (15 i/s) ; lecture en direct 20 ms p50, 27 ms p95 | `results/cam-windows.json`, `results/cam-windows-cuda.json` | réussi sous réserve |

P4 : la SKILLKORP SKP_W-10 descend à 15 i/s dans la lumière de la pièce, à 1280×720 comme à 640×480.
Rust attend la caméra 60 ms par image et décode en 3,8 ms : il ne perd aucune image par rapport au
lecteur natif. Les 30 i/s restent à mesurer en bonne lumière (même commande). Le `found_share` à 0 de
la mesure avec modèles vient de l'absence de cube devant la caméra.

Le CPU seul tient aussi P2 (15,6 ms Windows, 18,0 ms WSL), mais pas les lots : 8,4 ms par image en
lot de 8, contre 1,1 ms sur CUDA.

## Replis essayés

- P2 : DirectML sous Windows marche (12,7 ms p95) mais seulement sur ONNX Runtime 1.24.4, sa dernière
  version ; en lot de 8 il coûte 9,4 ms par image (rien gagné), à creuser avant d'en faire un moteur.
- P4 : `ffmpeg -f dshow` ne fait pas mieux (15 i/s) : la limite est la caméra, pas la capture.
- P3 : image par image, le 1080p60 plafonnait à 40 i/s ; les lots de 8 l'ont mené à 120 i/s.

## À reprendre dans `cube-infer`

- `ort =2.0.0-rc.15`, `load-dynamic`, feature `api-24` (pour charger aussi DirectML 1.24.4).
  Un seul `Environment` par processus ; `error_on_failure()` sur le moteur demandé, jamais de repli CPU muet.
- ONNX Runtime : CUDA 1.31.1 (`onnxruntime-gpu[cuda,cudnn]`, CUDA 13) sous WSL et Windows,
  `ORT_DYLIB_PATH` vers la bibliothèque ; DirectML 1.24.4 à part (`setup-ort.sh`, `setup-ort.ps1`).
- Sessions GPU : 1 fil intra-op sans attente active (5 cœurs occupés tombent à 1, même vitesse).
- Une forme d'entrée fixe par session CUDA : changer de taille de lot d'un appel à l'autre coûte
  ~100 ms ; les lots vidéo sont donc toujours complétés à 8.
- La seconde vue zoomée du site (`detectWithZoom` : cube < 1/4 de l'image, zoom ×4) est
  indispensable en portrait : 11 % de cubes trouvés sans elle, 99,2 % avec.
- Mode vidéo par lots de 8 (même résultat qu'image par image, test à l'appui) ; image par image en direct.

## À reprendre dans `cube-capture`

- Vidéo : `ffmpeg`/`ffprobe` en sous-processus, rawvideo rgb24, `-fps_mode passthrough`, rotation lue
  dans les side data (la taille est inversée à 90°/270°), dernière image incomplète ignorée.
- Webcam : `nokhwa =0.10.11` (Media Foundation) en MJPEG, dans son propre fil, seule la dernière
  image gardée. Sa cadence rapportée est fausse (dénominateur lu : 30/1 donne 1) : la prendre du
  format demandé. Windows partage la caméra : une application déjà dessus n'empêche pas l'ouverture.

## Verdict

Rust confirmé. Les deux modèles tournent sur le RTX sous WSL et Windows avec les sorties de Python,
la vidéo va plus vite que le temps réel, et la webcam livre en Rust tout ce qu'elle produit. Seule
réserve : refaire la mesure webcam en bonne lumière pour voir les 30 i/s.
