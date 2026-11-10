# Leg / arm detectors: results

Trained with `scripts/train_limbs.sh` (YOLO11s, 960 px, mirroring off, Mac M4 Pro MPS, ~25 min per version).
Data: 184 train + 47 val frames from the Junior Worlds 2026 broadcasts (val = last 20% of each
broadcast, different swimmers). Weights in `experiments/limbs/output/<version>/weights/best.pt` (not in git).

## Val (47 frames, 83 boxes), 2026-11-09

| Version | mAP50 | mAP50-95 | Precision | Recall | Best epoch |
|---------|-------|----------|-----------|--------|------------|
| 4 classes (left/right leg, left/right arm) | 0.344 | 0.179 | 0.44 | 0.33 | 40 of 70 |
| 2 classes (leg, arm) | **0.545** | **0.228** | 0.51 | 0.60 | 60 of 90 |

Per class (mAP50): 4 classes: left_leg 0.44, right_leg 0.54, left_arm 0.16, right_arm 0.23.
2 classes: leg **0.71**, arm 0.38.

Side correct when the 4-class model found the limb (confusion matrix): legs 14 of 17 (82%),
arms 31 of 58 (53%, a coin flip).

**Reading:** the detector finds legs reasonably well; arms are much harder. Left/right works
for legs but not for arms, which suggests taking the side from the pose model inside the box
(milestone 2 step 3) rather than from the detector. Val is small (83 boxes), so these numbers
are rough.

## Test, 2026-11-09

95 of 97 test frames labeled (2 Paris 2025 frames not done). New pools = Doha 2024 + Paris 2025/2026
(72 frames); Budapest 2024 (23 frames) is the same city as training, scored separately.

mAP50:

| | Val (training pool) | Budapest (same city) | **New pools** |
|---|---|---|---|
| 2 classes: all | 0.55 | 0.47 | **0.23** |
| 2 classes: leg | 0.71 | 0.71 | **0.37** |
| 2 classes: arm | 0.38 | 0.23 | **0.09** |
| 4 classes: all | 0.34 | 0.26 | **0.09** |

New pools by video (2 classes): Doha 0.25, Paris 2025 0.20, Paris 2026 0.29 (all similar).
Side correct on new pools (4 classes): legs 20 of 42, arms 17 of 34 (both a coin flip).

**Reading:**
- The model doesn't carry over to new pools: legs drop from 0.71 to 0.37, arms almost vanish.
  Budapest (also a close-up fan video, like the new-pool test videos) scores like the training
  pool, so the drop is about the pool, not about close-ups vs broadcast shots.
- On new pools the errors are mostly **misses** (recall 0.29), not invented limbs.
- Left/right from the detector is a coin flip on new pools, for legs too. Supports detecting
  "leg"/"arm" and taking the side from the pose model inside the box.
- All training frames come from one pool (Junior Worlds 2026, Budapest). More pools in training
  is the most likely fix; stronger colour augmentation is a cheap thing to try first.

## Stronger colour augmentation, 2026-11-09

Same as 2 classes above, plus `hsv_h=0.05 hsv_s=0.9 hsv_v=0.6` (defaults 0.015 / 0.7 / 0.4),
to cover the measured water colour differences (Paris is much duller and darker, Doha bluer):

```bash
yolo detect train data=data/raw/datasets/m2_2cls/data.yaml model=yolo11s.pt imgsz=960 epochs=100 \
  patience=30 batch=8 fliplr=0.0 seed=0 device=mps hsv_h=0.05 hsv_s=0.9 hsv_v=0.6 \
  project=$PWD/experiments/limbs/output name=2cls_color
```

| mAP50 | 2 classes | + stronger colour |
|---|---|---|
| Val (training pool) | 0.55 | 0.55 |
| Budapest | 0.47 | 0.41 |
| **New pools** | **0.23** | **0.21** |
| New pools: leg / arm | 0.37 / 0.09 | 0.33 / 0.10 |
| Doha / Paris 2025 / Paris 2026 | 0.25 / 0.20 / 0.29 | 0.24 / 0.19 / 0.29 |

**Reading:** no gain (the differences are within noise for 72 test frames and one training run).
Water colour alone is not what separates the pools; the rest of the look (camera height and
angle, lighting, reflections, water texture, swimsuits) matters too. More pools in the training
data remains the main fix.

## Batch 2: more pools in training, 2026-11-10

Added 141 labeled frames from 4 new pools (Fukuoka 2023, Markham 2025, Rome 2022, Funchal 2025;
see docs/videos.md): train 296 (184 Budapest + 112 new) + val 76. Same settings
(`scripts/train_limbs.sh`, ~30 s per round, ~50 min per version). Batch-1 models kept as
`*_batch1`. "Training pool" = the 47 Budapest val frames only, so it is the same set for both.

mAP50, 2 classes:

| | Batch 1 (1 pool) | **Batch 2 (5 pools)** |
|---|---|---|
| Training pool (Budapest val) | 0.55 | 0.60 |
| Budapest 2024 test | 0.47 | 0.46 |
| **New pools (Doha + Paris)** | **0.23** | **0.40** |
| New pools: leg | 0.37 | **0.56** |
| New pools: arm | 0.09 | **0.24** |
| Doha 2024 | 0.25 | 0.40 (leg 0.66) |
| Paris 2025 | 0.20 | 0.28 |
| Paris 2026 | 0.29 | 0.60 (leg 0.76) |

4 classes: new pools 0.09 -> 0.24, training pool 0.34 -> 0.46. Side correct on new pools:
legs 27 of 54 (50%, still a coin flip), arms 23 of 33 (70%).

**Reading:**
- More pools is the fix: new-pool mAP50 up from 0.23 to 0.40 with 112 extra frames, and the
  training pool didn't get worse (0.55 -> 0.60).
- Doha's legs gain may be helped by a swimmer overlap: Platanioti (the Doha test swimmer)
  probably also swims in the Fukuoka final used for training. Paris 2026 has no overlap and
  gained the most (0.29 -> 0.60).
- Arms are still weak (0.24), and legs' left/right from the detector is still a coin flip on new
  pools, so the sides should come from the pose model.
- One training run per setting and 72 test frames: differences under ~0.05 are noise.

## Step 4: leg/arm outlines (YOLO11s-seg), 2026-11-10

SAM 2.1 outlined every labeled box (867 limbs, none failed the automatic check), so the
training outlines cost no labeling. YOLO11s-seg trained on them with the same settings and split
(`scripts/train_limbs.sh seg`, 100 rounds, ~68 min; still improving at round 100).

**Outline score (mask mAP50):**

| | Frames | Mask mAP50 | Leg | Arm |
|---|---|---|---|---|
| Val (all pools) | 76 | 0.62 | 0.62 | 0.61 |
| **New pools (Doha + Paris)** | 41 | **0.62** | 0.69 | 0.55 |
| Budapest 2024 | 15 | 0.87 | 0.99 | 0.75 |

Test = the reviewed test frames: 56 of 95 accepted, 39 skipped (an outline wrong or missing;
Doha 13 of 24 skipped, the others 8-9 each). **The test score is optimistic**: the skipped frames
are the harder ones (the detector's box mAP50 on new pools is 0.41 on the accepted frames vs
0.33 on the skipped), and the outlines are scored against SAM's outlines, which is what the
model learned to copy. So read 0.62 as "copies SAM well on new pools where SAM works", not as
"outlines 62% right on any frame".

**Boxes, compared fairly.** The seg model's own box score is measured against boxes around the
SAM outlines, not the hand-drawn boxes, so it can't be compared with the detector directly.
`experiments/limbs/box_vs_hand.py` scores both models against the same hand-drawn boxes
(box mAP50; this scorer is simpler than Ultralytics', so the detector reads 0.36 here vs 0.40
in the table above):

| | Detector (batch 2) | Outline model | Leg (det / seg) | Arm (det / seg) |
|---|---|---|---|---|
| Val (76 frames, all pools) | 0.50 | **0.56** | 0.63 / 0.62 | 0.37 / **0.50** |
| **New pools (Doha + Paris)** | 0.36 | **0.41** | 0.49 / 0.51 | 0.24 / **0.30** |
| Budapest 2024 test | 0.49 | **0.55** | 0.70 / 0.74 | 0.29 / **0.36** |

**Reading:**
- The outline model finds limbs a little better than the detector on all three sets (+0.05 to
  +0.06), mostly arms. One run each, so this is near the noise level (~0.05), but it points the
  same way on every set.
- A likely reason: SAM's outlines are more consistent than hand-drawn box edges, so the model
  learns a cleaner target. Same labeling time, a bit better boxes, plus outlines.
- On the accepted test frames only, the gap is bigger: new pools 0.47 vs 0.41, Budapest 0.72
  vs 0.59 (outline model vs detector, boxes vs hand-drawn boxes). On the skipped (hard) frames
  they are equal (0.32 vs 0.33).
