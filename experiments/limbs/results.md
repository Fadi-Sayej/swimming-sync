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
