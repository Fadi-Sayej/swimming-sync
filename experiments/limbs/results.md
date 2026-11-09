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

## Test (new pools: Doha + Paris; Budapest separately)

Pending: test frames not labeled yet.
