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

## Why the boxes miss on new pools (diagnosis, 2026-11-10)

Detector (batch 2), new-pool test frames, boxes with confidence >= 0.25:

- **Misses, not loose boxes.** Arms: 30% found (IoU >= 0.5), 8% loose, **62% missed**. Legs: 52%
  found, 16% loose, 32% missed (14% of legs are found but called "arm").
- **Doha: two kinds of shots the model never saw.** 6 frames are filmed *through the pool wall*
  (underwater window camera) and 2 are very wide shots with tiny limbs. Together they hold 27 of
  Doha's 57 limbs, and only 2 are found. On Doha's normal shots, 20 of 30 limbs (67%) are found,
  about as many as Budapest 2024 (70%).
- **Paris 2025 is the hardest normal video:** 37% of limbs found (Paris 2026 56%). Its misses are
  mostly arms at the surface next to the face or body, often in splash, and some clearly visible
  raised arms.
- **Not a lack of arm labels:** training has 279 arms and 274 legs.
- **Limb size is not the main cause:** test close-ups have much bigger limbs than training
  (median long side 660-840 px vs ~340 px), but Paris 2026 has the biggest limbs and the best
  score.
- **Both models were still improving at round 100** (detector val 0.485 -> 0.500 over the last
  5 rounds), so training longer is a free try.

## Outline model, 200 rounds + split-limb fix (#21), 2026-11-11

`EPOCHS=200 NAME_SUFFIX=_200ep scripts/train_limbs.sh seg`. Stopped early at round 129 (no val
gain for 30 rounds); best round 104. Same frames as the 100-round model, outlines redone with the
#21 fix (29 limbs changed).

| | 100 rounds | 200 rounds + #21 |
|---|---|---|
| Outline mAP50, val | 0.64 | 0.62 |
| Outline mAP50, new pools (41 frames) | 0.62 (leg 0.69, arm 0.55) | 0.62 (leg 0.71, arm 0.53) |
| Outline mAP50, Budapest (15 frames) | 0.87 (leg 0.99, arm 0.75) | 0.61 (leg 0.97, arm 0.25) |
| Boxes vs hand boxes, val | 0.56 | 0.50 |
| Boxes vs hand boxes, new pools | 0.41 | 0.34 |
| Boxes vs hand boxes, Budapest | 0.55 | 0.49 |

- **No gain, a bit worse.** Training longer doesn't help: the val best came at round 104 and the
  rest was flat. The 100-round model's best was at round 78, so "still improving at round 100"
  was true for the detector, not the outline model.
- The Budapest drop is mostly arms on 15 frames (28 limbs), so a few limbs swing it a lot.
- Two things changed at once (rounds and #21). A 100-round run on the #21 outlines
  (`seg_fix21`) separates them.
- The 100-round model (`output/seg`) stays the one we use.

## Batch 3: close-ups from 5 more pools, 2026-11-11

Added 114 labeled frames (of 132) from 9 close-up solo videos: Munich, Sweden, Delfin Cup,
Singapore 2025, Junior Worlds 2024 (`docs/videos.md`). 108 legs, 85 arms. Train 387 + val 99
(the old frames keep their split). Same settings; batch-2 models kept, new ones are `*_b3`.
Same Ultralytics scorer as the batch 2 table (the batch-2 numbers below re-run, they match).

mAP50, 2 classes:

| | Batch 2 (5 pools) | **Batch 3 (10 pools)** |
|---|---|---|
| Training pool (Budapest val, 47 frames) | 0.60 | 0.61 |
| Budapest 2024 test | 0.46 | **0.53** |
| **New pools (Doha + Paris)** | **0.40** | **0.41** |
| New pools: leg | 0.56 | 0.59 |
| New pools: arm | 0.24 | 0.23 |
| Doha 2024 | 0.40 (leg 0.66, arm 0.14) | 0.32 (leg 0.43, arm 0.21) |
| **Paris 2025** | 0.28 (leg 0.35, arm 0.22) | **0.47** (leg 0.61, arm 0.32) |
| Paris 2026 | 0.60 (leg 0.76, arm 0.43) | 0.55 (leg 0.80, arm 0.30) |

4 classes: new pools 0.24 -> 0.26, Budapest 2024 0.23 -> 0.43. Side right when the 4-class
model finds the limb (confidence >= 0.25, IoU >= 0.5), new pools: legs 22/42 -> 25/50 (still a
coin flip), arms 12/16 -> 8/13.

**Reading:**
- **The close-ups fixed what they were meant to fix:** Paris 2025, the close-up pool, 0.28 -> 0.47.
- **Overall new pools flat** (0.40 -> 0.41): Doha and Paris 2026's arms dropped by about as much.
  Doha's legs drop may partly undo the swimmer-overlap boost noted in batch 2 (Platanioti in
  the Fukuoka training video), now a smaller share of training.
- Arms are still the weak part (0.23), and the detector's left/right for legs is still a coin
  flip, so the sides should come from the pose model (step 3).
- ~25 frames per pool: changes under ~0.05 overall, or ~0.1 for one pool, are noise.

### Outline model with batch 3 (`seg_b3`, 100 rounds, #21 outlines)

1060 limbs outlined by SAM, none failed the check. Best val round 77. Test outlines are the same
56 reviewed frames as before (41 new pools, 15 Budapest).

| | Outline model, batch 2 | **Outline model, batch 3** |
|---|---|---|
| Outline mAP50, new pools | 0.62 (leg 0.69, arm 0.55) | **0.70** (leg 0.75, arm 0.65) |
| Outline mAP50, Budapest 2024 | 0.87 (leg 0.99, arm 0.75) | 0.73 (leg 0.98, arm 0.49) |

Boxes vs the hand-drawn boxes (`box_vs_hand.py`, all 95 test frames), all four models:

| | Detector b2 | Detector b3 | Outline b2 | **Outline b3** |
|---|---|---|---|---|
| **New pools** | 0.36 | 0.39 | 0.41 | **0.45** |
| New pools: leg | 0.49 | 0.59 | 0.51 | 0.60 |
| New pools: arm | 0.24 | 0.18 | 0.30 | 0.30 |
| Doha 2024 | 0.37 | 0.32 | 0.32 | 0.33 |
| Paris 2025 | 0.29 | 0.41 | 0.44 | 0.47 |
| Paris 2026 | 0.53 | 0.52 | 0.55 | **0.65** |
| Budapest 2024 | 0.49 | 0.53 | 0.55 | 0.57 |

**Reading:**
- **The batch-3 outline model is the best leg/arm box finder so far:** new pools 0.36 (first
  detector) -> 0.45, with outlines as a bonus (0.70 on new pools). It is the model to use.
- Its arms hold at 0.30 where the batch-3 detector's dropped to 0.18: the outline target helps arms.
- Budapest arm outlines dropped (0.75 -> 0.49), as in the 200-round run, also on #21 outlines.
  Both runs used the #21 fix, so it may be the cause; but it is ~12 arms on 15 frames, and the
  Budapest *boxes* went up (0.55 -> 0.57). Watch it, not acted on.
- Doha stays ~0.33 for every model: its through-the-wall shots need their own training frames
  (or are out of scope; to ask Assaf).

## Step 3: left/right from the pose model inside the boxes, 2026-11-11

`python experiments/limbs/sides_eval.py` (code: `src/swimsync/limbs/sides.py`). Each leg/arm box
takes the pose skeleton limb (yolo11m-pose, COCO) that overlaps it most; two leg boxes always get
one left and one right. Truth: the hand-drawn 4-class test labels. "Decided" = boxes a skeleton
limb overlapped. Side right, % of decided:

| New pools (72 frames) | Arms | Legs |
|---|---|---|
| Your boxes, pose on the full frame | **79%** (31/39) | 41% (16/39) |
| Your boxes, pose zoomed on the swimmer | 66% (21/32) | 75% (30/40) |
| Outline model's boxes, full frame | **80%** (16/20) | 36% (10/28) |
| Outline model's boxes, zoomed | 88% (14/16) | 42% (11/26) |
| 4-class detector (batch 3), for comparison | 62% (8/13) | 50% (25/50) |

Budapest 2024 (23 frames): arms 89% (8/9) full frame; legs 35% full frame, 57% zoomed.

Also tried running the pose upside down (image rotated 180 degrees, which keeps her own left/right;
all 95 frames, your boxes): legs 49% (full frame) and 35% (zoomed), or 42% / 50% taking whichever
way up matches the boxes better. No help.

**Reading:**
- **Arms: the pose model gives the side, ~80% right** (full frame), better than the 4-class
  detector (62%) and on more arms. Use it for arm sides.
- **Legs: no better than a coin flip, often worse.** Looking at the wrong ones: most leg frames
  show only the legs (verticals, splits, legs out of the water). With no torso or face, the pose
  model invents a body and often puts both its left and right leg on the same real leg. The
  one 75% result (your boxes, zoomed) doesn't hold with the model's boxes (42%).
- Leg sides need something that sees more than one frame (follow the legs from a moment where
  the body is visible) or a model trained for it (keypoint labels on leg crops). To discuss with
  Assaf on Nov 24.
- The joints themselves can't be scored yet: there are no hand-labeled keypoints.

### Leg sides by following the legs through the video, 2026-11-11

`python experiments/limbs/track_eval.py` (code: `src/swimsync/limbs/track.py`). For each test
frame, the 3 s of video before and after it: leg boxes from the batch-3 outline model, pose
skeletons, camera cuts. The leg sides come from the nearest frame in the same shot where the pose
model sees the torso (both shoulders and hips, confidence >= 0.5) and both legs are boxed, then
the two leg boxes are followed frame by frame to the test frame. All 95 video frames matched
their test images.

| Legs, side right (% of decided) | New pools, your boxes | New pools, model boxes | Budapest, your boxes |
|---|---|---|---|
| Pose on the frame alone | 41% (16/39) | 36% (10/28) | 35% (8/23) |
| Following the legs | 50% (34/68) | 48% (22/46) | 48% (14/29) |
| Following, else pose | 51% (38/75) | 47% (24/51) | 42% (14/33) |

**Reading:**
- **No real gain: still a coin flip.** It decides more legs, but half of them wrong.
- The "torso visible" frame was a median 0.4 s away: the pose model reports a confident torso even
  on legs-only frames (it invents the body), so those frames aren't trustworthy either.
- Only 3 test legs have the face and torso clearly in view, too few to check whether the pose
  model and the labels agree when the whole body is visible.
- Open question for Assaf: how sure are the leg labels themselves (hard to see while labeling),
  and do we need her own left/right for legs, or is "leg 1 / leg 2" (or image left/right) enough
  for the sync score?

### How sure are the leg labels? (2026-11-11)

`experiments/limbs/recheck_sides.py`: 30 test frames (all 4 test videos), both legs pre-boxed as
"leg?", sides relabeled without seeing the old ones. **58 of 60 leg sides the same as the first
time (97%), none left unsure.** The labels are consistent, so leg sides are learnable: the ~50%
of every model so far is the models' limit, not the labels'.

## Medium outline model (YOLO11m-seg, batch 3), 2026-11-12

Same data and settings as `seg_b3`, but the medium model (3x the size) and 4 images per step
instead of 8 (memory, #26). 100 rounds, best val round 96 (val outlines 0.55 vs 0.63 for the small model).

| | Small (`seg_b3`) | Medium (`seg_m_b3`) |
|---|---|---|
| Boxes vs hand boxes, new pools | **0.45** (leg 0.60, arm 0.30) | 0.36 (leg 0.53, arm 0.18) |
| Doha 2024 | **0.33** | 0.28 |
| Paris 2025 | **0.47** | 0.34 |
| Paris 2026 | **0.65** | 0.57 |
| Budapest 2024 | **0.57** | 0.45 |
| Outline mAP50, new pools | **0.70** (leg 0.75, arm 0.65) | 0.58 (leg 0.69, arm 0.46) |
| Outline mAP50, Budapest 2024 | **0.73** | 0.56 |

**Reading:** the medium model is worse everywhere, mostly on arms. With ~390 training frames a
bigger model has more to learn from too little data; it was also still improving at round 96, so
it would need more rounds (~2 min each on the Mac). Not worth it: **the small model stays.**
A bigger model is worth retrying only with more labeled frames, on the college GPU.

## Mirrored 4-class detector (leg sides), 2026-11-12

`python -m swimsync.dataset.mirror` then `scripts/train_limbs.sh 4cls_mirror`: every training frame
also mirrored, with left/right labels swapped (774 train images). The model can no longer learn
"her left leg is on the image's left", only her body. YOLO11s, 100 rounds, same test frames.

| 4-class detector | Batch 3 | **Mirrored** |
|---|---|---|
| mAP50, new pools | 0.26 | **0.34** |
| Doha 2024 / Paris 2025 / Paris 2026 | 0.25 / 0.25 / 0.40 | 0.30 / 0.34 / 0.46 |
| Budapest 2024 | **0.43** | 0.39 |
| Leg side right, new pools (found legs) | 50% (25/50) | 44% (24/54) |
| Leg side right, Budapest | 68% (19/28) | 40% (12/30) |
| Arm side right, new pools | 62% (8/13) | 47% (9/19) |

**Reading:**
- Mirroring helps *finding* the limbs (twice the training frames: new pools 0.26 -> 0.34), not their sides.
- The old model's 68% on Budapest was the camera side, not her body: with that shortcut taken away
  it drops to 40%. A box detector sees one leg at a time and can't tell her left from her right.
- Leg sides stay open (#25). The labels are reliable (97%), so the next try is a pose model
  fine-tuned on swimmer keypoints (hips/knees/ankles with sides), to discuss with Assaf.

### Both legs in one crop (quick test), 2026-11-13

One crop around both hand-drawn leg boxes, a classifier (YOLO11s-cls, 256 px) asked "is the
leg on the image's left her own left?". 171 training frames with both legs, plus mirrored copies
with the answer flipped (so the camera side can't be learned). Not kept in the repo (quick test).

| | Right |
|---|---|
| Val (during training) | 50-63%, no steady rise |
| Test, new pools | 62% (23/37) |
| Test, Budapest 2024 | 75% (12/16) |

**Reading:** maybe a little better than a coin flip, but with 37 test frames 62% is within
chance. Seeing both legs isn't enough with this little data; the keypoint-trained pose model is
the next try.

