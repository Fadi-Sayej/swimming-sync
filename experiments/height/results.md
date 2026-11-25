# Leg height (milestone 3): results

Goal and score: `docs/superpowers/specs/2026-11-24-leg-height-design.md`. Plan:
`docs/superpowers/plans/2026-11-24-leg-height.md`.

## Ready-made knees, 2026-11-24

`python experiments/height/knee_check.py`. Existing labels only: the knee clicks of "M2 keypoints -
test" and the hand-drawn leg boxes of the 4-class test set (114 leg boxes, 105 knees clicked; a
leg box "has its knee out" when a hand knee click is inside it). yolo11m-pose, knees with
confidence >= 0.3, sides ignored.

| | Full frame | Zoomed on the legs |
|---|---|---|
| Knee out / not, agrees with the hand labels | 66 / 114 (58%) | 56 / 114 (49%) |
| Model knee within 10% of the leg's size from the hand knee | 22 / 105 (21%) | 16 / 105 (15%) |
| Within 20% | 37 / 105 (35%) | 39 / 105 (37%) |

**Reading:** the ready-made knees are far too weak for the leg height score (the bar was 80%
within 20% and 85% agreement): on legs-only frames the pose model invents a body and puts its
knees in the wrong place, as it did for leg sides. **We train our own knee model (plan Task 8)**,
starting from the 192 frames with hand knee clicks in "M2 keypoints - train".

## Our knee model, 2026-11-24

Training crops: one per hand-drawn leg box of the 192 "M2 keypoints - train" frames (239 train +
50 val crops, padded 30%), the knee clicked inside the box or none (`python -m
swimsync.dataset.export_knees`). Same check as above (114 test leg boxes, 105 knees).

| Knee finder | Knee within 10% | Within 20% | Knee out / not agrees |
|---|---|---|---|
| Ready-made, full frame | 21% | 35% | 58% |
| Ours, 1 keypoint (YOLO11s-pose, `knees`) | 7% | 24% | 72% |
| **Ours, 17-point layout** (YOLO11m-pose, low rate, `knees17`) | **48%** | **76%** | **76%** |

- One keypoint: the point part of the model starts from scratch, and with 239 crops it learns
  "the middle of the box" (the knee sits mid-shin on almost every crop).
- Keeping the ready-made model's 17-point layout (knee, ankle, hip put in both side slots, sides
  don't matter) keeps what it knows about knees, and a 10x lower learning rate
  (`optimizer=AdamW lr0=0.0002`, 60 rounds) keeps it from being wiped: knees within 20% go from 35%
  to 76%. This is the knee finder now (`OurKnees` default).
- It says "knee out" on 19 legs where none was clicked (94 vs 83): too eager; the confidence cut
  (0.5) gets checked on the new leg-height labels.

```bash
.venv/bin/yolo pose train data=data/raw/datasets/m3_knees17/data.yaml model=yolo11m-pose.pt imgsz=384 \
  epochs=60 patience=20 batch=16 fliplr=0.5 flipud=0.5 optimizer=AdamW lr0=0.0002 seed=0 device=mps \
  project=$PWD/experiments/limbs/output name=knees17
```

## Leg height on the test frames, 2026-11-25

`python experiments/height/height_eval.py --knees ours` against "M3 leg height - test" (85 of 91
frames labeled: 61 test frames from the 4 test videos + 30 phone-video frames; 4 skipped as not a
pool shot by the CLIP check, `MIN_POOL = 0.7`). Raw output: `experiments/height/output/eval_ours.txt`.

**Each leg you clicked** (toes/knee/water), matched to the nearest leg the outline model found:

| | New pools | Budapest | Phone videos |
|---|---|---|---|
| Legs out of the water (with points) | 45 | 18 | 15 |
| Found by the outline model | 35 (78%) | 17 | 14 |
| Knee out / under agrees with you | **29 / 35 (83%)** | 12 / 17 (71%) | **13 / 14 (93%)** |
| (same with the ready-made knees) | 16 / 35 (46%) | 6 / 17 (35%) | 7 / 14 (50%) |
| Score difference when both say knee out | 7.5 points (22 legs) | 3.4 (9) | 4.7 (9) |
| Toes within 20% of the leg | 96% | 100% | 100% |
| Water point within 20% | 91% | 94% | 93% |
| Knee within 20% | 76% | 100% | 93% |

All the mistakes on these legs are false FAILs (knee out by hand, under by the model); all the
hand FAILs were answered without points (legs under water, or knee under with only shins up), so
false passes only show per frame:

**Per frame, number of legs passing agrees with you: 36 of 81 frames (44%)** (a leg not seen out
of the water counts as FAIL). Why the others disagree (`scratchpad` diagnosis):
- 21 frames: the model passes a leg you failed. Mostly legs standing at about knee height: the
  shins are up, the knee just under the surface, and the knee model puts a knee just above the
  waterline. Twice the "leg" is an arm next to her head.
- 15 frames: the model finds one leg where you clicked two (legs pressed together become one
  outline; the second leg then counts as FAIL). This is the open "legs pressed together" question.

**Tried, no help:** a higher knee-confidence cut (0.5 to 0.95) or requiring the knee to sit further
above the water (5% to 45% of the leg): every setting trades false passes for false fails; chosen on
half the frames, the other half scores 62% / 39% (cut) and 62% / 39% (position). The model's knee
sits at the same place on passed and failed legs (median 84% vs 89% of the way from toes to water):
the boundary (water at knee height) is exactly where most of these legs are.

**Reading:** where a knee is clearly out, the model finds it and the percentage is within ~3-8
points of yours; the phone videos are as good as the broadcasts. What fails is the boundary case
(water at the knee), which needs training examples of it: legs with the knee just under the
surface. Next: label such frames from the training videos (Fri/Sat) and retrain the knee model.

### Knee on the wrong leg, 2026-11-25

In the phone videos both legs' lines sometimes met at one knee: the crop around one leg also shows
the other, and our knee model took that leg's knee. A knee now counts only on its own leg (inside
its outline grown by 5% of the leg, at least 15 px; `measure.inside_leg`). Same test frames:
knee out / under 83% / 71% / 86% (new pools / Budapest / phone; one phone leg fewer), per frame 36 of
81 as before. A 15 px margin alone cost Budapest 71% -> 65% (the knee sits at the outline's edge).

### Next: labels at the boundary

"M3 leg height - train" (`python -m swimsync.dataset.height_labels train`): 215 frames from the 18
training videos (about 8 per video near the boundary: the model FAILs a leg or puts the knee in
the last 30% before the water), pre-filled with the model's points and answers.
