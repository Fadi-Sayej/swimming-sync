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
