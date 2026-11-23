# Swimming Sync: what we did and why (up to Nov 23, 2026)

This document explains the whole project so far in order: what we built, which models we
trained, which ready-made models we used without training, where we used hand-written rules
instead of a model, and what worked and what didn't. Details and exact numbers are in
[experiments/limbs/results.md](../experiments/limbs/results.md), problems in
[problems.md](problems.md), and the day-by-day log in [worklog.md](worklog.md).

---

## 1. The goal

Score how synchronized artistic swimmers are, and give coaches a tool that shows skeletons and
out-of-sync moments. Ready-made pose models do badly in the pool (splash, water, upside-down
bodies, legs only), so the project builds pieces that work in the pool, step by step.

## 2. Three kinds of building blocks

Everything we use is one of three things. Keeping them apart makes the flow easy to follow:

| Kind | What it is | Example |
|---|---|---|
| **Trained model** | A model we trained on our own labeled frames | the leg/arm outline model |
| **Ready-made model** | Someone else's model, used as is (no training by us) | yolo11m-pose, SAM 2, CLIP |
| **Rule** | A few lines of logic we wrote by hand; no learning | "feet point up", the left/right geometry |

All trained models are YOLO11 (Ultralytics), trained on the Mac (M4 Pro). Labels were made in
Label Studio. Test numbers are always on 4 videos never used in training: Budapest 2024, Doha
2024, Paris 2025, Paris 2026 ("new pools" = Doha + the two Paris videos, pools never seen in
training).

---

## 3. Milestone 1 (done Nov 9): how much of her body is above the water

**Question:** in a solo video, what share of the swimmer's body is out of the water?

**Flow:**
```
video frame -> ready-made pose model (yolo11m-pose): her keypoints
            -> RULE: find the waterline (water color, per column, then a straight-line fit)
            -> RULE: pick the swimmer = biggest person whose box reaches below the waterline
            -> RULE: each keypoint above or below the line -> % of body above water
```

- **No model trained.** A color rule found the waterline well (error 4.1 px on 50 hand-labeled
  frames), so training a model for it wasn't needed.
- Rules had to be fixed several times: the line landed on blue walls and banners, judges were
  picked instead of the swimmer, a line was drawn on the end logo screen (problems #3-#6, #9-#10).
- **Result:** ~23% of her body above water on average. The ready-made pose model misses the
  swimmer in ~26% of frames; a bigger pose model (yolo11x) didn't help. This led to milestone 2.

---

## 4. Milestone 2 (agreed Nov 9): limbs, sides, pose, segmentation

Four parts agreed with Assaf:

1. **Leg boxes** with her left/right.
2. **Arm boxes** with her left/right.
3. **Pose model inside the boxes** (trained on swimmer keypoints).
4. **Segmentation** of water / splash / swimmer.

Where each part stands today:

| Part | Status | Best result (test) |
|---|---|---|
| 1. Leg boxes | Works | 0.60 mAP50 on new pools |
| 1. Leg left/right | Partly solved | 91% right on 21% of frames, or 77% on ~80% |
| 2. Arm boxes | Weak | 0.30 on new pools (0.51 on val) |
| 2. Arm left/right | Works | ~80% (ready-made pose + matching rule) |
| 3. Pose inside boxes | Trained, not better | ready-made model kept |
| 4. Segmentation | Paused | color rules failed; needs SAM 3 |

### 4.1 Data: picking and labeling frames

- **Videos:** 3 Junior Worlds 2026 broadcasts first, then 4 more pools (batch 2), then 9 close-up
  videos from 5 more pools (batch 3). Test: the 4 held-out videos above.
- **Picking frames (rules + ready-made models, no training):** the frame picker runs the
  ready-made pose model to find people, a **rule** keeps people surrounded by pool-colored water
  (the swimmer, not judges), and the ready-made **CLIP** model drops frames that aren't pool shots
  (crowd, logos). A rule spreads the picks so they aren't seconds apart.
- **Labeling:** boxes pre-filled by the pose model, corrected by hand in Label Studio. One box per
  leg/arm, with her left/right.

### 4.2 Parts 1+2: finding the leg and arm boxes

Trained models, in order:

| # | Model | Result | Kept? |
|---|---|---|---|
| 1 | Detector, 4 classes (left/right leg, left/right arm), batch 1 | new pools 0.09 | No |
| 2 | Detector, 2 classes (leg, arm), batch 1 | new pools 0.23 (val 0.55): doesn't carry over to new pools | No |
| 3 | Same + stronger color changes in training | 0.21: no gain, color isn't the gap | No |
| 4 | Detector, 2 classes, + batch 2 (4 more pools) | new pools 0.40: **more pools was the fix** | No (replaced) |
| 5 | Detector + batch 3 (5 more pools, close-ups) | 0.41 | No (replaced) |
| 6 | **Outline model** (YOLO11s-seg), batch 3 | boxes 0.45 (legs **0.60**, arms 0.30), outlines 0.70 | **Yes** |
| 7 | Outline model trained 200 rounds | no gain | No |
| 8 | Outline model, medium size (YOLO11m-seg) | worse (0.36) | No |

- **The outline model's training outlines were not drawn by hand.** The ready-made **SAM 2**
  model turned every hand-drawn box into an outline automatically (867 limbs, ~2 min). A rule
  checks each outline fits its box. Training on outlines gave better boxes than training on
  boxes, mostly for arms.
- **Arms are the weak part.** Tried without success: boxes from the pose model's arm points
  (0.15-0.19), higher resolution, zooming on her, filling missed arms from nearby video frames.
  Most Doha misses are through-the-wall underwater shots, which training doesn't have.

### 4.3 Parts 1+2: her left and right

#### Arms: ready-made pose model + a matching rule (works, ~80%)

```
arm boxes (outline model) + skeleton (ready-made yolo11m-pose)
  -> RULE: each arm box takes the skeleton arm that overlaps it most -> that arm's side
```

The 4-class detector got arm sides 62% right; this gets ~80%. Our own trained pose model got
52-58%, so the ready-made one is used.

#### Legs: the long story

Leg sides were the hardest problem of the milestone (problem #25). Attempts in order:

| # | Attempt | Kind | Result | Why it failed / worked |
|---|---|---|---|---|
| 1 | 4-class detector says left/right leg | trained | 25/50, coin flip | Sees one leg at a time; can't know which one |
| 2 | Ready-made pose model + matching rule (like arms) | ready-made + rule | 41% | On legs-only frames it invents a body and puts both legs on one real leg |
| 3 | Pose model upside down / zoomed | ready-made | 35-50% | Same problem |
| 4 | Follow legs through the video from a frame where her body is visible | rule | ~50% | The pose model "sees" a torso even when there is none |
| 5 | Check our own labels: relabel 30 frames blind | - | 97% same | **Labels are reliable; the models are the limit** |
| 6 | Mirrored 4-class detector (mirror copies, sides swapped) | trained | 44% | Finds limbs better (0.34) but still can't tell sides |
| 7 | Classifier on a crop of both legs, "is the left one her left?" (256 px, 512 px, + nearby frames) | trained | 54-62% | Training error stuck at a coin flip |
| 8 | Pose model trained on our keypoints | trained | 41-52% | Too few frames (192); points got slightly worse |

**The key finding (Nov 14):** "is this her left or right?" is the one question whose answer
**flips in a mirror image**. Image models are pretrained to see an image and its mirror as almost
the same, so they can't learn it: training error stayed at exactly a coin flip (0.693), even on
the training images. So we stopped asking models for left/right directly.

**The new approach:** ask the model something that does *not* flip in a mirror, then get
left/right with a geometry **rule**:

| # | Attempt | Kind | Result |
|---|---|---|---|
| 9 | **Front/back model**: do her legs face the camera or away? (884 frames, one click each) | trained + rule | 91% right when sure, but sure on only 9% of frames |
| 10 | + **carry rule**: copy sides from a sure frame up to 0.1 s away by following the boxes | rule | **10/11 (91%) on 21% of frames** |
| 11 | SAM 2 follows each leg from the sure frame (instead of the carry rule) | ready-made | 57%: masks merge when legs touch |
| 12 | Side views by hand: which way her front faces + which leg is nearer (test check) | hand + rule | **38/40 (95%)**: the geometry works |
| 13 | **View model** (which way her front faces), 473 frames you labeled | trained | **works: 86%** with hand nearer-leg answers |
| 14 | Nearer-leg model on both legs | trained | learned nothing (same mirror problem) |
| 15 | Nearer-leg model on one leg at a time ("nearer or farther?") | trained | 74% on held-back legs, but on test no better than always "A" |
| 16 | View model + nearer-leg model, fully automatic | trained + rule | 67% on 81% of frames (77% with "A is nearer") |

So today: **front/back + carry is accurate but rare; side views cover most frames but the
nearer-leg model is weak.**

### 4.4 Part 3: pose model trained on swimmers

- Labeled 12 keypoints (hips, knees, ankles, shoulders, elbows, wrists) on 192 training + 87 test
  frames, sides pre-filled from the boxes.
- Fine-tuned YOLO11m-pose on crops around her limbs, twice (normal and 10x lower learning rate).
- **Result:** it finds her better (box 0.63 -> 0.87) but the points got slightly worse (72% ->
  71% within 20% of her size) and sides no better. **The ready-made pose model stays.** 192
  frames are too few to beat a model trained on huge datasets.

### 4.5 Part 4: water / splash / swimmer segmentation

- Tried SAM 2 for the swimmer plus **color rules** for water (blue-green) and splash (white near
  her). The rules fail: splash and glare look alike, banners look like water (problem #30).
- **Paused.** SAM 3 can segment by name ("splash", "water") but its weights need access from
  Meta. To ask Assaf.

---

## 5. All models we trained

| Model | Does | Trained on | Result | Kept? |
|---|---|---|---|---|
| Detector 4 classes (+ mirrored version) | leg/arm boxes with side | batches 1-3 | sides a coin flip | No |
| Detector 2 classes (batches 1, 2, 3, + color version) | leg/arm boxes | batches 1-3 | new pools 0.23 -> 0.41 | No (outline model better) |
| **Outline model `seg_b3`** (YOLO11s-seg) | leg/arm outlines + boxes | batch 1-3, SAM outlines | legs 0.60, arms 0.30 | **Yes** |
| Outline model 200 rounds / medium size | same | same | no better / worse | No |
| Both-legs side classifiers (quick tests) | "is the left box her left?" | 171 frames + mirrors | coin flip | No |
| Pose model (YOLO11m-pose, 2 runs) | 17 keypoints | 192 frames | not better than ready-made | No |
| Front/back classifier v1 | legs face camera / away | 88 frames | 43% | No |
| **Front/back classifier `facing`** (YOLO11s-cls) | legs face camera / away / can't tell | 884 frames | 91% when sure | **Yes** |
| **View classifier `view`** (YOLO11s-cls) | which way her front faces (7 answers) | 473 + front/back frames | 86% sides with hand nearer leg | **Yes** |
| Nearer-leg classifier `nearer` | A or B nearer (both legs) | ~300 frames + mirrors | learned nothing | No |
| Nearer-leg classifier `nearleg_noaug` | is this leg nearer or farther? | 254 frames x 2 legs | no better than "A" on test | For now |

## 6. Ready-made models we used without training, and why

| Model | Used for | Why not train our own |
|---|---|---|
| **yolo11m-pose** | milestone 1 keypoints; arm sides; pre-filling labels | our fine-tuned version was worse (only 192 frames) |
| yolo11x-pose | tried as a bigger pose model | no better, 3x slower |
| **SAM 2.1** | outlines from boxes (free training labels); tried for following legs in video | it outlines well without training; following legs failed (masks merge) |
| **CLIP** | frame picker: "is this a pool shot?" | a yes/no on general images; works out of the box |
| SAM 3 (wanted) | water / splash segmentation by name | weights need Meta access |

## 7. Rules we wrote, and why a rule instead of a model

| Rule | What it says | Why a rule | Code |
|---|---|---|---|
| Waterline | per image column, water starts where the tallest blue/cyan stretch starts; fit a straight line, ignore outliers | color is enough (4.1 px error), no labels needed | `waterline/detect.py` |
| Swimmer pick | the biggest person whose box reaches below the waterline | judges and spectators are never in the water | `waterline/above_water.py` |
| Frame picker | keep people surrounded by pool water; spread picks apart | cheap and reliable for choosing frames | `dataset/pick_frames.py` |
| Outline check | a SAM outline must fit its box; keep all pieces of a limb cut in two (#21) | catches SAM's mistakes automatically | `dataset/limb_masks.py` |
| Arm side matching | each arm box takes the skeleton arm that overlaps it most | links the pose model's output to our boxes | `limbs/sides.py` |
| Feet up | legs out of the water point up | front/back + feet up gives the hand-drawn sides on 88% of frames; the pose model's feet direction did worse (29 of 36) | `dataset/facing.py` |
| Front/back -> sides | upside down and facing the camera: her left leg is on the image's left; facing away: the other way | geometry; this is what a model can't learn (mirror problem) | `facing.sides_from_facing` |
| Sure only | use the front/back model only when it is >= 80% sure | 91% right when sure vs 78% forced | `limbs/track.py` |
| Carry | copy sides from a sure frame up to 0.1 s away, following the boxes | further carries go wrong (she spins, boxes swap) | `limbs/track.py` |
| Side-view geometry | her left = head direction x front direction; if that points at the camera, the nearer leg is her left | geometry; checked 95% with hand answers | `facing.sides_from_view` |
| Crossed legs (Nov 23) | when sides come from where the boxes are (front/back), and the two leg boxes almost cover each other, answer "can't tell" | "which box is further left" is a few pixels there; it was wrong on 2 of 3 such frames | `facing.apart` |
| Mirrored training copies | each training frame written 4 times with the answers mirrored; built-in random flips off | so random flips can't teach wrong sides (Assaf's question) | `facing.export_view` |
| Color rules for water/splash | blue-green = water, white near her = splash | tried as a quick start; **failed** | scratch only |

## 8. When we chose not to train a model

- **Waterline:** the color rule was accurate enough, so training would have cost labels for nothing.
- **Outlines for training:** SAM 2 drew them from the boxes, so no outline labeling.
- **Leg left/right directly:** a model can't learn a question that flips in a mirror; we train
  models on mirror-proof questions (front/back, view) and use geometry rules for the sides.
- **Feet direction:** assumed up (a rule) because the pose model's direction was worse.
- **Pose points and arm sides:** the ready-made model beat our trained one.

## 9. What runs on a new video today

```
frame
 |- outline model (trained) ............ leg and arm boxes + outlines
 |- arms: ready-made pose model + matching rule ............ her left/right arm (~80%)
 |- legs, if exactly 2 legs found:
 |    |- front/back model (trained) sure >= 0.8? -> front/back rule -> sides (~91%)
 |    |- else a sure frame within 0.1 s? -> carry rule -> sides
 |    |- else view model (trained):
 |         |- legs face camera/away -> front/back rule -> sides
 |         |- side view -> nearer-leg model (trained, weak) -> side-view rule -> sides
 |    (front/back rule on crossed legs -> "can't tell")
 |- legs, if 1 or 3+ found ............ no sides
```

The demo clips for the Nov 24 meeting (`experiments/limbs/meeting_nov24.py`) run this without the
carry step.

## 10. Open questions for Assaf (Nov 24)

1. Is 91% right on a fifth of the frames, or 77% on 80%, more useful for the sync score? Can the
   score use leg measures that don't need left/right?
2. "A (the left box) is nearer" in 75% of our labels: a real camera effect, or a labeling habit?
3. Through-the-wall underwater shots: in scope?
4. College GPU access.
5. SAM 3 access for segmentation, or another way?
6. Next labeling: more nearer-leg answers, or the arm-box batch?
