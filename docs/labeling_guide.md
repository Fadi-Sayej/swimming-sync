# Labeling guide: leg and arm boxes (milestone 2)

## Labels

| Label | Key | What |
|-------|-----|------|
| `left_leg` | 1 | her **own** left leg, hip to toes |
| `right_leg` | 2 | her **own** right leg, hip to toes |
| `left_arm` | 3 | her **own** left arm, every visible part from shoulder to fingertips |
| `right_arm` | 4 | her **own** right arm, every visible part from shoulder to fingertips |

Left/right is always the swimmer's own side, not the side of the image. When she faces the
camera, her left leg is on the right of the image. When she is upside down, think about which
way her body faces before choosing.

## Rules

1. **Box only what you can see**, including a limb seen through the water. Don't guess where a
   hidden part continues.
2. **Tight boxes**: the box edges touch the outermost visible pixels (toes, fingertips).
3. **Legs together** (e.g. a vertical): still two boxes, even if they overlap almost completely.
4. **Can't tell which side?** Skip the frame. A wrong side teaches the model the wrong thing.
5. **Nothing visible** (fully underwater, or too small to see): submit the frame with no boxes.
6. **Only the swimmer.** No boxes on coaches, judges, officials or people on the deck.
7. **Pre-filled boxes come from the pose model.** Treat them as a rough start: fix the edges,
   delete wrong ones, and **check the side on every one** (the model often swaps left and right
   when she is upside down).

## Frames

`python -m swimsync.dataset.pick_frames` picks the frames (see the script for options):

- **train**: ~400 frames from the three Junior Worlds 2026 broadcasts (Budapest).
- **test**: 25 frames from each test video (Doha 2024, Paris 2025, Paris 2026, Budapest 2024).
  Label them the same way. They are never used for training; they give the mAP50 score.
  Budapest 2024 is reported separately (same city as the training videos, see `docs/videos.md`).

About 10% of the frames have no swimmer found by the model. Most are underwater moments;
label whatever you can see, or submit with no boxes.

Images and import files are in `data/raw/frames/m2/` (not in git).

## Label Studio

**Start it** (each labeling session), then open http://localhost:8080:

```bash
scripts/labelstudio.sh
```

Log in with the email and password in `data/raw/labelstudio/credentials.txt`. Your labels are
saved in `data/raw/labelstudio/` (not in git): don't delete that folder. Stop with Ctrl+C.

**Two projects:** `M2 legs+arms - train` (401 frames) and `M2 legs+arms - test` (97 frames).
Later batches: `M2 legs+arms - train2` (4 more pools) and `M2 legs+arms - train3` (132 close-ups
from 5 more pools, see `docs/videos.md`). Same rules.

**Per frame:**
1. Click **Label All Tasks** (or open a frame). Pre-filled boxes show up as a prediction.
2. Fix them: drag the corners, change the label (click the box, then 1-4), delete wrong ones
   (Backspace), draw missing ones (press 1-4, then drag).
3. **Submit** (Ctrl+Enter). Not a pool shot, or can't tell the side? **Skip**.

About 5% of the frames are TV graphics or athlete intros that slipped through: just skip them.

**When done:** Export → YOLO, save the zip into `data/labels/m2/`.

### First-time setup (already done on Fadi's Mac)

```bash
python3 -m venv ~/.venvs/label-studio && ~/.venvs/label-studio/bin/pip install label-studio
python -m swimsync.dataset.pick_frames --train data/raw/jr2026_*.mp4 --test data/raw/test/*.mp4
scripts/labelstudio.sh          # in one terminal; creates the login from credentials.txt
python -m swimsync.dataset.labelstudio   # in another: creates both projects, imports the frames
```

`credentials.txt` has 4 lines: `url: http://localhost:8080`, `email: ...`, `password: ...`,
`api_token: <40 hex characters>`. Don't run `label-studio --version`: in 1.23 it starts a
second server that blocks port 8080 (see docs/problems.md #15).

## Reviewing limb outlines (step 4, segmentation)

`python -m swimsync.dataset.limb_masks` lets SAM 2 outline the leg or arm inside every labeled
box (no drawing). Only the **test** outlines are reviewed, because they give the score; the
training outlines are used as they are.

Project **M2 limb outlines - test** (95 frames). For each frame:
- **All outlines right** (each covers its leg/arm, roughly to the edges): **Submit**.
  Small wobbles at the edges are fine.
- **Any outline wrong** (covers water, misses most of the limb, wrong limb, a limb without an
  outline): **Skip** the frame. Don't fix outlines by hand; it's not worth the time.

Then run `python -m swimsync.dataset.limb_masks` again: the submitted frames become the test set.

## Rechecking leg sides

`python experiments/limbs/recheck_sides.py create` makes the project **M2 leg sides - recheck**:
30 test frames with both legs already boxed as `leg?`. For each box, press **1** (her own left)
or **2** (her own right) after clicking it; don't move the boxes. Can't tell? Leave it as `leg?`.
Don't look at the old labels. Then `python experiments/limbs/recheck_sides.py score` shows how
often the side matches the first labeling.

## Keypoints (step 3, pose model training)

Projects **M2 keypoints - train** (200 frames from every training pool) and **M2 keypoints -
test** (88 test frames, never trained on). Made by `python -m swimsync.dataset.keypoints`.

12 points, her **own** side: blue = left, orange = right.

| Point | Left key | Right key |
|-------|----------|-----------|
| shoulder | q | w |
| elbow | e | r |
| wrist | t | y |
| hip | a | s |
| knee | d | f |
| ankle | g | h |

Each frame opens with:
- **Thin boxes:** your earlier leg/arm labels. They're only a guide, so don't edit them.
- **Pre-filled points:** from the pose model. Their sides already follow your boxes, and
  points outside your boxes were removed.

1. **Fix the pre-filled points:** drag a point that is off; delete one on the wrong limb.
2. **Add missing points you can see:** press the key, then click. A joint seen through the water
   counts.
3. **Don't guess hidden joints.** In a vertical the hips are usually under water: leave them out.
   A point you can't place within about a joint's width is better left out.
4. **Knee and elbow** go at the centre of the bend; **ankle** at the ankle bone (not the toes);
   **wrist** at the wrist (not the fingertips).
5. **Seen from behind or the side still counts.** If you see the back of the knee (or the
   side of it), put the knee point at the middle of the leg where thigh meets shin. Same for
   elbows. A joint is "hidden" only when that part of the limb isn't in view: under water you
   can't see through, behind the other leg or her body, or out of the frame.
6. **Can't tell her left from right?** Skip the frame.
7. Submit, even if only one or two points are visible.
8. **Watch for extra points.** A point label stays selected after you place a point, so every
   later click on the image adds another one. Before Submit, check the list on the right has
   each point at most once.

About 1-2 minutes per frame. Label train first; test can come last, but it's needed for the score.

## Front / back of the legs (leg sides, #25)

Projects **M2 front/back - train** (~1000 frames from the training videos) and **M2 front/back -
test** (~150 frames from the test videos, never trained on). Made by
`python -m swimsync.dataset.facing`. The two leg boxes found by the outline model are drawn
faintly as a guide.

One question per frame: **which way do her legs face?**

| Key | Choice | What you see |
|-----|--------|--------------|
| 1 | front | kneecaps, tops of the feet, toes pointing toward you, the front of her swimsuit |
| 2 | back | backs of the knees, heels, soles of the feet, buttocks |
| 3 | can't tell | side view (the kneecap shows as a bump on one edge of the leg, even if the legs look side by side), too blurry, legs hidden, or not her legs |

- Answer for **her body**, even when one leg is bent or turned a bit: in a split, which way does
  her front face?
- The boxes are only a guide: if they're wrong or on the wrong thing, still answer for her legs;
  if you can't see her legs at all, choose can't tell.
- Press the key, then Submit (Ctrl+Enter). About 3 seconds per frame.

The model learns front/back; her left and right then follow from the direction toward her
feet and where each leg is (`swimsync.dataset.facing`).

**When done:** `python -m swimsync.dataset.facing export` (Label Studio running) writes the crops to
`data/raw/datasets/m2_facing`.


## Side views (leg sides, #25)

Front/back has no answer when she is side-on (about half the frames). There, her sides follow
from two other facts: which way the front of her legs faces, and which leg is nearer you.
Project **M2 side view - test**: the 53 test frames with both legs hand-drawn, made by
`python -m swimsync.dataset.facing sideview`. The two leg boxes are drawn as **A** (blue, the one
on the image's left) and **B** (orange). The boxes don't tell you her sides.

**Question 1: which way does the front of her legs face?** (kneecaps, tops of the feet, the
front of her swimsuit)

| Key | Choice |
|-----|--------|
| 1 | front: toward you |
| 2 | back: away from you |
| 3 | image left: toward the image's left edge (side view) |
| 4 | image right: toward the image's right edge (side view) |
| 5 | can't tell |

**Question 2, side views only (3 or 4): which leg is nearer you?**

| Key | Choice | What you see |
|-----|--------|--------------|
| a | A | leg A covers part of B, or is sharper / bigger / less under water |
| b | B | the other way round |
| c | can't tell | |

- Skip question 2 when you chose front, back or can't tell.
- Answer for **her body**, as in front/back: in a split seen from the side, where does her front face?
- About 5 seconds per frame, then Submit (Ctrl+Enter).

**When done:** `python -m swimsync.dataset.facing sideview check` (Label Studio running) gives her
sides from your answers and compares them with the hand-drawn sides.

**Training frames:** project **M2 side view - train** (`python -m swimsync.dataset.facing sideview train`):
the 473 training frames answered "can't tell" in front/back, same two questions. Here A and B are
the outline model's leg boxes, so they can be wrong: if a box isn't on one of her legs, press 5
(can't tell).

## Leg height (milestone 3): "M3 leg height - test"

61 test frames, one per screen; the gray boxes are your earlier leg boxes, only a guide.
**Leg 1 = the leg on the image's left** (blue points), leg 2 = the other (orange points).

For each leg:

1. Click the points (pick the label with its key, then click):
   - `q` / `a`: **toes** (the tip of the foot, leg 1 / leg 2)
   - `w` / `s`: **knee**, only if the knee is out of the water
   - `e` / `d`: **water**, where the leg goes into the water (the middle of the leg at the surface)
2. Answer the question for each leg:
   - Leg 1: `1` knee out, `2` knee under water, `3` not out of the water
   - Leg 2: `4` knee out, `5` knee under water, `6` not out of the water, `7` pressed to leg 1
     (both legs together, can't be told apart: click leg 1's points only)
3. Submit.

- Splash hides where the leg enters the water: click your best guess of the middle of the leg at the surface.
- Leg bent: the knee is where it bends.
- Only one leg in the picture: it is leg 1; leg 2 = "not out of the water".
- **Whole leg out of the water** (water at or below her hip, or she is fully in the air): click the
  water point at her **hip** (the top of the thigh). That gives 100%.
- **Toes cut off by the edge of the picture:** click the knee and the water point only (no toes) and
  answer "knee out" (or "knee under water"). Pass/fail still counts; the percentage can't be measured.
