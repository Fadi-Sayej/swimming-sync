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

