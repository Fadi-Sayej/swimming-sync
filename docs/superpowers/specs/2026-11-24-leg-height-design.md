# Leg height score: design (2026-11-24)

## Why

At the Nov 24 meeting Assaf set a new goal: measure how much of the legs is out of the water, with
the knee as the key point. Leg left/right is no longer the priority. This is the height judges look
at in artistic swimming (how high the legs stand out of the water), and it feeds the sync score later.

**Due:** Tuesday Dec 1, 2026, 14:00 (next meeting).

## What Fadi shows on Tuesday

A video of the working pipeline on videos the models never saw: on each leg out of the water it
shows the leg, the **knee keypoint**, **where the leg enters the water**, and the **measurement**.
Plus a few numbers that show the measurement is right (checked against hand labels).

## The score (decided by Fadi)

Per leg, per frame:

- **Knee under water: FAIL**, also when the whole leg is under the water (Fadi, Nov 25).
- **Knee out of the water:** the share of the whole leg (toes to hip) that is out of the water.
  Toes-to-knee counts as half the leg, so: water at the knee = 50%, water halfway up the thigh =
  75%, water at the hip (whole leg out) = 100%.

How it is computed (the hip is usually hidden, so the thigh is taken as long as toes-to-knee):

```
out    = length along the leg from the toes to where it enters the water
         (toes -> knee -> water, following the bend if the knee is bent)
score  = out / (2 x toes-to-knee length), at most 100%
```

Lengths are in pixels, but the score is a ratio, so it doesn't depend on how far the camera is.

## Must work on

All videos filmed from above the water: TV broadcasts, fan/phone close-ups (our test videos), and
phone videos from Instagram. No waterline across the whole picture is needed: the water level is
where each leg's outline meets the water.

## Not in scope

- Leg left/right (each leg is scored on its own; "leg 1 / leg 2" is enough).
- Arms.
- Underwater and through-the-wall shots (no leg out of the water to measure).
- Teams (several swimmers in the picture); solos only for now. *(my call)*

## The pieces

| Piece | How | New or existing |
|---|---|---|
| Leg outline | the outline model (`seg_b3`): legs 0.60 boxes / 0.70 outlines on new pools | existing |
| Toes | the end of the leg's outline farthest from the water | new rule |
| Where the leg enters the water | the end of the leg's outline at the water (the bottom end when her feet point up) | new rule |
| Knee | a keypoint. First check the ready-made pose model's knees on the 87 test frames where Fadi already clicked knees; if they are not good enough, train a knee keypoint model on leg crops (the 192 labeled frames + new labels) | check, then maybe train |
| Knee out or under | knee found on the leg, above where the leg enters the water: out; else FAIL | new rule |
| Score | the formula above | new rule |
| Video | built on the meeting's model-output videos (`experiments/limbs/meeting_nov24.py`) | adapt |

**Main risk:** finding the knee. The ready-made pose model is weak on legs-only frames (its points
are within 10% of her size only 45% of the time). A missed knee would wrongly give FAIL, so the
video shows how sure the knee detection is.

## Labels (Fadi's time: today 2h, Fri 3h, Sat 3h, Sun 1h)

- **Test labels (first, today):** ~100 frames from the 4 test videos (later + Instagram), every leg
  out of the water: click the **toes**, the **knee** (or choose "knee under water"), and **where the
  leg enters the water**. About 40 s per frame, ~1-1.5 h. The true score comes from these clicks with
  the same formula.
- **Knee training labels (only if the ready-made knee is too weak):** ~300 leg frames from the
  training videos, same clicks, ~3-4 h (Fri/Sat).
- **Instagram:** Claude finds 3-5 public solo videos (legs out of the water, filmed from above);
  downloaded one by one with yt-dlp (with Fadi's browser login if Instagram asks for one); used only
  as extra test videos, not re-published.

## How we know it works

On the test labels:
- knee out / under: % right;
- score: average difference from the hand score, in percentage points;
- each point (toes, knee, water): distance from the hand click, as % of the leg length.

The "thigh as long as toes-to-knee" assumption is not tested by these labels (the hip is rarely
visible); it is part of the definition. *(my call)*

## Schedule

| Day | Fadi | Claude |
|---|---|---|
| Tue Nov 24 | approve this; label test frames (2h) | check ready-made knees on existing labels; set up test labeling |
| Wed-Thu | - | build points + score + video; first numbers; decide whether to train a knee model; set up training labels |
| Fri-Sat | label knees for training if needed (6h) | find Instagram videos; train knee model; Instagram tests |
| Sun-Mon | check the video (1h) | final numbers, video, meeting notes |

## My calls (to check)

- Solos only; teams later.
- In the video, besides the per-frame score, a summary per clip: share of frames that pass and the
  average score.

## Open (Fadi asks Assaf if needed)

- Legs pressed together, when the outline model draws one outline around both: not decided. If a
  decision is needed, Fadi asks Assaf; until then, report how often it happens.

## Left to the builder

Thresholds, crop sizes, model size, the exact video layout.
