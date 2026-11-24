# Leg Height Score Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Per leg and per frame, say whether the knee is out of the water and what share of the leg (toes to hip) is out, and show it on video.

**Architecture:** The existing outline model (`seg_b3`) gives each visible leg's outline; rules turn the outline into a toes point and a water point; a knee finder (ready-made pose first, our own knee model only if needed) gives the knee; a score rule turns toes/knee/water into pass/fail and a percentage. Checked against new hand labels in Label Studio ("M3 leg height - test").

**Tech Stack:** Python 3.13 venv, Ultralytics 8.4.193 (YOLO11), OpenCV, NumPy, Label Studio 1.23.2 (local), pytest, yt-dlp.

**Spec:** `docs/superpowers/specs/2026-11-24-leg-height-design.md`

## Global Constraints

- Due Tue Dec 1, 2026, 14:00. Fadi's labeling time: Tue 2h, Fri 3h, Sat 3h, Sun 1h.
- Score: knee under water = FAIL; knee out = `out / (2 x toes-to-knee)`, at most 100%, where `out` = length along the leg toes -> knee -> water. Water at the knee = 50%, mid-thigh = 75%, hip = 100%.
- A leg not out of the water gets no score (not FAIL).
- Legs pressed together (one outline for both): not decided; count how often, don't design around it.
- Solos only. Videos filmed from above the water. No full-frame waterline.
- Git: Fadi commits and pushes. Every "commit" step = give Fadi a copy-paste block, no Co-Authored-By line, never include `.gitignore`.
- Anything touching Label Studio: filter output with `grep -v -i "email\|password\|token"`; never print `data/raw/labelstudio/credentials.txt`.
- `source .venv/bin/activate` before python; one training at a time on the Mac; long jobs in the background.
- Every problem hit goes in `docs/problems.md` (short entry), every day in `docs/worklog.md`.

## Review Focus

1. **Knee out but not found** (blurry, splash): the leg would get FAIL wrongly. Per the spec, no knee found = FAIL (a knee under water is a knee the model can't find); the video shows the knee confidence and Task 5 counts false FAILs from missed knees separately. Pinned in Task 3 (`test_no_knee_found_is_a_fail`).
2. **Leg lying flat at the surface** (ballet leg, split on the surface): which end is the water is ambiguous. Expected: no crash; the lower end in the image is taken as the water; Task 3 test `test_flat_leg_takes_lower_end_as_water`.
3. **Knee found below the water point** (a knee under water seen through clear water): expected FAIL. Task 3 `test_knee_past_the_water_is_under`.
4. **Tiny or empty outline** (a few pixels of foot): expected no score, no crash. Task 3 `test_tiny_mask_gives_no_ends`.
5. **Toes and knee at the same place** (zero shin length): expected no score instead of a division by zero. Task 3 `test_zero_shin_gives_no_score`.

---

### Task 1: How good are the ready-made pose model's knees?

Decides whether we must train our own knee model (Task 8). Uses only labels that already exist: the knee clicks in "M2 keypoints - test" (87 frames) and the leg boxes in `data/raw/datasets/m2_4cls` (test).

**Files:**
- Create: `experiments/height/knee_check.py`
- Create: `experiments/height/results.md` (first section)

**Interfaces:**
- Consumes: `swimsync.dataset.export_yolo.labeled_frames(api, title) -> {name: result}`, `swimsync.dataset.export_pose.labeled_points(result) -> {point name: (x, y)}`, `swimsync.limbs.sides.pose_people(model, img, boxes=None, device=None) -> [array (17, 3)]`, `swimsync.dataset.facing._api()`.
- Produces: the numbers in `experiments/height/results.md`; nothing imported later.

- [ ] **Step 1: Write `knee_check.py`.** For every test frame with hand leg boxes (classes 0, 1 of the 4-class labels): a hand leg box "has a knee out" if a hand `left_knee`/`right_knee` click lies inside it (box grown 10%). Run yolo11m-pose (a) on the full frame and (b) zoomed (`pose_people(..., boxes=leg boxes)`); predicted knees = COCO 13, 14 with conf >= 0.3. Per leg box report: knee out per hand vs predicted knee inside the box (agreement %), and for hand knees, distance to the nearest predicted knee as % of the box's longer side (share within 10% / 20%). Sides are ignored (nearest knee).
- [ ] **Step 2: Run it.** `source .venv/bin/activate && python experiments/height/knee_check.py 2>&1 | grep -v -i "email\|password\|token"`. Expected: a table for full frame and zoomed.
- [ ] **Step 3: Write the result** in `experiments/height/results.md` ("Ready-made knees, 2026-11-24") with the reading: **train our own knee model (Task 8) if the better variant finds the knee within 20% on fewer than 80% of knees or agrees on knee out/under on fewer than 85% of legs.**
- [ ] **Step 4: Commit block for Fadi** (`experiments/height/knee_check.py experiments/height/results.md`).

### Task 2: Test labeling project "M3 leg height - test"

Fadi labels today, so this comes right after Task 1.

**Files:**
- Create: `src/swimsync/dataset/height_labels.py`
- Test: `tests/test_height_labels.py`

**Interfaces:**
- Consumes: `LabelStudioAPI` (`post`, `get`), `project_titles`, `labeled_frames`, the 4-class test labels for the leg guide boxes.
- Produces:
  - `TITLE = "M3 leg height - test"`
  - `height_config() -> str` (Label Studio XML)
  - `height_task(image_url: str, leg_boxes: list, w: int, h: int) -> dict` (legs sorted left to right in the image: leg 1 = left)
  - `parse_height(result: list) -> dict[int, dict]` -> `{1: {"status": str, "toes": (x, y) | None, "knee": ..., "water": ...}, 2: {...}}` in pixels; status one of `"knee out"`, `"knee under water"`, `"not out of the water"`, `"pressed to leg 1"` (leg 2 only).
  - `python -m swimsync.dataset.height_labels` creates the project (frames = M2 test frames with >= 1 hand leg box; images from `data/raw/frames/m2/images/test`), never imports twice; `... add <folder>` adds frames from another folder (Instagram, Task 7).

Label config: KeyPointLabels `points` with values `1 toes`, `1 knee`, `1 water`, `2 toes`, `2 knee`, `2 water` (hotkeys q w e / a s d; leg 1 blue, leg 2 orange); Choices `leg1` (required: knee out, knee under water, not out of the water; hotkeys 1 2 3) and `leg2` (required: same + "pressed to leg 1"; hotkeys 4 5 6 7); RectangleLabels `guide` (value `leg`, opacity 0.1) pre-filled with the hand leg boxes. Header text: "Leg 1 = the leg on the image's left. Click toes, knee (only if out of the water) and where the leg enters the water."

- [ ] **Step 1: Failing tests** in `tests/test_height_labels.py`:
  - `test_config_has_six_points_and_two_leg_questions`: `height_config()` contains `value="1 toes"`, `value="2 water"`, `name="leg1"`, `name="leg2"`, `pressed to leg 1`.
  - `test_task_orders_legs_left_to_right`: `height_task(url, [(500,0,600,100),(100,0,200,100)], 1000, 500)` -> the first guide box has x = 10.0 (percent).
  - `test_parse_reads_points_in_pixels_and_status`: a result with `1 toes` at (50%, 20%) on a 1000x500 image and `leg1` = "knee under water" -> `parse_height(r)[1] == {"status": "knee under water", "toes": (500.0, 100.0), "knee": None, "water": None}`.
  - `test_parse_missing_leg_answer_is_none_status`: no `leg2` choice -> `parse_height(r)[2]["status"] is None`.
- [ ] **Step 2: Run** `python -m pytest tests/test_height_labels.py -q` -> FAIL (module missing).
- [ ] **Step 3: Implement** `height_labels.py` (reuse `labelstudio.LabelStudioAPI`, the local-storage pattern of `facing.sideview`).
- [ ] **Step 4: Run** the tests -> PASS; full suite `python -m pytest -q` -> all pass.
- [ ] **Step 5: Create the project** `python -m swimsync.dataset.height_labels 2>&1 | grep -v -i "email\|password\|token"`. Expected: `'M3 leg height - test': project N, ~70-80 frames`. Open one task in http://localhost:8080 and check the guide boxes sit on the legs.
- [ ] **Step 6: Labeling guide**: add "Leg height (M3)" to `docs/labeling_guide.md` (what to click, the four statuses, legs pressed together -> leg 2 "pressed to leg 1"). Tell Fadi the project is ready.
- [ ] **Step 7: Commit block for Fadi.**

### Task 3: Score rules (toes, water, knee -> pass/fail and %)

**Files:**
- Create: `src/swimsync/height/__init__.py`, `src/swimsync/height/score.py`
- Test: `tests/test_height_score.py`

**Interfaces:**
- Produces:
  - `leg_ends(mask: np.ndarray, min_pixels: int = 200) -> tuple[tuple, tuple] | None`: (toes, water) in pixels. Ends along the mask's main axis (PCA), each end = mean of the mask pixels in the last 5% along the axis; water = the end lower in the image (larger y). None if fewer than `min_pixels` pixels.
  - `knee_is_out(toes, knee, water, margin: float = 0.05) -> bool`: the knee's position along toes -> water is between `margin` and `1 - margin`.
  - `height_score(toes, knee, water) -> float | None`: `min(1, (|toes-knee| + |knee-water|) / (2 |toes-knee|))`; None if `|toes-knee| < 1` px.
  - `@dataclass LegHeight: status: str` ("pass" | "fail" | "no knee found" | "no score"), `score: float | None`, `toes`, `knee`, `water` (tuples or None), `knee_conf: float | None`.
  - `judge_leg(toes, water, knee, knee_conf) -> LegHeight`: no toes/water -> "no score"; knee None -> "no knee found"; knee not out -> "fail" (score None); else "pass" with `height_score`.

- [ ] **Step 1: Failing tests** in `tests/test_height_score.py`:
  - `test_water_at_knee_is_half`: toes (0,0), knee (0,100), water (0,100) -> `height_score == 0.5`.
  - `test_water_mid_thigh_is_three_quarters`: toes (0,0), knee (0,100), water (0,150) -> 0.75.
  - `test_whole_leg_out_is_capped_at_full`: water (0,260) -> 1.0.
  - `test_bent_leg_measures_along_the_leg`: toes (0,0), knee (100,0), water (100,50) -> 0.75.
  - `test_zero_shin_gives_no_score`: toes == knee -> None.
  - `test_knee_past_the_water_is_under`: toes (0,0), water (0,100), knee (0,120) -> `knee_is_out` False and `judge_leg(...).status == "fail"`.
  - `test_no_knee_is_not_a_fail`: `judge_leg((0,0), (0,100), None, None).status == "no knee found"`.
  - `test_vertical_mask_ends`: a 20x200 vertical bar mask (rows 50-250) -> toes y < 60, water y > 240.
  - `test_flat_leg_takes_lower_end_as_water`: a bar tilted 10 degrees, rising to the right -> water is the left (lower) end; no exception.
  - `test_tiny_mask_gives_no_ends`: 50-pixel mask -> None.
- [ ] **Step 2: Run** `python -m pytest tests/test_height_score.py -q` -> FAIL.
- [ ] **Step 3: Implement** `score.py`.
- [ ] **Step 4: Run** -> PASS; full suite passes.
- [ ] **Step 5: Commit block for Fadi.**

### Task 4: Measure a frame (outline model + knee finder + rules)

**Files:**
- Create: `src/swimsync/height/measure.py`
- Test: `tests/test_height_measure.py`

**Interfaces:**
- Consumes: Task 3 (`leg_ends`, `judge_leg`, `LegHeight`), `swimsync.limbs.sides.pose_people`, outline model weights `experiments/limbs/output/seg_b3/weights/best.pt`.
- Produces:
  - `pick_knee(people: list[np.ndarray], mask: np.ndarray, conf_min: float = 0.3, grow: int = 15) -> tuple[float, float, float] | None`: among COCO knees (13, 14) of all people, those inside the mask grown by `grow` px; highest confidence; (x, y, conf).
  - `class PoseKnees: __init__(self, weights="yolo11m-pose.pt", device=None, zoom=True)`; `knees(self, img, legs: list[tuple[box, mask]]) -> list[tuple | None]` (one per leg; zoom = `pose_people(..., boxes=leg boxes)`, choice from Task 1's better variant).
  - `measure_frame(img, seg_model, knee_finder, device, conf: float = 0.25) -> list[LegHeight]`: one per leg outline of the outline model (class "leg"), masks from `r.masks.data` resized to the image; legs sorted left to right.
- [ ] **Step 1: Failing tests** in `tests/test_height_measure.py`:
  - `test_pick_knee_takes_the_surest_knee_inside_the_leg`: two people, knees inside the mask at conf 0.5 and 0.9 -> the 0.9 one.
  - `test_pick_knee_ignores_knees_outside_the_leg`: knee 100 px outside -> None.
  - `test_pick_knee_ignores_unsure_knees`: conf 0.2 -> None.
- [ ] **Step 2: Run** -> FAIL. **Step 3: Implement.** **Step 4: Run** -> PASS; full suite passes.
- [ ] **Step 5: Smoke run** on one test frame: `python -c "..."` printing the `LegHeight`s for `data/raw/frames/m2/images/test/paris2026_wc_reijnen_00029.0s.jpg`; expected: 1-2 legs, each with a status.
- [ ] **Step 6: Commit block for Fadi.**

### Task 5: Score against Fadi's labels

Runs once Fadi has labeled (Tue evening at the earliest); rerun after Task 8 if a knee model is trained.

**Files:**
- Create: `experiments/height/height_eval.py`
- Modify: `experiments/height/results.md`

**Interfaces:**
- Consumes: `parse_height` (Task 2), `measure_frame`, `PoseKnees` (Task 4), `knee_is_out`, `height_score` (Task 3).
- Produces: numbers in `results.md`; `--knees {pose,ours}` option for Task 8.

Matching: a labeled leg (with toes clicked) matches the predicted leg whose toes point is nearest, if closer than 25% of the labeled toes -> water length; otherwise "leg not found". Hand truth per leg: status from the label; hand score = `height_score(toes, knee, water)` from the clicks when "knee out".

- [ ] **Step 1: Write `height_eval.py`** reporting, per test video group (new pools / Budapest / Instagram):
  - legs labeled out of the water; found by the outline model;
  - knee out/under agreement on found legs, and how many "no knee found";
  - for legs both call "knee out": mean absolute score difference (percentage points);
  - point errors (toes, knee, water) as % of the hand toes -> water length;
  - count of "pressed to leg 1".
- [ ] **Step 2: Run** `python experiments/height/height_eval.py 2>&1 | grep -v -i "email\|password\|token"`.
- [ ] **Step 3: Write the section** "Leg height on the test frames" in `results.md` with a reading; log any problem in `docs/problems.md`.
- [ ] **Step 4: Commit block for Fadi.**

### Task 6: The video

**Files:**
- Create: `experiments/height/height_video.py`

**Interfaces:**
- Consumes: `measure_frame`, `PoseKnees`, `LegHeight`; clip windows as in `experiments/limbs/meeting_nov24.py` (`clip_starts`).
- Produces: `experiments/height/output/height_<video>.mp4` per test video, `height_all.mp4` joined, and `--video PATH --start S --seconds N` for any video (Instagram).

Per frame: each leg's outline (light), a line toes -> knee -> water, dots at toes (white), knee (yellow), water (cyan); next to the leg "knee out: 72%" (green) or "knee under water: FAIL" (red) or "no knee found" (yellow). Corner panel: video name ("never seen in training"), each leg's result and knee confidence. End of each clip: 2 s card with share of frames passing and average score. Text sized like the meeting videos (readable when shared on screen); H.264 (`avc1`).

- [ ] **Step 1: Write `height_video.py`.** **Step 2: Run** in the background on the 4 test videos. **Step 3: Check 3 frames per clip** (grab with OpenCV, look at them) for readable text and dots on the legs. **Step 4: Commit block for Fadi.**

### Task 7: Instagram videos (Claude finds them)

**Files:**
- Modify: `docs/videos.md` (new section "Milestone 3: phone videos (Instagram)")
- Data: `data/raw/instagram/` (not in git), frames in `data/raw/frames/m2/images/instagram/`

- [ ] **Step 1: Find 3-5 public Instagram reels** of solo artistic swimming filmed from above the water with legs out (verticals), by web search (`site:instagram.com/reel artistic swimming solo`, coaches' and clubs' public accounts). Not team routines.
- [ ] **Step 2: Download one by one**: `yt-dlp -o "data/raw/instagram/%(id)s.mp4" <url>`; if Instagram asks for a login, ask Fadi before using `--cookies-from-browser chrome` (his login). Log failures in `docs/problems.md`.
- [ ] **Step 3: Record** each video (link, account, what it shows) in `docs/videos.md`; for research use only, not re-published.
- [ ] **Step 4: Pick ~6 frames per video** where the outline model finds a leg (spread >= 1 s apart) into `data/raw/frames/m2/images/instagram/` and add them to the labeling project: `python -m swimsync.dataset.height_labels add instagram`.
- [ ] **Step 5: Run Task 6's video** on each (`--video`). **Step 6: Commit block for Fadi.**

### Task 8 (only if Task 1 or Task 5 says the ready-made knees are too weak): our own knee model

**Files:**
- Create: `src/swimsync/dataset/export_knees.py`, `tests/test_export_knees.py`
- Modify: `src/swimsync/height/measure.py` (add `class OurKnees` with the same `knees(img, legs)` interface), `scripts/train_limbs.sh` (case `knees*`: `task=pose`, `model=yolo11s-pose.pt`, `data=data/raw/datasets/m3_knees/data.yaml`)

**Interfaces:**
- Produces: `knee_line(knee: tuple | None, leg_box, crop) -> str` (YOLO pose line, one object "leg" with 1 keypoint, visibility 0 when no knee); dataset `data/raw/datasets/m3_knees` (`kpt_shape: [1, 3]`, `flip_idx: [0]`, crops around each leg box padded 30%); `OurKnees(weights=".../knees/weights/best.pt")`.

- [ ] **Step 1: Failing tests**: `test_knee_line_normalises_to_the_crop` (knee at the crop's centre -> `0 0.5 ... 0.5 0.5 2`), `test_no_knee_gives_invisible_point` (-> `... 0 0 0`).
- [ ] **Step 2-4:** run -> FAIL, implement, run -> PASS.
- [ ] **Step 5: Training labels**: knees from "M2 keypoints - train" (192 frames) first; if Fadi has time Fri/Sat, a "M3 leg height - train" project (same config as Task 2, ~300 frames from the training videos) adds more.
- [ ] **Step 6: Train** `IMGSZ=320 BATCH=16 EPOCHS=100 scripts/train_limbs.sh knees` in the background (mirroring is safe: no sides).
- [ ] **Step 7: Rerun Task 5** with `--knees ours`; keep whichever knee finder scores better; update `results.md`.
- [ ] **Step 8: Commit block for Fadi.**

### Task 9: Meeting materials and docs (Mon Nov 30)

**Files:**
- Create: `experiments/height/output/meeting_dec1/` (notes PDF + videos copied/linked)
- Modify: `docs/worklog.md`, `docs/project_explained.md` (new section "Milestone 3: leg height"), `README.md` (commands), `docs/problems.md`

- [ ] **Step 1:** Notes for Fadi (what to show, what to say, the numbers and what they mean, open questions incl. legs pressed together), as `.md` + PDF, in the style of the Nov 24 notes.
- [ ] **Step 2:** Update the docs listed above. **Step 3:** Final full test run `python -m pytest -q`. **Step 4:** Commit block for Fadi.
