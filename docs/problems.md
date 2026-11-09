# Problems and fixes

One entry per problem: what went wrong, why, how we fixed it. Newest at the bottom.

## Setup

### 1. Python too old (2026-10-30)
- **Problem:** The Mac's system Python is 3.9; the project needs 3.10+.
- **Fix:** Built `.venv` from Miniconda's Python 3.13 (`~/miniconda3/bin/python3 -m venv .venv`).

### 2. yt-dlp not installed (2026-10-30)
- **Problem:** `yt-dlp` command not found.
- **Fix:** Installed it inside the venv (`pip install yt-dlp`).
- **Note:** It warns about a missing JavaScript runtime and ffmpeg. The download still worked (1080p, 25 fps).

## Milestone 1: waterline and % above water

Video: Worlds 2025 Singapore solo, underwater side camera that pans sideways (`data/raw/solo.mp4`).

### 3. Waterline found on the blue wall, not the water (2026-10-30)
- **Problem:** Auto waterline came out at y≈170 px instead of y≈420 px. The swimmer looked 100% underwater in every frame.
- **Why:** The detector took the *first* blue area from the top of each column. The dark-blue wall behind the judges is also blue.
- **Fix:** Take the start of the *tallest* blue area in each column instead (that is the pool).
- **Result:** Line now at y≈422 px; 90% of frames within 417–426 px.

### 4. Waterline tilted by blue ad banners (2026-10-30)
- **Problem:** Some frames gave a tilted line (up to ~120 px difference left to right).
- **Why:** Navy banners (SONY, World Aquatics) touch the water, so in those columns the "water" starts too high. The old line fit let them pull the line.
- **Fix:** RANSAC line fit: keep only the columns that agree with each other and ignore the rest.

### 5. Judges/spectators measured instead of the swimmer (2026-10-30)
- **Problem:** "% of body above water" jumped to 100% in 1,434 frames, so the average came out at 50% instead of ~23%.
- **Why:** The script picked the biggest person in the frame. When the model missed the swimmer underwater, the biggest person was a judge or spectator above the water.
- **Fix:** Pick the biggest person whose box reaches *below* the waterline. If none, the frame counts as "swimmer not found".
- **Result:** 100% frames went from 1,434 to 5. Body-mask average is now 23.2%, which agrees with the keypoint average (22.5%).

### 8. Labeling tool: pressing `n` did nothing (2026-11-02)
- **Problem:** Clicks worked in the labeling window, but `n` (next) did nothing, so no labels were saved.
- **Why (likely):** The tool only accepted a lowercase English `n`. A Hebrew/Arabic keyboard layout or Caps Lock sends a different key, and on macOS the window can also miss keyboard focus.
- **Fix:** Right-click = save and next (right-click with no points = skip), so the mouse alone is enough. Enter/Space also mean next, and keys work with Caps Lock. Unrecognized keys are printed in the terminal.

### 9. The waterline is a band, not a line (2026-11-02)
- **Problem:** While labeling, the water surface shows as a thin strip (~15 px), not a sharp line, so it was unclear where to click.
- **Why:** The camera sits right at the surface and sees it almost edge-on.
- **Decision:** Click the real water surface. Usually that is the bottom edge of the strip, where the solid blue starts. Frames where a splash hides the line are skipped.
- **Catch (found by Fadi):** In the start/end shots the water is higher than the blue tiles, so the top of the water is clear, not blue. The real surface is ~10 px above the blue there (see #11).

### 10. Smoothing carried a bad line across a scene cut (2026-11-02)
- **Problem:** Evaluation showed one frame (4051) off by 285 px.
- **Why:** The detector smooths the line over time. The logo screen just before frame 4051 gave a line at y=5, and the smoothing averaged it into the next pool shot (y=131 instead of ~426).
- **First wrong guess:** We thought 4051 was the logo screen and the label was a mistake. It wasn't: 4051 is a pool shot and the label was correct.
- **Fix (2nd try):** A line far (>40 px) from the current one is never averaged in. If most columns agree on it (confidence >= 0.5), it is a scene cut and the line restarts there at once. If few agree (e.g. splash, ~0.2), it is ignored unless it lasts 10 frames.
- **1st try failed:** "Restart if the new height holds 3 frames" missed frame 4051 (a 1-frame pool shot at the very end) and was fooled by a splash that held for 3 frames.
- **Result (all 50 labels):** 4.1 px mean error, 9.6 px max, 99.0% above/below agreement. Splash frames at 3.5 s are fixed too.

### 6. Waterline drawn on the end logo screen (fixed 2026-11-02)
- **Problem:** The last 3 s (frames 3976-4050) are an all-blue logo screen with no pool, but the detector put a line at the top (y=5) with high confidence.
- **Why:** The whole screen is blue, so in every column the "water" starts at the top row.
- **1st try (not enough):** Ignore columns whose water reaches the top edge (no surface in view). The logo screen then gave a weak line at y≈555 from the few columns under the logo text (5-9% of columns).
- **Fix:** (1) columns with water up to the top edge are skipped; (2) a line that fewer than 15% of columns agree on counts as "no waterline" (pool frames are always >= 21%, the logo <= 9%); (3) after 1 s (25 frames) with no waterline, the smoothing forgets the old line.
- **Result:** No line on frames 4000-4050. The first second of the logo still shows the last pool line (held on purpose so short misses like splashes keep a line). Pool frames, averages and the 50-label evaluation are unchanged.
- **Side note:** In 6 of those held frames (3987-3992) the pose model sees a "person" in the logo pattern and measures it at 0% above water. Same as before the fix; effect on the average is ~0.05 points.

## Milestone 2: leg and hand detectors

### 12. Long video downloads stopped halfway (2026-11-09)
- **Problem:** The 2-hour broadcast downloads (4-5 GB each) stopped at 3.4 GB with "Failed to resolve ... googlevideo.com".
- **Why:** The internet connection dropped for a few minutes; yt-dlp gave up after its default 10 retries.
- **Fix:** Re-ran with `--continue --retries 50 --fragment-retries 50`. It resumes from the `.part` file instead of starting over.
- **Also:** The fan-channel titles ("Solo Free 2026-5.The Netherlands") name the swimmer's country, not the pool. Checked each video's description for the real event before using it as a different-pool test video (see `docs/videos.md`).

### 13. A "different pool" test video was from the training city (2026-11-09)
- **Problem:** The Budapest 2024 World Cup video was picked as a test video from another pool, but the Junior Worlds 2026 training videos turned out to be from Budapest too (a "HUNGARY" banner in the broadcast; confirmed in the video description).
- **Why it matters:** Testing on the same pool as training makes the score look better than it would be on a new pool.
- **Fix:** Budapest 2024 stays as a test video but its score is reported separately. The main "new pool" test set is Doha 2024 + Paris 2025/2026.

### 14. Finding "swimmer in the pool" frames in a full broadcast (2026-11-09)
- **Problem:** Broadcasts mix pool shots with sofa, walk-on, interview and score-graphic shots. The frame picker must keep only the pool shots with the swimmer.
- **1st idea dropped:** Reuse the milestone 1 waterline detector to skip underwater views. From above, the pool edge also looks like a "waterline", so it would have dropped the wide shots too.
- **2nd idea dropped:** Share of blue in the frame. The navy sofa backdrop counts as "water" (0.81).
- **Fix:** Keep the biggest person whose surroundings (box grown by 50%) are at least 30% pool water. Pool water is a narrower colour range: hue 80-102 (cyan/teal), so the navy backdrop (hue ~106) is out.
- **Then:** That range was first tuned on the bright Junior Worlds pool only and missed the darker Paris and Budapest water (brightness ~125 vs a 140 cut). Lowered to saturation >= 50, brightness >= 90 after measuring all five pools; the sofa still fails. On 98 sample frames: no sofa/deck/interview shot kept, swimmer found in 44/50 pool frames where the model saw a person.
- **Known gap:** In the widest shots the swimmer is ~20-40 px tall and the pose model rarely finds her, so those shots are rare in the picks.
- **Still 16% junk:** Looking at all 400 first picks, 64 were not pool shots: presenters in front of a cyan stage backdrop, the empty arena before the session, countdown clocks, medals, sofa shots. The cyan backdrop has the same colour as the water.
- **3rd idea dropped:** Also require pool water just below the person's box. It kept 52 of the 64 (most boxes touch the bottom of the frame, and below the presenters is more cyan backdrop).
- **Final fix:** CLIP (an image-text model, no training) scores each frame against "a swimmer in a swimming pool"-type prompts vs "a person on a stage", "people on a sofa", "an empty arena", "a TV graphic" etc. Frames under 0.3 "pool" are not picked. On the same 400: removed 56 of 64 junk frames, dropped 5 of 336 good ones. After the re-pick ~5% junk is left (mostly animated transitions that show swimmers); those get skipped while labeling.

### 15. Label Studio rejected the API token (2026-11-09)
- **Problem:** Every API call to Label Studio returned "Invalid token.", even though the token was in its database.
- **Wrong guesses:** (1) new versions turn off old-style tokens, so I turned them on; (2) the server only reads that at start, so I restarted it. Neither helped.
- **Why:** `label-studio --version` (run to check the install) doesn't just print the version in 1.23: it starts a server on port 8080 with a default database. That stray server held the port, so my real server never started and every request went to the empty one.
- **Fix:** Stopped the stray server, deleted its empty default database, started the real one with `scripts/labelstudio.sh`. Don't run `label-studio --version`; use `pip show label-studio`.

### 16. Labeling: many small boxes per leg, and 42% of frames skipped (2026-11-09)
- **Problem 1:** Some frames had a leg labeled with 3-4 small boxes along it instead of one box. The model would learn that a leg is a short piece.
- **Fix:** Rule made explicit: one box per limb, from where it leaves the water to the toes / fingertips. Found the 6 frames with doubled boxes through the Label Studio API; Fadi merged them.
- **Problem 2:** 170 of 401 training frames were skipped (no limbs, TV graphics, or left/right unclear), leaving 231 labeled frames.
- **Response:** Upside-down rule for left/right (front of the legs visible = her left is on your left; back visible = on your right). Frames with no limbs should be submitted with no boxes, not skipped. Train both a 4-class (left/right) and a 2-class (leg/arm) model to see if the sides should come from the detector at all.

### 17. Image mirroring would teach the wrong side (2026-11-09)
- **Problem:** YOLO training mirrors images left-right by default (`fliplr=0.5`). A mirrored left leg looks like a right leg but keeps the `left_leg` label.
- **Fix:** `fliplr=0.0` in `scripts/train_limbs.sh`, for both versions so the comparison is fair.

### 18. Training results saved in the wrong folder (2026-11-09)
- **Problem:** `project=experiments/limbs/output` saved to `runs/detect/experiments/limbs/output/` instead.
- **Why:** Newer Ultralytics puts relative `project` paths inside `runs/<task>/`.
- **Fix:** Moved the results; `scripts/train_limbs.sh` now passes an absolute path.

## Still open

### 7. Pose model misses the swimmer underwater
- **Problem:** Swimmer found in only 74% of frames, mostly missed when fully underwater.
- **Impact:** This is the main problem the project is meant to solve (fine-tuning pose models for the pool).
- **Tried (2026-11-04):** The largest model of the same family (YOLO11x-pose, ~3x bigger) instead of YOLO11m-pose. No better: swimmer found 73.7% vs 74.1%. Each finds ~270 frames the other misses, but 714 pool frames (18%) are missed by both.
- **What the misses look like:** she lies flat at the surface (seen edge-on), dives under water in bubbles, or hasn't entered the pool yet (opening walk-on; that "miss" is correct).
- **Conclusion:** model size isn't the issue; the model has to be fine-tuned on pool frames.

### 11. Detector misses clear water above the blue (2026-11-02)
- **Problem:** In the start/end shots the detector puts the line ~10 px below the real surface.
- **Why:** It only looks for blue. Where the water rises above the blue tiles, the top of the water is clear, so it isn't counted.
- **Impact:** Small effect (~10 px, a few % of body height). A trained water segmentation model (pipeline step 3) should fix it.
