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

## Still open

### 6. Waterline wrong on the end logo screen
- **Problem:** The last ~3 s are a logo screen with no pool, but the detector still returns a line there.
- **Impact:** No swimmer there, so no effect on the averages. Not fixed yet.

### 7. Pose model misses the swimmer underwater
- **Problem:** Swimmer found in only 74% of frames, mostly missed when fully underwater.
- **Impact:** This is the main problem the project is meant to solve (fine-tuning pose models for the pool).

### 11. Detector misses clear water above the blue (2026-11-02)
- **Problem:** In the start/end shots the detector puts the line ~10 px below the real surface.
- **Why:** It only looks for blue. Where the water rises above the blue tiles, the top of the water is clear, so it isn't counted.
- **Impact:** Small effect (~10 px, a few % of body height). A trained water segmentation model (pipeline step 3) should fix it.
