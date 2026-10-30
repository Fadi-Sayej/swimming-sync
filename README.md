# Swimming Sync

Pose estimation and synchronization scoring for artistic swimming teams.

Final project, Software Engineering, Azrieli College of Engineering Jerusalem.
Student: Fadi Sayej · Supervisor: Dr. Assaf Spanier

## Goal

Off-the-shelf pose models fail in the pool: underwater limbs are blurry, splashes hide joints,
and the waterline splits the body. This project trains and compares several models to fix that,
then uses the best pipeline to score how synchronized the swimmers are and to power a coach tool
(skeleton overlay + out-of-sync highlights).

## Pipeline

1. **Detect + track** each swimmer (fine-tuned detector + ByteTrack + small re-ID model)
2. **Pose** per swimmer (fine-tuned YOLO-pose / RTMPose + temporal smoothing)
3. **Water & splash segmentation** (water surface, splash, swimmer)
4. **Sync score** from joint angles over time
5. **Coach app**: upload a video, see skeletons and out-of-sync moments

Every model is evaluated on the same test set split by condition:
above water · underwater · splash · waterline-crossing.

## Repo layout

```
data/raw/          videos (not committed)
data/labels/       annotations (CVAT / Label Studio exports)
src/swimsync/      project code
  baseline/        off-the-shelf pose baseline
experiments/       configs and results per experiment
notebooks/         exploration
app/               coach app (FastAPI + React)
docs/              work log, reports
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Run the baseline

Put a video in `data/raw/`, then:

```bash
python -m swimsync.baseline.run_pose data/raw/video.mp4
```

Output goes to `experiments/baseline/`: an annotated video and a JSON file with keypoints per frame.

## Milestone 1 (due Nov 9): how much of the body is out of the water

Solo swimmer, good-quality video.

```bash
# 0. get a video (pip install yt-dlp)
yt-dlp -f "bv*[height<=1080][ext=mp4]" -o "data/raw/solo.mp4" "<youtube-url>"

# 1. run pose + waterline + % above water
python -m swimsync.waterline.above_water data/raw/solo.mp4 --seg

#    camera doesn't move? give the waterline by hand instead of auto-detecting it:
python -m swimsync.waterline.above_water data/raw/solo.mp4 --line x1,y1,x2,y2

# 2. hand-label the true waterline on 50 frames (click 2 points per frame)
python -m swimsync.waterline.label data/raw/solo.mp4 --n 50

# 3. compare detected vs. true waterline
python -m swimsync.waterline.evaluate data/raw/solo.mp4
```

Output in `experiments/above_water/output/solo/`: annotated video (keypoints green = above water,
red = below), `per_frame.csv`, `above_water.png` graph, `summary.json`, `evaluation.json`.

Waterline v1 = blue/cyan color threshold + robust line fit, skipping the swimmer's box.
If the pool color differs, tune `hue`/`min_sat` in `src/swimsync/waterline/detect.py`.

## Work log

See [docs/worklog.md](docs/worklog.md).
