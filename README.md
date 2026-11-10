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

Waterline v1 = blue/cyan color threshold; in each column the waterline is where the tallest
stretch of water starts (so blue walls/banners above the pool are skipped), then a RANSAC line
fit ignores columns that disagree. Frames with no surface in view (e.g. an all-blue logo
screen) give no waterline. The swimmer's box is skipped. The swimmer is the biggest
person whose box reaches below the waterline, so judges and spectators are never picked.
If the pool color differs, tune `hue`/`min_sat` in `src/swimsync/waterline/detect.py`.

Result: ~23% of the body above water on average, waterline error 4.1 px on 50 hand-labeled
frames. The pose model misses the swimmer in ~26% of frames, and a bigger model doesn't help.

## Milestone 2 (next meeting Nov 24, 2026): leg and hand detectors

Agreed with Assaf on Nov 9. Videos where the camera is above the water.

1. **Leg detector:** a bounding-box model that finds only the legs, one box per leg (left / right).
2. **Hand detector:** the same for hands/arms.
3. **Pose inside the boxes:** once the boxes work, train the pose model on the cropped leg and hand regions.
4. **Segmentation:** water / splash / swimmer. Start with SAM 2 (no training) to make masks,
   then fine-tune YOLO11-seg on them.

Plan:
- Collect above-water videos (YouTube, different pools and routines).
- Extract frames and label leg and hand boxes in CVAT or Label Studio (start with ~300-500 frames).
- Fine-tune YOLO11 detection on the Mac (MPS); use the college GPU if training is too slow.
- Evaluate with mAP50 on held-out frames from videos not used in training.

```bash
# pick ~400 train + ~100 test frames, with pre-filled boxes from the pose model
python -m swimsync.dataset.pick_frames --train data/raw/jr2026_*.mp4 --test data/raw/test/*.mp4
# label them in Label Studio (http://localhost:8080)
scripts/labelstudio.sh
```

How to label: [docs/labeling_guide.md](docs/labeling_guide.md). Videos: [docs/videos.md](docs/videos.md).

After labeling, build the training sets from Label Studio and train (Label Studio must be running):

```bash
python -m swimsync.dataset.export_yolo   # data/raw/datasets/m2_4cls and m2_2cls
scripts/train_limbs.sh                   # YOLO11s, both versions; DEVICE=0 on the college GPU
```

Leg/arm outlines (segmentation) from the same boxes, with SAM 2 (no drawing):

```bash
python -m swimsync.dataset.limb_masks    # outlines + data/raw/datasets/m2_seg; test outlines go to Label Studio for review
scripts/train_limbs.sh seg               # YOLO11s-seg
EPOCHS=200 NAME_SUFFIX=_200ep scripts/train_limbs.sh seg   # longer run, keeps the earlier results
```

Batch 3 (close-up fan videos from 5 new pools, `data/raw/train3/`):

```bash
python -m swimsync.dataset.pick_frames --train data/raw/train3/*.mp4 --split train3 --n-each 15 --step 1
python -m swimsync.dataset.labelstudio train3   # project "M2 legs+arms - train3"
```

## Tests

```bash
pytest
```

## Work log

See [docs/worklog.md](docs/worklog.md).
Problems we hit and how we fixed them: [docs/problems.md](docs/problems.md).
Videos used and where they come from: [docs/videos.md](docs/videos.md).
