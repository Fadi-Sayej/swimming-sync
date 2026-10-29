"""Run an off-the-shelf pose model on a swimming video.

Usage:
    python -m swimsync.baseline.run_pose data/raw/video.mp4 [--model yolo11n-pose.pt] [--conf 0.25]

Writes to experiments/baseline/output/<video_name>/:
    annotated.mp4   video with skeletons and track IDs drawn
    keypoints.json  per frame: track id, box, 17 COCO keypoints (x, y, confidence)
    summary.json    detection rate and mean keypoint confidence
"""

import argparse
import json
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO


def run(video_path: Path, model_name: str, conf: float, out_root: Path) -> None:
    out_dir = out_root / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    writer = cv2.VideoWriter(
        str(out_dir / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frames = []
    frames_with_people = 0
    kp_conf_sum = 0.0
    kp_count = 0

    # stream=True keeps memory low; persist=True keeps track IDs across frames
    results = model.track(
        source=str(video_path),
        conf=conf,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    for i, r in enumerate(tqdm(results, total=total or None, desc="frames")):
        writer.write(r.plot())

        people = []
        if r.boxes is not None and len(r.boxes) > 0:
            frames_with_people += 1
            ids = r.boxes.id.int().tolist() if r.boxes.id is not None else [None] * len(r.boxes)
            boxes = r.boxes.xyxy.tolist()
            kps = r.keypoints.data.tolist() if r.keypoints is not None else [[]] * len(boxes)
            for tid, box, kp in zip(ids, boxes, kps):
                people.append({"track_id": tid, "box": box, "keypoints": kp})
                for _, _, c in kp:
                    kp_conf_sum += c
                    kp_count += 1

        frames.append({"frame": i, "people": people})

    writer.release()

    (out_dir / "keypoints.json").write_text(json.dumps(frames))

    n = len(frames)
    summary = {
        "video": video_path.name,
        "model": model_name,
        "frames": n,
        "frames_with_people_pct": round(100 * frames_with_people / n, 1) if n else 0,
        "mean_keypoint_confidence": round(kp_conf_sum / kp_count, 3) if kp_count else 0,
        "unique_track_ids": len(
            {p["track_id"] for f in frames for p in f["people"] if p["track_id"] is not None}
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--model", default="yolo11n-pose.pt", help="any Ultralytics pose model")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--out", type=Path, default=Path("experiments/baseline/output"))
    a = p.parse_args()
    run(a.video, a.model, a.conf, a.out)


if __name__ == "__main__":
    main()
