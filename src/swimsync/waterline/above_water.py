"""How much of the swimmer's body is out of the water, frame by frame.

Usage:
    python -m swimsync.waterline.above_water data/raw/solo.mp4
    python -m swimsync.waterline.above_water data/raw/solo.mp4 --line 0,410,1920,395   # fixed manual waterline
    python -m swimsync.waterline.above_water data/raw/solo.mp4 --seg                  # also use body masks

Writes to experiments/above_water/output/<video_name>/:
    annotated.mp4   skeleton, waterline, keypoints colored above (green) / below (red)
    per_frame.csv   waterline, % of keypoints above water, % of body mask above water, confidences
    keypoints.json  every keypoint with x, y, confidence, above/below
    above_water.png % above water over time
    summary.json    averages for the whole video
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .detect import SmoothedWaterline, detect_waterline, line_from_points

KP_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def pick_device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return 0
    return "cpu"


def main_swimmer(result):
    """Index of the biggest detected person (solo video = one swimmer)."""
    if result.boxes is None or len(result.boxes) == 0:
        return None
    xyxy = result.boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return int(np.argmax(areas))


def run(video, pose_model, seg_model, fixed_line, kp_conf, out_root):
    from ultralytics import YOLO

    out_dir = out_root / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device: {device}")

    pose = YOLO(pose_model)
    seg = YOLO(seg_model) if seg_model else None

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    smoother = SmoothedWaterline()
    rows, kp_log = [], []

    for i in tqdm(range(total or 10**9), desc="frames"):
        ok, frame = cap.read()
        if not ok:
            break
        r = pose.predict(frame, device=device, verbose=False)[0]
        idx = main_swimmer(r)

        box = r.boxes.xyxy[idx].cpu().numpy().tolist() if idx is not None else None
        if fixed_line is not None:
            line = fixed_line
        else:
            line = smoother.update(detect_waterline(frame, exclude_boxes=[box] if box else []))

        canvas = r.plot(boxes=False) if idx is not None else frame.copy()
        row = {"frame": i, "time_s": round(i / fps, 3)}

        if line is not None:
            cv2.line(canvas, (0, int(line.y_at(0))), (w, int(line.y_at(w))), (255, 255, 0), 2)
            row.update(line_a=round(line.a, 5), line_b=round(line.b, 1), line_conf=round(line.confidence, 2))

        if idx is not None and line is not None:
            kp = r.keypoints.data[idx].cpu().numpy()  # (17, 3): x, y, conf
            above = line.is_above(kp[:, 0], kp[:, 1])
            seen = kp[:, 2] >= kp_conf
            n_seen = int(seen.sum())
            n_above = int((above & seen).sum())
            row.update(
                kp_detected=n_seen,
                kp_above=n_above,
                pct_kp_above=round(100 * n_above / n_seen, 1) if n_seen else "",
                conf_above=round(float(kp[above, 2].mean()), 3) if above.any() else "",
                conf_below=round(float(kp[~above, 2].mean()), 3) if (~above).any() else "",
            )
            for (x, y, c), a in zip(kp, above):
                color = (0, 200, 0) if a else (0, 0, 255)
                cv2.circle(canvas, (int(x), int(y)), 6, color, -1 if c >= kp_conf else 1)
            kp_log.append({
                "frame": i,
                "keypoints": [
                    {"name": n, "x": float(x), "y": float(y), "conf": float(c), "above": bool(a)}
                    for n, (x, y, c), a in zip(KP_NAMES, kp, above)
                ],
            })

        if seg is not None and line is not None:
            s = seg.predict(frame, device=device, classes=[0], retina_masks=True, verbose=False)[0]
            j = main_swimmer(s)
            if j is not None and s.masks is not None:
                m = s.masks.data[j].cpu().numpy() > 0.5
                ys, xs = np.nonzero(m)
                if xs.size:
                    row["pct_mask_above"] = round(100 * float(line.is_above(xs, ys).mean()), 1)

        if "pct_kp_above" in row and row["pct_kp_above"] != "":
            txt = f"above water: {row['pct_kp_above']}% of keypoints"
            if "pct_mask_above" in row:
                txt += f" | {row['pct_mask_above']}% of body"
            cv2.putText(canvas, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        writer.write(canvas)
        rows.append(row)

    cap.release()
    writer.release()

    cols = ["frame", "time_s", "line_a", "line_b", "line_conf", "kp_detected", "kp_above",
            "pct_kp_above", "pct_mask_above", "conf_above", "conf_below"]
    with open(out_dir / "per_frame.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    (out_dir / "keypoints.json").write_text(json.dumps(kp_log))

    summary = summarize(rows, video, pose_model)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(rows, out_dir / "above_water.png")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out_dir}")


def _mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) not in (None, "")]
    return round(float(np.mean(vals)), 3) if vals else None


def summarize(rows, video, model):
    n = len(rows)
    return {
        "video": video.name,
        "model": model,
        "frames": n,
        "pct_frames_swimmer_found": round(100 * sum("kp_detected" in r for r in rows) / n, 1) if n else 0,
        "pct_frames_waterline_found": round(100 * sum("line_a" in r for r in rows) / n, 1) if n else 0,
        "mean_pct_kp_above": _mean(rows, "pct_kp_above"),
        "mean_pct_mask_above": _mean(rows, "pct_mask_above"),
        "mean_conf_above_water": _mean(rows, "conf_above"),
        "mean_conf_below_water": _mean(rows, "conf_below"),
    }


def plot(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    for key, label in [("pct_kp_above", "% keypoints above water"), ("pct_mask_above", "% body mask above water")]:
        pts = [(r["time_s"], r[key]) for r in rows if r.get(key) not in (None, "")]
        if pts:
            t, v = zip(*pts)
            ax.plot(t, v, label=label, linewidth=1.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("% above water")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--model", default="yolo11m-pose.pt", help="Ultralytics pose model (n/s/m/l/x)")
    p.add_argument("--seg", action="store_true", help="also measure % of body mask above water")
    p.add_argument("--seg-model", default="yolo11m-seg.pt")
    p.add_argument("--line", help="fixed waterline x1,y1,x2,y2 (static camera); skips auto detection")
    p.add_argument("--kp-conf", type=float, default=0.3, help="min confidence to count a keypoint as detected")
    p.add_argument("--out", type=Path, default=Path("experiments/above_water/output"))
    a = p.parse_args()

    fixed = line_from_points(*map(float, a.line.split(","))) if a.line else None
    run(a.video, a.model, a.seg_model if a.seg else None, fixed, a.kp_conf, a.out)


if __name__ == "__main__":
    main()
