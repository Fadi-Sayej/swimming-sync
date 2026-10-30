"""Compare the detected waterline with hand labels.

Usage:
    python -m swimsync.waterline.evaluate data/raw/solo.mp4

Reads:
    data/labels/<video>_waterline.json                      (from label.py)
    experiments/above_water/output/<video>/per_frame.csv    (from above_water.py)
    experiments/above_water/output/<video>/keypoints.json

Reports:
    waterline error   mean vertical distance (px) between detected and true line, across the image width
    above/below agree share of detected keypoints that get the same above/below answer with the true line
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .detect import Waterline, line_from_points


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--labels", type=Path, default=Path("data/labels"))
    p.add_argument("--results", type=Path, default=Path("experiments/above_water/output"))
    p.add_argument("--kp-conf", type=float, default=0.3)
    a = p.parse_args()

    width = int(cv2.VideoCapture(str(a.video)).get(cv2.CAP_PROP_FRAME_WIDTH))
    labels = json.loads((a.labels / f"{a.video.stem}_waterline.json").read_text())
    res_dir = a.results / a.video.stem
    with open(res_dir / "per_frame.csv") as f:
        pred = {int(r["frame"]): r for r in csv.DictReader(f)}
    kps = {e["frame"]: e["keypoints"] for e in json.loads((res_dir / "keypoints.json").read_text())}

    xs = np.linspace(0, width, 50)
    errors, agree, total_kp, missing = [], 0, 0, 0
    per_frame = []

    for f_str, pts in labels.items():
        f = int(f_str)
        gt = line_from_points(*pts)
        r = pred.get(f)
        if not r or not r.get("line_a"):
            missing += 1
            continue
        det = Waterline(float(r["line_a"]), float(r["line_b"]), 1.0)
        err = float(np.mean(np.abs(det.y_at(xs) - gt.y_at(xs))))
        errors.append(err)

        fa, fn = 0, 0
        for k in kps.get(f, []):
            if k["conf"] < a.kp_conf:
                continue
            same = bool(gt.is_above(k["x"], k["y"])) == k["above"]
            fa += same
            fn += 1
        agree += fa
        total_kp += fn
        per_frame.append({"frame": f, "waterline_err_px": round(err, 1),
                          "kp_agree_pct": round(100 * fa / fn, 1) if fn else None})

    report = {
        "labeled_frames": len(labels),
        "frames_without_detected_line": missing,
        "waterline_err_px_mean": round(float(np.mean(errors)), 1) if errors else None,
        "waterline_err_px_median": round(float(np.median(errors)), 1) if errors else None,
        "waterline_err_px_max": round(float(np.max(errors)), 1) if errors else None,
        "above_below_agreement_pct": round(100 * agree / total_kp, 1) if total_kp else None,
        "keypoints_checked": total_kp,
        "worst_frames": sorted(per_frame, key=lambda d: -d["waterline_err_px"])[:5],
    }
    (res_dir / "evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
