"""Leg height video (Task 6): on each leg the outline, toes (white), knee (yellow), where it enters
the water (cyan), and the result; a panel with each leg's score; a summary card per clip.

    python experiments/height/height_video.py                       # 10 s from each of the 4 test videos
    python experiments/height/height_video.py --video PATH --start 12 --seconds 15   # any video
Output: experiments/height/output/height_<name>.mp4 (+ height_all.mp4 for the test videos).
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parents[1] / "limbs"))
from meeting_nov24 import clip_starts, text  # noqa: E402

from swimsync.height.measure import SEG, OurKnees, PoseKnees, measure_frame  # noqa: E402
from swimsync.waterline.above_water import pick_device  # noqa: E402

OUT = Path("experiments/height/output")
GREEN, RED, WHITE, YELLOW, CYAN, GRAY = (60, 200, 60), (60, 60, 230), (255, 255, 255), (0, 220, 255), (255, 220, 0), (170, 170, 170)


def result_text(leg):
    if leg.status == "pass":
        return (f"knee out: {100 * leg.score:.0f}%" if leg.score is not None else "knee out (toes off the picture)"), GREEN
    if leg.status == "fail":
        return ("knee under water: FAIL" if leg.knee else "no knee found: FAIL"), RED
    return None, None


def draw(img, legs, title):
    out = img.copy()
    over = out.copy()
    sc = img.shape[1] / 1920
    th = max(2, round(3 * sc))
    for leg in legs:
        if leg.mask is not None:
            over[leg.mask] = (0.5 * over[leg.mask] + 0.5 * np.array((200, 200, 200))).astype(np.uint8)
    out = cv2.addWeighted(over, 0.5, out, 0.5, 0)
    lines = []
    for i, leg in enumerate(legs, 1):
        msg, col = result_text(leg)
        if msg is None:
            continue
        pts = [p for p in (leg.toes, leg.knee, leg.water) if p is not None]
        cv2.polylines(out, [np.array(pts, np.int32)], False, col, th)
        for p, c in ((leg.toes, WHITE), (leg.knee, YELLOW), (leg.water, CYAN)):
            if p is not None:
                cv2.circle(out, (int(p[0]), int(p[1])), round(12 * sc) + 2, c, -1)
                cv2.circle(out, (int(p[0]), int(p[1])), round(12 * sc) + 2, (0, 0, 0), max(1, th // 2))
        x1, y1 = int(leg.box[0]), int(leg.box[1])
        text(out, f"leg {i}: {msg}", (x1, max(y1 - 12, round(40 * sc))), 1.0 * sc, col, 2)
        conf = "" if leg.knee_conf is None else f"  (knee {leg.knee_conf:.2f} sure)"
        lines.append((f"leg {i}: {msg}{conf}", col))
    h = round((50 * max(len(lines), 1) + 110) * sc)
    cv2.rectangle(out, (0, 0), (round(1250 * sc), h), (20, 20, 20), -1)
    text(out, title, (round(16 * sc), round(46 * sc)), 1.1 * sc, (200, 200, 200), 2)
    for j, (ln, col) in enumerate(lines or [("no leg out of the water", GRAY)]):
        text(out, ln, (round(16 * sc), round((96 + 50 * j) * sc)), 1.1 * sc, col, 2)
    text(out, "white = toes, yellow = knee, cyan = enters the water; 50% = water at the knee, 100% = at the hip",
         (12, img.shape[0] - 16), 0.85 * sc, WHITE, 2)
    return out


def summary_card(shape, title, results):
    card = np.full(shape, 25, np.uint8)
    sc = shape[1] / 1920
    legs = [leg for r in results for leg in r if leg.status in ("pass", "fail")]
    passed = [leg for leg in legs if leg.status == "pass"]
    with_leg = sum(1 for r in results if any(leg.status in ("pass", "fail") for leg in r))
    rows = [title, f"frames with a leg out of the water: {with_leg} of {len(results)}",
            f"legs judged: {len(legs)}, knee out: {len(passed)} ({100 * len(passed) / max(len(legs), 1):.0f}%)",
            (f"average score when the knee is out: {100 * np.mean(scores):.0f}%"
             if (scores := [leg.score for leg in passed if leg.score is not None]) else "")]
    for i, r in enumerate(rows):
        text(card, r, (round(80 * sc), round((200 + 90 * i) * sc)), 1.6 * sc, WHITE, 3)
    return card


def run_clip(path, start, seconds, seg, knees, dev, name, combined=None):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
    title = f"{name}  (never seen in training)"
    writer, results, frame = None, [], None
    for _ in range(int(seconds * fps)):
        ok, img = cap.read()
        if not ok:
            break
        legs = measure_frame(img, seg, knees, dev)
        results.append(legs)
        frame = draw(img, legs, title)
        if writer is None:
            writer = cv2.VideoWriter(str(OUT / f"height_{name}.mp4"), cv2.VideoWriter_fourcc(*"avc1"), fps,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        if combined is not None:
            combined.write(cv2.resize(frame, (1920, 1080)))
    card = summary_card(frame.shape, title, results)
    for _ in range(int(2 * fps)):
        writer.write(card)
        if combined is not None:
            combined.write(cv2.resize(card, (1920, 1080)))
    writer.release()
    passed = sum(1 for r in results for leg in r if leg.status == "pass")
    print(f"{name}: {start:.1f}-{start + seconds:.1f} s, {len(results)} frames, {passed} legs with knee out", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--knees", choices=("ours", "pose"), default="ours")
    p.add_argument("--video")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--seconds", type=float, default=10.0)
    a = p.parse_args()
    dev = pick_device()
    OUT.mkdir(parents=True, exist_ok=True)
    seg = YOLO(SEG)
    knees = OurKnees(device=dev) if a.knees == "ours" else PoseKnees(device=dev)
    if a.video:
        run_clip(a.video, a.start, a.seconds, seg, knees, dev, Path(a.video).stem)
        return
    combined = cv2.VideoWriter(str(OUT / "height_all.mp4"), cv2.VideoWriter_fourcc(*"avc1"), 25, (1920, 1080))
    for v, t0 in clip_starts(a.seconds).items():
        run_clip(f"data/raw/test/{v}.mp4", t0, a.seconds, seg, knees, dev, v.split("_")[0], combined)
    combined.release()


if __name__ == "__main__":
    main()
