"""Hand-label the true waterline on sampled frames (ground truth for evaluation).

Usage:
    python -m swimsync.waterline.label data/raw/solo.mp4 --n 50

For each frame: click 2 points on the waterline (left side, then right side).
Keys:  n = next frame (save)   r = redo clicks   s = skip frame   q = quit and save

Saves data/labels/<video_name>_waterline.json  as  {frame: [x1, y1, x2, y2]}
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", type=Path)
    p.add_argument("--n", type=int, default=50, help="number of frames to label (evenly spaced)")
    p.add_argument("--out", type=Path, default=Path("data/labels"))
    a = p.parse_args()

    cap = cv2.VideoCapture(str(a.video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = np.linspace(0, total - 1, a.n).astype(int).tolist()

    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"{a.video.stem}_waterline.json"
    labels = json.loads(path.read_text()) if path.exists() else {}

    clicks = []

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append((x, y))

    win = "waterline: click 2 points | n=next r=redo s=skip q=quit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_click)

    for k, f in enumerate(frames):
        if str(f) in labels:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            continue
        clicks.clear()
        while True:
            view = frame.copy()
            cv2.putText(view, f"{k + 1}/{len(frames)}  frame {f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            for c in clicks:
                cv2.circle(view, c, 6, (0, 255, 255), -1)
            if len(clicks) == 2:
                cv2.line(view, clicks[0], clicks[1], (0, 255, 255), 2)
            cv2.imshow(win, view)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("n") and len(clicks) == 2:
                (x1, y1), (x2, y2) = clicks
                labels[str(f)] = [x1, y1, x2, y2]
                path.write_text(json.dumps(labels, indent=1))
                break
            if key == ord("r"):
                clicks.clear()
            if key == ord("s"):
                break
            if key == ord("q"):
                path.write_text(json.dumps(labels, indent=1))
                cv2.destroyAllWindows()
                print(f"Saved {len(labels)} labels to {path}")
                return

    path.write_text(json.dumps(labels, indent=1))
    cv2.destroyAllWindows()
    print(f"Saved {len(labels)} labels to {path}")


if __name__ == "__main__":
    main()
