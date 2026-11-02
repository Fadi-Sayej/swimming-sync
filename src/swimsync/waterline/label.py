"""Hand-label the true waterline on sampled frames (ground truth for evaluation).

Usage:
    python -m swimsync.waterline.label data/raw/solo.mp4 --n 50

For each frame: click 2 points on the waterline (left side, then right side).
Mouse: left-click = add point   right-click = next frame (save); right-click with no points = skip
Keys:  n / Enter / Space = next (save)   r / Backspace = redo   s = skip   q / Esc = quit and save
Keys also work with Caps Lock on. If keys do nothing (non-English keyboard layout or
the window has no keyboard focus), use right-click. Every label is saved right away.

Saves data/labels/<video_name>_waterline.json  as  {frame: [x1, y1, x2, y2]}
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


KEYS = {
    ord("n"): "next", 13: "next", 32: "next",
    ord("r"): "redo", 8: "redo", 127: "redo",
    ord("s"): "skip",
    ord("q"): "quit", 27: "quit",
}


def key_action(key):
    """Map a cv2.waitKey code to next/redo/skip/quit (None for anything else)."""
    if 65 <= key <= 90:  # Caps Lock / Shift: treat A-Z like a-z
        key += 32
    return KEYS.get(key)


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
    pending = []  # actions from the mouse (right-click = next, or skip if no points)

    def on_click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and len(clicks) != 1:
            pending.append("next" if clicks else "skip")

    win = "waterline: click 2 points | right-click or n = next | r = redo | s = skip | q = quit"
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
        pending.clear()
        while True:
            view = frame.copy()
            cv2.putText(view, f"{k + 1}/{len(frames)}  frame {f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            for c in clicks:
                cv2.circle(view, c, 6, (0, 255, 255), -1)
            if len(clicks) == 2:
                cv2.line(view, clicks[0], clicks[1], (0, 255, 255), 2)
            cv2.imshow(win, view)
            key = cv2.waitKey(30)
            action = key_action(key)
            if action is None and key not in (-1, 255):
                print(f"key code {key} not recognized (use right-click for next)")
            if pending:
                action = pending.pop()
            if action == "next" and len(clicks) == 2:
                (x1, y1), (x2, y2) = clicks
                labels[str(f)] = [x1, y1, x2, y2]
                path.write_text(json.dumps(labels, indent=1))
                print(f"saved frame {f} ({len(labels)} labels)")
                break
            if action == "next":
                print("click 2 points first")
            if action == "redo":
                clicks.clear()
            if action == "skip":
                break
            if action == "quit":
                path.write_text(json.dumps(labels, indent=1))
                cv2.destroyAllWindows()
                print(f"Saved {len(labels)} labels to {path}")
                return

    path.write_text(json.dumps(labels, indent=1))
    cv2.destroyAllWindows()
    print(f"Saved {len(labels)} labels to {path}")


if __name__ == "__main__":
    main()
