"""Leg sides from the view + nearer-leg models (#25), scored on the test frames.

Two models trained on the side-view answers (swimsync.dataset.facing export-view):
- view: which way the front of her legs faces (front, back, image left/right/up/down, unsure);
- nearer: which of the two leg boxes (A = the one on the image's left) is nearer the camera.
sides_from_view turns them into her left and right, assuming her feet point up.

Scored three ways on the 4-class test frames with both legs:
1. each model against your side-view answers on the same frames ("M2 side view - test");
2. her left/right from view + nearer alone;
3. the front/back model when it is sure (>= 0.8), else view + nearer.
Inputs as in facing_eval.py: hand boxes, and the outline model's boxes matched to them.

Usage: python experiments/limbs/view_eval.py [--min-conf 0.0]   (Label Studio running)
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from facing_eval import BOXES, DS, FACING, front_or_back, hand_legs, model_pair  # noqa: E402

from swimsync.dataset.export_yolo import labeled_frames  # noqa: E402
from swimsync.dataset.facing import (CLASS_VIEW, FEET_UP, SIDEVIEW_TITLE, _api, leg_focus, pair_crop,  # noqa: E402
                                     sides_from_facing, sides_from_view, square_crop, view_choices)
from swimsync.waterline.above_water import pick_device  # noqa: E402

VIEW = "experiments/limbs/output/view/weights/best.pt"
NEARER = "experiments/limbs/output/nearleg_noaug/weights/best.pt"  # or .../nearer/ (both legs, A/B)


def classify(model, crop, dev):
    r = model.predict(crop, imgsz=384, verbose=False, device=dev)[0]
    return model.names[int(r.probs.top1)], float(r.probs.top1conf)


def nearer_of(model, img, a, b, dev):
    """('A' or 'B', confidence). A one-leg model (near/far) scores each leg; the nearer has the higher score."""
    if "near" not in model.names.values():
        return classify(model, pair_crop(img, a, b), dev)
    near = [k for k, v in model.names.items() if v == "near"][0]
    pa, pb = (float(model.predict(leg_focus(img, x, y), imgsz=384, verbose=False, device=dev)[0].probs.data[near])
              for x, y in ((a, b), (b, a)))
    return ("A", pa / (pa + pb)) if pa >= pb else ("B", pb / (pa + pb))


def view_of(model, img, a, b, dev):
    x1, y1, x2, y2 = square_crop([a, b], img.shape[1], img.shape[0])
    c, p = classify(model, img[y1:y2, x1:x2], dev)
    return CLASS_VIEW[c], p


def sides(img, a, b, models, dev, min_conf, use_facing):
    """(sides for (A, B) or None, which step decided)."""
    if use_facing:
        fb, p = front_or_back(models["facing"], img, [a, b], dev)
        if fb in ("front", "back") and p >= 0.8:
            return sides_from_facing(fb, FEET_UP, a, b), "front/back model"
    view, p = view_of(models["view"], img, a, b, dev)
    if view == "can't tell" or p < min_conf:
        return None, None
    nearer = None
    if view not in ("front", "back"):
        nearer, q = nearer_of(models["nearer"], img, a, b, dev)
        if q < min_conf:
            return None, None
    return sides_from_view(view, nearer, a, b), "view + nearer"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-conf", type=float, default=0.0)
    p.add_argument("--view", default=VIEW)
    p.add_argument("--nearer", default=NEARER)
    a = p.parse_args()
    dev = pick_device()
    models = {"view": YOLO(a.view), "nearer": YOLO(a.nearer), "facing": YOLO(FACING)}
    boxer = YOLO(BOXES)
    yours, _ = labeled_frames(_api(), SIDEVIEW_TITLE)

    agree = Counter()
    groups = {"newpool": (DS / "test_newpool.txt").read_text().split(),
              "budapest": (DS / "test_budapest.txt").read_text().split()}
    for group, images in groups.items():
        n = Counter()
        for img_path in images:
            img = cv2.imread(img_path)
            truth = hand_legs(img_path, img.shape[1], img.shape[0])
            if len(truth) < 2:
                continue
            n["frames"] += 1
            # 1. models vs your answers (hand boxes)
            hand_a, hand_b = sorted([truth["left"], truth["right"]], key=lambda bx: bx[0] + bx[2])
            name = Path(img_path).name
            if name in yours:
                my_view, my_nearer = view_choices(yours[name])
                view, _ = view_of(models["view"], img, hand_a, hand_b, dev)
                agree["view", "asked"] += 1
                agree["view", "same"] += view == my_view
                agree["view", my_view, view] += 1
                if my_nearer in ("A", "B") and my_view not in ("front", "back", "can't tell"):
                    nearer, _ = nearer_of(models["nearer"], img, hand_a, hand_b, dev)
                    agree["nearer", "asked"] += 1
                    agree["nearer", "same"] += nearer == my_nearer
            # 2./3. her left and right
            inputs = {"hand boxes": (truth["left"], truth["right"]), "model boxes": model_pair(boxer, img, truth, dev)}
            for inp, legs in inputs.items():
                if legs is None:
                    continue
                n[inp, "found"] += 1
                left = legs[0]
                a_box, b_box = sorted(legs, key=lambda bx: bx[0] + bx[2])
                right_answer = ("left", "right") if a_box is left else ("right", "left")
                for method, use_facing in (("view + nearer", False), ("front/back, then view + nearer", True)):
                    s, step = sides(img, a_box, b_box, models, dev, a.min_conf, use_facing)
                    if s is None:
                        continue
                    n[inp, method, "given"] += 1
                    n[inp, method, "right"] += s == right_answer
                    n[inp, method, step, "given"] += 1
                    n[inp, method, step, "right"] += s == right_answer
        print(f"\n{group}: {n['frames']} frames with both legs")
        for inp in ("hand boxes", "model boxes"):
            print(f"  {inp} (both found on {n[inp, 'found']}):")
            for method in ("view + nearer", "front/back, then view + nearer"):
                g, r = n[inp, method, "given"], n[inp, method, "right"]
                steps = "; ".join(f"{st} {n[inp, method, st, 'right']}/{n[inp, method, st, 'given']}"
                                  for st in ("front/back model", "view + nearer") if n[inp, method, st, "given"])
                print(f"    {method:31s}: sides given {g} of {n[inp, 'found']}, right {r} "
                      f"({100 * r / max(g, 1):.0f}%)   [{steps}]")
    print(f"\nview model = your answer on {agree['view', 'same']} of {agree['view', 'asked']}; "
          f"nearer model = your answer on {agree['nearer', 'same']} of {agree['nearer', 'asked']}")
    views = ["front", "back", "image left", "image right", "image up", "image down", "can't tell"]
    print("your view (rows) vs model (columns):", " | ".join(v[:7] for v in views))
    for mv in views:
        row = [agree["view", mv, v] for v in views]
        if sum(row):
            print(f"  {mv:11s}", "  ".join(f"{x:7d}" for x in row))


if __name__ == "__main__":
    main()
