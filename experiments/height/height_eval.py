"""Leg height checked against the hand labels of "M3 leg height - test" (Task 5).

A labeled leg (toes clicked) matches the predicted leg whose toes point is nearest, if closer than
25% of the labeled toes -> water length; otherwise the outline model didn't find it. Hand truth:
status from the label, and when "knee out" the score from the clicks with the same formula.

Usage: python experiments/height/height_eval.py [--knees pose|ours] [--weights PATH]   (Label Studio running)
"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from swimsync.dataset.export_yolo import labeled_frames, project_titles
from swimsync.dataset.facing import _api
from swimsync.dataset.height_labels import TITLE, parse_height
from swimsync.height.measure import SEG, PoseKnees, measure_frame
from swimsync.dataset.scene import PoolShotClassifier
from swimsync.height.score import height_score, with_unseen_legs

MIN_POOL = 0.7  # CLIP "pool shot" probability; below it the frame is skipped (deck, backstage, logo)
from swimsync.waterline.above_water import pick_device

FOLDERS = ("test", "phone")


def group(name):
    if name.startswith("budapest"):
        return "Budapest 2024"
    if name.startswith(("doha", "paris")):
        return "new pools"
    return "phone videos"


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def knee_finder(kind, dev, weights=None):
    if kind == "ours":
        from swimsync.height.measure import OurKnees
        return OurKnees(weights, device=dev) if weights else OurKnees(device=dev)
    return PoseKnees(device=dev)


def image_path(name):
    for f in FOLDERS:
        p = Path("data/raw/frames/m2/images") / f / name
        if p.exists():
            return p
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--knees", choices=("pose", "ours"), default="pose")
    p.add_argument("--weights", help="our knee model's weights (default: OurKnees' own)")
    a = p.parse_args()
    dev = pick_device()
    api = _api()
    assert TITLE in project_titles(api), f"no project '{TITLE}'"
    lab, n_tasks = labeled_frames(api, TITLE)
    seg, knees, pool = YOLO(SEG), knee_finder(a.knees, dev, a.weights), PoolShotClassifier()
    n = defaultdict(Counter)
    diffs, point_err = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for name, result in sorted(lab.items()):
        path = image_path(name)
        if path is None:
            continue
        g = group(name)
        hand = parse_height(result)
        n[g]["frames"] += 1
        n[g]["pressed together"] += hand[2]["status"] == "pressed to leg 1"
        img = cv2.imread(str(path))
        if pool.score(img) < MIN_POOL:
            n[g]["not a pool shot (skipped)"] += 1
            continue
        pred = [leg for leg in measure_frame(img, seg, knees, dev) if leg.toes is not None]
        # per frame: her legs judged by hand (a leg not out of the water is a FAIL too, Nov 25)
        statuses = [l["status"] for l in hand.values() if l["status"] in ("knee out", "knee under water", "not out of the water")]
        model = with_unseen_legs(pred, expected=len(statuses))
        hand_pass = sum(st == "knee out" for st in statuses)
        model_pass = sum(q.status == "pass" for q in model)
        n[g]["frames judged"] += 1
        agree = hand_pass == model_pass and len(statuses) - hand_pass == sum(q.status == "fail" for q in model)
        n[g]["pass/fail counts agree"] += agree
        if not agree:
            n[g]["model passes more legs" if model_pass > hand_pass else
                 "model passes fewer legs" if model_pass < hand_pass else "other"] += 1
        for leg in hand.values():
            if leg["status"] not in ("knee out", "knee under water") or leg["water"] is None:
                continue
            # match on the toes, or on the water point when the toes are off the picture
            part = "toes" if leg["toes"] is not None else "water"
            length = (dist(leg["toes"], leg["water"]) if leg["toes"] is not None
                      else 2 * dist(leg["knee"], leg["water"]) if leg["knee"] is not None else None)
            if not length:
                continue
            n[g]["legs out"] += 1
            best = min(pred, key=lambda q: dist(getattr(q, part), leg[part]), default=None)
            if best is None or dist(getattr(best, part), leg[part]) > 0.25 * length:
                n[g]["leg not found"] += 1
                continue
            n[g]["leg found"] += 1
            hand_out = leg["status"] == "knee out"
            if best.knee is None:
                n[g]["no knee found"] += 1
                n[g]["no knee found, knee out by hand"] += hand_out
            model_out = best.status == "pass"
            n[g]["judged"] += 1
            n[g]["out/under agree"] += model_out == hand_out
            n[g]["false fail"] += hand_out and not model_out
            n[g]["false pass"] += model_out and not hand_out
            for part in ("toes", "water", "knee"):
                if leg[part] is not None and getattr(best, part) is not None:
                    point_err[g][part].append(dist(leg[part], getattr(best, part)) / length)
            if hand_out and model_out and leg["knee"] is not None and leg["toes"] is not None and best.score is not None:
                hs = height_score(leg["toes"], leg["knee"], leg["water"])
                if hs is not None:
                    diffs[g].append(abs(best.score - hs) * 100)
    print(f"'{TITLE}': {len(lab)} of {n_tasks} frames labeled; knees from: {a.knees}")
    for g, c in n.items():
        print(f"\n{g}: {c['frames']} frames ({c['not a pool shot (skipped)']} not a pool shot, skipped), "
              f"{c['legs out']} legs out of the water "
              f"(pressed together on {c['pressed together']} frames)")
        print(f"  per frame, number of legs passing / failing agrees on {c['pass/fail counts agree']} of "
              f"{c['frames judged']} frames ({100 * c['pass/fail counts agree'] / max(c['frames judged'], 1):.0f}%); "
              f"the others: model passes more legs {c['model passes more legs']}, fewer "
              f"{c['model passes fewer legs']}, same passes but a leg missing or extra {c['other']}")
        print(f"  outline model found {c['leg found']} of {c['legs out']} legs")
        print(f"  no knee found on {c['no knee found']} (knee out by hand on {c['no knee found, knee out by hand']})")
        print(f"  knee out / under agrees on {c['out/under agree']} of {c['judged']} "
              f"({100 * c['out/under agree'] / max(c['judged'], 1):.0f}%); false FAIL {c['false fail']}, "
              f"false pass {c['false pass']}")
        if diffs[g]:
            print(f"  score difference when both say knee out: mean {np.mean(diffs[g]):.1f} points, "
                  f"median {np.median(diffs[g]):.1f} ({len(diffs[g])} legs)")
        for part, e in point_err[g].items():
            print(f"  {part}: within 10% of the leg {np.mean(np.array(e) <= 0.1) * 100:.0f}%, "
                  f"within 20% {np.mean(np.array(e) <= 0.2) * 100:.0f}% ({len(e)} legs)")


if __name__ == "__main__":
    main()
