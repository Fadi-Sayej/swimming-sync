"""Pictures and demo clips for the Nov 24 meeting -> experiments/limbs/output/meeting_nov24/.

    python experiments/limbs/meeting_nov24.py failures  # where our current models go wrong, by cause
    python experiments/limbs/meeting_nov24.py sheet     # side-view answers: you vs the models, test frames
    python experiments/limbs/meeting_nov24.py mirrors   # how mirrored training copies swap the answers
    python experiments/limbs/meeting_nov24.py demo      # 8 s clips from the 4 test videos, sides drawn
    python experiments/limbs/meeting_nov24.py models    # 10 s clips: each model's output and which step fails

Leg sides in the demo: the front/back model when sure (>= 0.8), else the view model + one-leg
nearer model; arm sides from the ready-made pose model. Label Studio must run for `sheet`.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from facing_eval import BOXES, DS, FACING, hand_legs  # noqa: E402
from view_eval import NEARER, VIEW, nearer_of, sides, view_of  # noqa: E402

from swimsync.dataset.export_yolo import labeled_frames  # noqa: E402
from swimsync.dataset.facing import (FEET_UP, SIDEVIEW_TITLE, _api, mirror_label, sides_from_view,  # noqa: E402
                                     view_choices)
from swimsync.limbs.sides import assign_sides, pose_people  # noqa: E402
from swimsync.waterline.above_water import pick_device  # noqa: E402

OUT = Path("experiments/limbs/output/meeting_nov24")
BLUE, ORANGE, GRAY = (214, 120, 42), (52, 104, 235), (160, 160, 160)  # BGR
GREEN, RED = (60, 180, 60), (50, 50, 220)


def text(img, s, org, scale=0.55, color=(255, 255, 255), thick=1):
    d = max(1, thick)
    for dx, dy in ((-d, 0), (d, 0), (0, -d), (0, d), (-d, -d), (d, d), (-d, d), (d, -d)):
        cv2.putText(img, s, (org[0] + dx, org[1] + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def grid(tiles, cols):
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def short(view, nearer):
    if view in ("front", "back", "can't tell"):
        return {"front": "legs face you", "back": "legs face away", "can't tell": "can't tell"}[view]
    return f"front faces {view}, {nearer or '?'} nearer"


def sheet(dev):
    models = {"view": YOLO(VIEW), "nearer": YOLO(NEARER)}
    yours, _ = labeled_frames(_api(), SIDEVIEW_TITLE)
    rows = []
    for g in ("newpool", "budapest"):
        for p in (DS / f"test_{g}.txt").read_text().split():
            if Path(p).name not in yours:
                continue
            img = cv2.imread(p)
            t = hand_legs(p, img.shape[1], img.shape[0])
            if len(t) < 2:
                continue
            a, b = sorted([t["left"], t["right"]], key=lambda x: x[0] + x[2])
            truth = ("left", "right") if a is t["left"] else ("right", "left")
            yv, yn = view_choices(yours[Path(p).name])
            mv, _ = view_of(models["view"], img, a, b, dev)
            mn = nearer_of(models["nearer"], img, a, b, dev)[0] if mv not in ("front", "back", "can't tell") else None
            ys, ms = sides_from_view(yv, yn, a, b, FEET_UP), sides_from_view(mv, mn, a, b, FEET_UP)
            rows.append(dict(p=p, img=img, a=a, b=b, truth=truth, you=(yv, yn), model=(mv, mn),
                             you_ok=None if ys is None else ys == truth, model_ok=None if ms is None else ms == truth))
    right = [r for r in rows if r["model_ok"] and r["you_ok"] is not None]
    wrong = [r for r in rows if r["model_ok"] is False and r["you_ok"] is not None]
    pick = right[::max(len(right) // 6, 1)][:6] + wrong[::max(len(wrong) // 6, 1)][:6]
    tiles = []
    for r in pick:
        im = r["img"].copy()
        th = max(2, im.shape[1] // 400)
        for box, color in ((r["a"], BLUE), (r["b"], ORANGE)):
            cv2.rectangle(im, tuple(int(v) for v in box[:2]), tuple(int(v) for v in box[2:]), color, th)
        im = cv2.resize(im, (640, 360))
        im = np.vstack([np.full((96, 640, 3), 30, np.uint8), im])
        text(im, "A", (8, 76), 0.55, BLUE)
        text(im, f"(blue) = her {r['truth'][0]},", (24, 76), 0.55, (220, 220, 220))
        text(im, "B", (205, 76), 0.55, ORANGE)
        text(im, f"(orange) = her {r['truth'][1]}  (hand-drawn)", (221, 76), 0.55, (220, 220, 220))
        for i, (who, ans, ok) in enumerate((("You", r["you"], r["you_ok"]), ("Model", r["model"], r["model_ok"]))):
            verdict, col = ("sides RIGHT", GREEN) if ok else (("sides WRONG", RED) if ok is False else ("no sides", GRAY))
            text(im, f"{who}: {short(*ans)}", (8, 24 + 26 * i), 0.55)
            text(im, verdict, (520, 24 + 26 * i), 0.55, col)
        text(im, Path(r["p"]).stem, (8, 448), 0.45, GRAY)
        tiles.append(im)
    out = grid(tiles, 3)
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "3_side_view_answers.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 90])
    n_ok = sum(1 for r in rows if r["model_ok"])
    n_given = sum(1 for r in rows if r["model_ok"] is not None)
    print(f"sheet: first 2 rows model right, last 2 rows wrong; model right on {n_ok} of {n_given} given")


def failures(dev):
    """Where our current models go wrong on the test frames, 4 per cause (box model, direction, nearer leg)."""
    boxer = YOLO(BOXES)
    models = {"view": YOLO(VIEW), "nearer": YOLO(NEARER)}
    yours, _ = labeled_frames(_api(), SIDEVIEW_TITLE)
    causes = {"box": [], "view": [], "nearer": []}
    for g in ("newpool", "budapest"):
        for p in (DS / f"test_{g}.txt").read_text().split():
            img = cv2.imread(p)
            t = hand_legs(p, img.shape[1], img.shape[0])
            if len(t) < 2 or Path(p).name not in yours:
                continue
            r = boxer.predict(img, imgsz=960, conf=0.25, verbose=False, device=dev)[0]
            found = [b for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()) if boxer.names[int(c)] == "leg"]
            a, b = sorted([t["left"], t["right"]], key=lambda x: x[0] + x[2])
            yv, yn = view_choices(yours[Path(p).name])
            base = dict(p=p, img=img, a=a, b=b, you=(yv, yn))
            if len(found) != 2:
                causes["box"].append(dict(base, found=found, line=f"Box model found {len(found)} leg(s), you drew 2"))
                continue
            truth = ("left", "right") if a is t["left"] else ("right", "left")
            mv, _ = view_of(models["view"], img, a, b, dev)
            mn = nearer_of(models["nearer"], img, a, b, dev)[0] if mv not in ("front", "back", "can't tell") else None
            ms = sides_from_view(mv, mn, a, b, FEET_UP)
            if ms is None or ms == truth or sides_from_view(yv, yn, a, b, FEET_UP) != truth:
                continue  # only frames your answers get right and the models get wrong
            if mv != yv:
                causes["view"].append(dict(base, line=f"Direction model: {short(mv, None).split(',')[0]}"))
            elif mn != yn:
                causes["nearer"].append(dict(base, line=f"Nearer-leg model: {mn} nearer"))
    titles = {"box": "1. Box model doesn't find both legs (no sides possible)",
              "view": "2. Direction model wrong (which way she faces)",
              "nearer": "3. Nearer-leg model wrong"}
    rows = []
    for cause, items in causes.items():
        print(cause, len(items))
        tiles = []
        for r in items[::max(len(items) // 4, 1)][:4]:
            im = r["img"].copy()
            th = max(2, im.shape[1] // 400)
            for box, color in ((r["a"], BLUE), (r["b"], ORANGE)):
                cv2.rectangle(im, tuple(int(v) for v in box[:2]), tuple(int(v) for v in box[2:]), color, th)
            for box in r.get("found", []):
                cv2.rectangle(im, tuple(int(v) for v in box[:2]), tuple(int(v) for v in box[2:]), (255, 255, 255), th + 1)
            im = cv2.resize(im, (480, 270))
            im = np.vstack([np.full((62, 480, 3), 30, np.uint8), im])
            text(im, r["line"], (6, 22), 0.5, (120, 140, 255))
            if cause != "box":
                text(im, f"You: {short(*r['you'])}", (6, 48), 0.5)
            else:
                text(im, "white = model's legs, blue/orange = yours", (6, 48), 0.5, (200, 200, 200))
            text(im, Path(r["p"]).stem, (6, 325), 0.4, GRAY)
            tiles.append(im)
        while len(tiles) < 4:
            tiles.append(np.full((332, 480, 3), 30, np.uint8))
        head = np.full((40, 1920, 3), 15, np.uint8)
        text(head, f"{titles[cause]}: {len(items)} of 53 test frames", (8, 28), 0.75, (255, 255, 255), 2)
        rows += [head, np.hstack(tiles)]
    cv2.imwrite(str(OUT / "2_current_models_wrong.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def mirrors():
    stem = "delfin2026_solo1_00041.0s"
    tiles = []
    for flips, label in (("", "original"), ("h", "mirrored left-right"), ("v", "flipped top-bottom"), ("hv", "both")):
        crop = cv2.imread(str(next(Path("data/raw/datasets/m2_view/train").glob(f"*/{stem}{'_' + flips if flips else ''}.jpg"))))
        pair = cv2.imread(str(next(Path("data/raw/datasets/m2_nearer/train").glob(f"*/{stem}{'_' + flips if flips else ''}.jpg"))))
        view = mirror_label("image left", flips)
        near = mirror_label("A", flips)
        col = np.vstack([cv2.resize(crop, (320, 320)), cv2.resize(pair, (320, 320))])
        col = np.vstack([np.full((60, 320, 3), 30, np.uint8), col, np.full((60, 320, 3), 30, np.uint8)])
        text(col, label, (8, 24), 0.6)
        text(col, f"answer: front faces {view}", (8, 50), 0.5, (200, 230, 255))
        text(col, f"answer: {near} nearer", (8, 60 + 640 + 35), 0.5, (200, 230, 255))
        tiles.append(col)
    out = np.hstack(tiles)
    head = np.full((50, out.shape[1], 3), 30, np.uint8)
    text(head, "Each training frame is written 4 times, with the answers mirrored too; built-in flips are off.", (8, 32), 0.6)
    cv2.imwrite(str(OUT / "4_mirror_copies.jpg"), np.vstack([head, out]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("mirrors: written")


def clip_starts(seconds):
    """Start time per test video: 2 s before the window with the most two-leg test frames."""
    starts = {"paris2026_wc_reijnen": None, "paris2025_wc_thean": None, "budapest2024_wc_hoevertsz": None,
              "doha2024_worlds_platanioti": None}
    times = {}
    for p in (DS / "test_newpool.txt").read_text().split() + (DS / "test_budapest.txt").read_text().split():
        video, t = Path(p).stem.rsplit("_", 1)
        img = cv2.imread(p)
        if len(hand_legs(p, img.shape[1], img.shape[0])) == 2:
            times.setdefault(video, []).append(float(t[:-1]))
    for v in starts:
        ts = sorted(times.get(v, []))
        starts[v] = max(0.0, max(ts, key=lambda t0: sum(t0 <= t < t0 + seconds - 2 for t in ts)) - 2) if ts else 30.0
    return starts


def demo(dev, seconds=8.0):
    boxer, pose = YOLO(BOXES), YOLO("yolo11m-pose.pt")
    models = {"view": YOLO(VIEW), "nearer": YOLO(NEARER), "facing": YOLO(FACING)}
    starts = clip_starts(seconds)
    for v, t0 in starts.items():
        cap = cv2.VideoCapture(f"data/raw/test/{v}.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
        writer, n = None, int(seconds * fps)
        for _ in range(n):
            ok, img = cap.read()
            if not ok:
                break
            frame = draw_frame(img, boxer, pose, models, dev)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(OUT / f"5_demo_{v.split('_')[0]}.mp4"), cv2.VideoWriter_fourcc(*"avc1"),
                                         fps, (w, h))
            writer.write(frame)
        writer.release()
        print(f"demo {v}: {t0:.1f}-{t0 + seconds:.1f} s", flush=True)


def models_video(dev, seconds=10.0):
    """Each model's own output on every frame, and which step fails (the main problems)."""
    from facing_eval import front_or_back

    from swimsync.dataset.facing import sides_from_facing

    boxer, pose = YOLO(BOXES), YOLO("yolo11m-pose.pt")
    fb_model, view_model, near_model = YOLO(FACING), YOLO(VIEW), YOLO(NEARER)
    combined, size = None, None
    for v, t0 in clip_starts(seconds).items():
        cap = cv2.VideoCapture(f"data/raw/test/{v}.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        cap.set(cv2.CAP_PROP_POS_MSEC, t0 * 1000)
        w = None
        for _ in range(int(seconds * fps)):
            ok, img = cap.read()
            if not ok:
                break
            r = boxer.predict(img, imgsz=960, conf=0.25, verbose=False, device=dev)[0]
            names = [boxer.names[int(c)] for c in r.boxes.cls.tolist()]
            boxes = r.boxes.xyxy.tolist()
            polys = r.masks.xy if r.masks is not None else [None] * len(boxes)
            legs = [b for k, b in zip(names, boxes) if k == "leg"]
            arms = [b for k, b in zip(names, boxes) if k == "arm"]
            lines, side = [], {}
            ok_legs = len(legs) == 2
            lines.append((f"1. Outline model: {len(legs)} leg(s), {len(arms)} arm(s)" +
                          ("" if ok_legs else "  (left/right needs both legs)"), None if ok_legs else (0, 200, 255)))
            if ok_legs:
                a, b = sorted(legs, key=lambda x: x[0] + x[2])
                fb, pfb = front_or_back(fb_model, img, [a, b], dev)
                sure = fb in ("front", "back") and pfb >= 0.8
                lines.append((f"2. Front/back model: {fb} ({pfb:.2f})" + ("  sure" if sure else "  not sure"), None))
                view, pv = view_of(view_model, img, a, b, dev)
                lines.append((f"3. Direction model: {view} ({pv:.2f})", None))
                result, how = None, ""
                if sure:
                    result, how = sides_from_facing(fb, FEET_UP, a, b), "front/back model + rule"
                elif view == "can't tell":
                    how = "direction model can't tell"
                elif view in ("front", "back"):
                    result, how = sides_from_view(view, None, a, b), "direction model + rule"
                else:
                    near, pn = nearer_of(near_model, img, a, b, dev)
                    lines.append((f"4. Nearer-leg model: {near} ({pn:.2f})" + ("  <- weak model" if pn < 0.7 else ""),
                                  RED if pn < 0.7 else None))
                    result, how = sides_from_view(view, near, a, b), "direction + nearer-leg models + rule"
                if not result and how.endswith("rule"):
                    how = "rule can't tell (legs crossed, or front up/down)"
                if result:
                    side[tuple(a)], side[tuple(b)] = result
                    lines.append((f"=> leg sides from: {how}", GREEN))
                else:
                    lines.append((f"=> no leg sides: {how or 'legs crossed'}", RED))
            else:
                lines.append(("=> no leg sides on this frame", (0, 200, 255)))
            if arms:
                arm_sides, _ = assign_sides([("arm", b) for b in arms], pose_people(pose, img, device=dev))
                side.update({tuple(b): s for b, s in zip(arms, arm_sides) if s})
                n_side = sum(1 for s in arm_sides if s)
                lines.append((f"Arms: ready-made pose model + rule: {n_side} of {len(arms)} given a side", None))
            out = img.copy()
            th = max(2, img.shape[1] // 500)
            overlay = out.copy()
            for k, bx, poly in zip(names, boxes, polys):
                color = {"left": BLUE, "right": ORANGE}.get(side.get(tuple(bx)), GRAY)
                if poly is not None and len(poly) > 2:
                    cv2.fillPoly(overlay, [poly.astype(np.int32)], color)
                p1, p2 = tuple(int(x) for x in bx[:2]), tuple(int(x) for x in bx[2:])
                cv2.rectangle(out, p1, p2, color, th)
                text(out, f"{side.get(tuple(bx), '?')} {k}", (p1[0] + 4, max(p1[1] - 8, 20)), 0.5 * th, color, th)
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
            sc = img.shape[1] / 1920
            panel_h = int((50 * len(lines) + 90) * sc)
            cv2.rectangle(out, (0, 0), (int(1450 * sc), panel_h), (20, 20, 20), -1)
            text(out, f"{v.split('_')[0]}  (test video, never seen in training)", (int(16 * sc), int(48 * sc)),
                 1.15 * sc, (200, 200, 200), 2)
            for i, (ln, col) in enumerate(lines):
                text(out, ln, (int(16 * sc), int((100 + 50 * i) * sc)), 1.1 * sc, col or (255, 255, 255), 2)
            text(out, "blue = her left, orange = her right, gray = no side", (12, img.shape[0] - 16), 0.45 * th,
                 (255, 255, 255), th)
            if w is None:
                w = cv2.VideoWriter(str(OUT / f"6_models_{v.split('_')[0]}.mp4"), cv2.VideoWriter_fourcc(*"avc1"),
                                    fps, (out.shape[1], out.shape[0]))
            if combined is None:
                size = (out.shape[1], out.shape[0])
                combined = cv2.VideoWriter(str(OUT / "6_models_all.mp4"), cv2.VideoWriter_fourcc(*"avc1"), 25, size)
            w.write(out)
            combined.write(cv2.resize(out, size))
        w.release()
        print(f"models video {v}: {t0:.1f}-{t0 + seconds:.1f} s", flush=True)
    combined.release()


def draw_frame(img, boxer, pose, models, dev):
    r = boxer.predict(img, imgsz=960, conf=0.25, verbose=False, device=dev)[0]
    found = [(boxer.names[int(c)], b) for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())]
    legs = [b for k, b in found if k == "leg"]
    arms = [b for k, b in found if k == "arm"]
    side, how = {}, "legs: not 2 legs found"
    if len(legs) == 2:
        a, b = sorted(legs, key=lambda x: x[0] + x[2])
        s, step = sides(img, a, b, models, dev, 0.0, True)
        if s:
            side[tuple(a)], side[tuple(b)] = s
            how = f"legs: {step}"
        else:
            how = "legs: model can't tell"
    if arms:
        arm_sides, _ = assign_sides([("arm", b) for b in arms], pose_people(pose, img, device=dev))
        side.update({tuple(b): s for b, s in zip(arms, arm_sides) if s})
    out = img.copy()
    th = max(2, img.shape[1] // 500)
    for kind, boxes in (("leg", legs), ("arm", arms)):
        for b in boxes:
            s = side.get(tuple(b))
            color = {"left": BLUE, "right": ORANGE}.get(s, GRAY)
            p1, p2 = tuple(int(v) for v in b[:2]), tuple(int(v) for v in b[2:])
            cv2.rectangle(out, p1, p2, color, th)
            text(out, f"{s or '?'} {kind}", (p1[0] + 4, max(p1[1] - 8, 20)), 0.5 * th, color, th)
    text(out, how + " | arms: pose model | blue = her left, orange = her right", (12, img.shape[0] - 16),
         0.45 * th, (255, 255, 255), th)
    return out


def main():
    what = sys.argv[1:] or ["sheet", "mirrors", "demo"]
    dev = pick_device()
    OUT.mkdir(parents=True, exist_ok=True)
    if "failures" in what:
        failures(dev)
    if "sheet" in what:
        sheet(dev)
    if "mirrors" in what:
        mirrors()
    if "models" in what:
        models_video(dev)
    if "demo" in what:
        demo(dev)


if __name__ == "__main__":
    main()
