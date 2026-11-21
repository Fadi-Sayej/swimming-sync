"""Leg sides via front/back: do her legs face the camera or away from it? (#25)

"Is this her left or right leg?" flips when the image is mirrored, and ready-made image models
barely tell an image from its mirror, so they can't learn it. Front/back doesn't flip in a mirror,
and the sides follow by geometry: front/back + the direction toward her feet (knee -> ankle) +
where each leg box is.

Labeling: one click per frame (front / back / can't tell), with the two leg boxes found by the
outline model drawn as a guide.

    python -m swimsync.dataset.facing          # Label Studio "M2 front/back - train" and "- test"
    python -m swimsync.dataset.facing export   # -> data/raw/datasets/m2_facing (Label Studio running)
"""

import json
import sys
from pathlib import Path

import numpy as np

LEG = {"left": (11, 13, 15), "right": (12, 14, 16)}  # COCO hip, knee, ankle
FEET_UP = np.array([0.0, -1.0])  # legs out of the water (image y grows downwards)


def feet_dir(kp):
    """Unit vector toward her feet: per leg, first -> last visible of hip/knee/ankle, averaged.

    Doesn't depend on which leg is which, so the pose model's points work even when it mixes up
    her sides. None if no leg has two visible points or the legs point opposite ways.
    """
    vs = []
    for idx in LEG.values():
        pts = [np.asarray(kp[i][:2], float) for i in idx if kp[i][2] > 0]
        if len(pts) >= 2:
            v = pts[-1] - pts[0]
            if np.linalg.norm(v) > 0:
                vs.append(v / np.linalg.norm(v))
    if not vs:
        return None
    v = np.mean(vs, axis=0)
    return v / np.linalg.norm(v) if np.linalg.norm(v) > 0.3 else None


def feet_from_people(people, legs, conf_min=0.3):
    """feet_dir of the person with the most sure hip/knee/ankle points inside the leg boxes."""
    best, best_n = None, 0
    for p in people:
        p = np.array(p, float)
        p[:, 2] = p[:, 2] >= conf_min
        n = sum(any(b[0] <= p[i, 0] <= b[2] and b[1] <= p[i, 1] <= b[3] for b in legs)
                for idx in LEG.values() for i in idx if p[i, 2])
        if n > best_n:
            best, best_n = p, n
    return None if best is None else feet_dir(best)


def facing(left_c, right_c, feet, min_sin=0.3):
    """'front' (legs face the camera) or 'back', from where her left and right legs are.

    left_c, right_c: centres of her left / right leg; feet: direction toward her feet.
    None when the legs line up along her body (seen from the side), where it can't be told.
    """
    d = np.asarray(left_c, float) - np.asarray(right_c, float)
    if feet is None or np.linalg.norm(d) == 0:
        return None
    cross = feet[0] * d[1] - feet[1] * d[0]
    if abs(cross) < min_sin * np.linalg.norm(d) * np.linalg.norm(feet):
        return None
    return "front" if cross < 0 else "back"


def _centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def sides_from_facing(front_or_back, feet, box_a, box_b):
    """Her sides for two leg boxes: ("left", "right") if box_a is her left leg, else reversed."""
    if facing(_centre(box_a), _centre(box_b), feet, min_sin=0) == front_or_back:
        return ("left", "right")
    return ("right", "left")


# direction the front of her legs faces, x right / y up / z toward the camera
VIEW_DIRS = {"front": (0, 0, 1), "back": (0, 0, -1), "image left": (-1, 0, 0), "image right": (1, 0, 0),
             "image up": (0, 1, 0), "image down": (0, -1, 0)}


def sides_from_view(view, nearer, box_a, box_b, feet=FEET_UP):
    """Her sides for two leg boxes from the view, or None if the view gives no answer.

    view: which way the front of her legs (kneecaps, tops of the feet) faces, a VIEW_DIRS key.
    nearer: "A" or "B", the leg nearer the camera (needed when she is side-on).
    feet: direction toward her feet in the image (x right, y down); default feet up.

    Her left = head direction x front direction (with the head opposite the feet). If that
    points at or away from the camera, the nearer leg decides; if it lies in the image, the
    leg further along it is her left.
    """
    if view not in VIEW_DIRS or feet is None:
        return None
    head = -np.array([feet[0], -feet[1], 0.0])
    left = np.cross(head, np.array(VIEW_DIRS[view], float))
    if np.linalg.norm(left) < 0.3:  # front along her body: not a real pose
        return None
    if abs(left[2]) >= 0.7 * np.linalg.norm(left):
        if nearer not in ("A", "B"):
            return None
        a_is_left = (left[2] > 0) == (nearer == "A")
    else:
        d = np.array(_centre(box_a)) - np.array(_centre(box_b))
        a_is_left = d[0] * left[0] - d[1] * left[1] > 0
    return ("left", "right") if a_is_left else ("right", "left")


def square_crop(boxes, img_w, img_h, pad=0.2):
    """Square around all boxes (side = longer side * (1 + 2 * pad)), moved inside the image.

    Square because the classifier centre-crops a square: a tall crop of a vertical would lose
    her feet. Shrinks only when the image is too small for it.
    """
    b = np.asarray(boxes, float)
    x1, y1 = b[:, :2].min(axis=0)
    x2, y2 = b[:, 2:].max(axis=0)
    side = min(round(max(x2 - x1, y2 - y1) * (1 + 2 * pad)), img_w, img_h)
    out = []
    for lo, hi, size in ((x1, x2, img_w), (y1, y2, img_h)):
        start = min(max(round((lo + hi) / 2 - side / 2), 0), size - side)
        out += [start, start + side]
    return out[0], out[2], out[1], out[3]


def facing_choice(result):
    """'front', 'back' or 'unsure' (can't tell) from a front/back label; None if no answer."""
    for r in result:
        if r.get("type") == "choices":
            choice = r["value"]["choices"][0]
            return choice if choice in ("front", "back") else "unsure"
    return None


def write_facing_dataset(out, splits, max_size=512, pad=0.2):
    """splits: {split: [(image path, leg boxes, 'front'/'back')]} -> out/<split>/<answer>/<name>.jpg."""
    import shutil

    import cv2

    if out.exists():
        shutil.rmtree(out)
    for split, items in splits.items():
        for path, legs, answer in items:
            img = cv2.imread(str(path))
            x1, y1, x2, y2 = square_crop(legs, img.shape[1], img.shape[0], pad)
            crop = img[y1:y2, x1:x2]
            if crop.shape[0] > max_size:
                crop = cv2.resize(crop, (max_size, max_size), interpolation=cv2.INTER_AREA)
            (out / split / answer).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / split / answer / Path(path).name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])


def facing_config():
    return """<View>
  <Header value="Do her legs face you (front: kneecaps, tops of the feet, toes toward you) or away (back: backs of the knees, heels, soles, buttocks)? Side view or unsure: can't tell."/>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <Choices name="facing" toName="image" choice="single-radio" showInline="true" required="true">
    <Choice value="front" hotkey="1"/>
    <Choice value="back" hotkey="2"/>
    <Choice value="can't tell" hotkey="3"/>
  </Choices>
  <RectangleLabels name="legs" toName="image" strokeWidth="1" opacity="0.15">
    <Label value="leg" background="#2a78d6"/>
  </RectangleLabels>
</View>"""


def facing_task(image_url, leg_boxes, img_w, img_h):
    """Task with the two leg boxes (pixels) shown as a guide."""
    result = [{"from_name": "legs", "to_name": "image", "type": "rectanglelabels",
               "original_width": img_w, "original_height": img_h,
               "value": {"x": 100 * x1 / img_w, "y": 100 * y1 / img_h, "width": 100 * (x2 - x1) / img_w,
                         "height": 100 * (y2 - y1) / img_h, "rotation": 0, "rectanglelabels": ["leg"]}}
              for x1, y1, x2, y2 in leg_boxes]
    return {"data": {"image": image_url}, "predictions": [{"model_version": "seg_b3 legs", "result": result}]}


def view_config():
    return """<View>
  <Header value="1) Front of her legs (kneecaps, tops of the feet) faces: you, away, or the image's left / right / top / bottom. 2) If left / right / top / bottom: which leg is nearer you, A or B?"/>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <Choices name="view" toName="image" choice="single-radio" showInline="true" required="true">
    <Choice value="front" hotkey="1"/>
    <Choice value="back" hotkey="2"/>
    <Choice value="image left" hotkey="3"/>
    <Choice value="image right" hotkey="4"/>
    <Choice value="can't tell" hotkey="5"/>
    <Choice value="image up" hotkey="6"/>
    <Choice value="image down" hotkey="7"/>
  </Choices>
  <Choices name="nearer" toName="image" choice="single-radio" showInline="true">
    <Choice value="A" hotkey="a"/>
    <Choice value="B" hotkey="b"/>
    <Choice value="can't tell" hotkey="c"/>
  </Choices>
  <RectangleLabels name="legs" toName="image" strokeWidth="2" opacity="0.1">
    <Label value="A" background="#2a78d6"/>
    <Label value="B" background="#eb6834"/>
  </RectangleLabels>
</View>"""


def view_task(image_url, leg_boxes, img_w, img_h):
    """Task with the two leg boxes (pixels) drawn as A (the one on the image's left) and B."""
    a, b = sorted(leg_boxes, key=lambda bx: bx[0] + bx[2])
    task = facing_task(image_url, [a, b], img_w, img_h)
    for r, name in zip(task["predictions"][0]["result"], ("A", "B")):
        r["value"]["rectanglelabels"] = [name]
    task["predictions"][0]["model_version"] = "hand leg boxes"
    return task


def view_choices(result):
    """(view, nearer) from a side-view label; None for an unanswered question."""
    got = {r["from_name"]: r["value"]["choices"][0] for r in result if r.get("type") == "choices"}
    return got.get("view"), got.get("nearer")


def pick_spread(times, n, min_gap=1.0):
    """Up to n times, at least min_gap apart, spread over the whole list."""
    kept = []
    for t in sorted(times):
        if not kept or t - kept[-1] >= min_gap:
            kept.append(t)
    if len(kept) <= n:
        return kept
    return [kept[i] for i in np.linspace(0, len(kept) - 1, n).round().astype(int)]


def find_two_legs(videos, boxer, device, conf=0.4):
    """{video: [(t, [box, box])]} at the scanned in-pool times where exactly two legs are found."""
    import cv2
    from tqdm import tqdm

    out = {}
    for video, times in videos.items():
        cap, found = cv2.VideoCapture(str(video)), []
        for t in tqdm(times, desc=Path(video).name):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            r = boxer.predict(frame, imgsz=960, conf=conf, verbose=False, device=device)[0]
            legs = [b for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()) if boxer.names[int(c)] == "leg"]
            if len(legs) == 2:
                found.append((t, legs))
        out[str(video)] = found
    return out


def _api():
    from swimsync.dataset.labelstudio import LabelStudioAPI

    cred = dict(line.split(": ", 1) for line in
                Path("data/raw/labelstudio/credentials.txt").read_text().splitlines() if ": " in line)
    return LabelStudioAPI(cred["url"], cred["api_token"])


def export():
    """data/raw/datasets/m2_facing/{train,val,test}/{front,back,unsure}/ from the labeled frames.

    "unsure" (can't tell: mostly side views) is kept as a class so the model can say it too.
    """
    from collections import Counter

    from swimsync.dataset.export_yolo import labeled_frames, project_titles, split_train_val

    api, root = _api(), Path("data/raw/frames/m2")
    titles, data = project_titles(api), {}
    for split in ("train", "test"):
        title = f"M2 front/back - {split}"
        if title not in titles:
            continue
        legs = {r["file"]: r["legs"] for r in json.loads((root / f"facing_{split}.json").read_text())}
        lab, n = labeled_frames(api, title)
        answers = {name: facing_choice(r) for name, r in lab.items()}
        data[split] = [(root / "images" / f"facing_{split}" / name, legs[name], a)
                       for name, a in sorted(answers.items()) if a]
        print(f"{title}: {len(lab)} labeled of {n}: {dict(Counter(a for *_, a in data[split]))}")
    by_name = {Path(item[0]).name: item for item in data["train"]}
    train, val = split_train_val(list(by_name))
    splits = {"train": [by_name[n] for n in train], "val": [by_name[n] for n in val]}
    if data.get("test"):
        splits["test"] = data["test"]
    out = Path("data/raw/datasets/m2_facing")
    write_facing_dataset(out, splits)
    for s, items in splits.items():
        print(f"{s}: {dict(Counter(a for _, _, a in items))}")
    print(f"Saved to {out}")


SIDEVIEW_TITLE = "M2 side view - test"
SIDEVIEW_TRAIN_TITLE = "M2 side view - train"


def _hand_test_legs():
    """[(image path, her left leg box, her right leg box, w, h)]: 4-class test frames with both legs."""
    import cv2

    ds, out = Path("data/raw/datasets/m2_4cls"), []
    for img in sorted((ds / "images/test").glob("*.jpg")):
        h, w = cv2.imread(str(img)).shape[:2]
        legs = {}
        for line in (ds / "labels/test" / (img.stem + ".txt")).read_text().splitlines():
            c, cx, cy, bw, bh = map(float, line.split())
            if int(c) in (0, 1):
                legs[int(c)] = ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)
        if len(legs) == 2:
            out.append((img, legs[0], legs[1], w, h))
    return out


def _hand_feet(name):
    """Direction toward her feet from the hand-labeled hip/knee/ankle points (keypoint test set)."""
    import cv2

    ds = Path("data/raw/datasets/m2_pose")
    lab, img = ds / "labels/test" / (Path(name).stem + ".txt"), ds / "images/test" / name
    if not lab.exists():
        return None
    h, w = cv2.imread(str(img)).shape[:2]
    kp = np.array(lab.read_text().split()[5:], float).reshape(17, 3) * [w, h, 1]
    return feet_dir(kp)


def sideview(args):
    """Side-view labels on the test frames with hand-drawn leg sides, then check the geometry.

        python -m swimsync.dataset.facing sideview          # Label Studio "M2 side view - test"
        python -m swimsync.dataset.facing sideview check    # sides from the labels vs hand sides
        python -m swimsync.dataset.facing sideview train    # "M2 side view - train": the training
                                                            # frames answered "can't tell" in front/back
    """
    import random
    import shutil
    from collections import Counter

    from swimsync.dataset.export_yolo import labeled_frames, project_titles

    api, frames = _api(), _hand_test_legs()
    if args == ["check"]:
        lab, n = labeled_frames(api, SIDEVIEW_TITLE)
        k = Counter()
        for img, left, right, *_ in frames:
            if img.name not in lab:
                continue
            view, nearer = view_choices(lab[img.name])
            a, b = sorted([left, right], key=lambda bx: bx[0] + bx[2])
            feet = _hand_feet(img.name)
            k["feet from keypoints"] += feet is not None
            sides = sides_from_view(view, nearer, a, b, FEET_UP if feet is None else feet)
            group = view if view in ("front", "back", "can't tell") else "side / top / bottom"
            k[group, "frames"] += 1
            if sides is not None:
                k[group, "given"] += 1
                k[group, "right"] += sides == (("left", "right") if a is left else ("right", "left"))
        print(f"{len(lab)} of {n} frames labeled; feet direction from keypoints on {k['feet from keypoints']}")
        for g in ("front", "back", "side / top / bottom", "can't tell"):
            print(f"  {g:19s}: {k[g, 'frames']} frames, sides given {k[g, 'given']}, "
                  f"right {k[g, 'right']} ({100 * k[g, 'right'] / max(k[g, 'given'], 1):.0f}%)")
        given = sum(k[g, "given"] for g in ("front", "back", "side / top / bottom"))
        right = sum(k[g, "right"] for g in ("front", "back", "side / top / bottom"))
        print(f"  all: sides given on {given} of {len(lab)}, right {right} ({100 * right / max(given, 1):.0f}%)")
        return
    titles = project_titles(api)
    if args == ["train"]:
        if SIDEVIEW_TRAIN_TITLE in titles:
            print(f"'{SIDEVIEW_TRAIN_TITLE}' already exists; not importing again.")
            return
        import cv2

        root = Path("data/raw/frames/m2")
        legs = {r["file"]: r["legs"] for r in json.loads((root / "facing_train.json").read_text())}
        lab, _ = labeled_frames(api, "M2 front/back - train")
        names = sorted(n for n, r in lab.items() if facing_choice(r) == "unsure")
        img_dir = (root / "images" / "facing_train").resolve()
        tasks = []
        for name in names:
            h, w = cv2.imread(str(img_dir / name)).shape[:2]
            tasks.append(view_task(f"/data/local-files/?d=images/facing_train/{name}", legs[name], w, h))
        random.Random(0).shuffle(tasks)
        pid = api.post("/api/projects", {"title": SIDEVIEW_TRAIN_TITLE, "label_config": view_config()})["id"]
        api.post("/api/storages/localfiles", {"project": pid, "title": "frames", "path": str(img_dir),
                                              "use_blob_urls": False})
        api.post(f"/api/projects/{pid}/import", tasks)
        print(f"'{SIDEVIEW_TRAIN_TITLE}': project {pid}, {len(tasks)} frames")
        return
    if SIDEVIEW_TITLE in titles:
        api.patch(f"/api/projects/{titles[SIDEVIEW_TITLE]}", {"label_config": view_config()})
        print(f"'{SIDEVIEW_TITLE}' already exists; updated its questions, not importing again.")
        return
    img_dir = Path("data/raw/frames/m2/images/sideview_test").resolve()
    img_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for img, left, right, w, h in frames:
        shutil.copy(img, img_dir / img.name)
        tasks.append(view_task(f"/data/local-files/?d=images/sideview_test/{img.name}", [left, right], w, h))
    random.Random(0).shuffle(tasks)
    pid = api.post("/api/projects", {"title": SIDEVIEW_TITLE, "label_config": view_config()})["id"]
    api.post("/api/storages/localfiles", {"project": pid, "title": "frames", "path": str(img_dir),
                                          "use_blob_urls": False})
    api.post(f"/api/projects/{pid}/import", tasks)
    print(f"'{SIDEVIEW_TITLE}': project {pid}, {len(tasks)} frames")


def main():
    import cv2
    from ultralytics import YOLO

    from swimsync.dataset.export_yolo import project_titles
    from swimsync.waterline.above_water import pick_device

    if sys.argv[1:] == ["export"]:
        return export()
    if sys.argv[1:2] == ["sideview"]:
        return sideview(sys.argv[2:])
    n_total = {"train": int(sys.argv[1]) if len(sys.argv) > 1 else 1000, "test": 150}
    root = Path("data/raw/frames/m2").resolve()
    scans = [json.loads(p.read_text()) for p in sorted((root / "scan").glob("*.json"))]
    groups = {"train": {}, "test": {}}
    for s in scans:
        video = Path(s["video"])
        groups["test" if video.parent.name == "test" else "train"][video] = [c["t"] for c in s["candidates"]]
    cache = root / "facing_legs.json"
    found = json.loads(cache.read_text()) if cache.exists() else {}
    todo = {v: t for g in groups.values() for v, t in g.items() if str(v) not in found}
    if todo:
        found.update(find_two_legs(todo, YOLO("experiments/limbs/output/seg_b3/weights/best.pt"), pick_device()))
        cache.write_text(json.dumps(found))

    api = _api()
    titles = project_titles(api)
    for split, videos in groups.items():
        title = f"M2 front/back - {split}"
        if title in titles:
            print(f"'{title}' already exists (id {titles[title]}); not importing again.")
            continue
        # share per video: by the square root of its two-leg frames, so the long Budapest
        # broadcasts don't fill the set
        avail = {v: found[str(v)] for v in videos}
        weight = {v: np.sqrt(len(f)) for v, f in avail.items()}
        total_w = sum(weight.values()) or 1
        img_dir = root / "images" / f"facing_{split}"
        img_dir.mkdir(parents=True, exist_ok=True)
        tasks, rows = [], []
        for v, frames in sorted(avail.items(), key=lambda x: str(x[0])):
            n = round(n_total[split] * weight[v] / total_w)
            by_t = dict((t, b) for t, b in frames)
            cap = cv2.VideoCapture(str(v))
            for t in pick_spread(list(by_t), n, min_gap=1.0):
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
                if not ok:
                    continue
                name = f"{Path(v).stem}_{t:07.1f}s.jpg"
                cv2.imwrite(str(img_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                h, w = frame.shape[:2]
                tasks.append(facing_task(f"/data/local-files/?d=images/facing_{split}/{name}", by_t[t], w, h))
                rows.append({"file": name, "video": Path(v).name, "t": t, "legs": by_t[t]})
        (root / f"facing_{split}.json").write_text(json.dumps(rows, indent=0))
        import random
        random.Random(0).shuffle(tasks)
        pid = api.post("/api/projects", {"title": title, "label_config": facing_config()})["id"]
        api.post("/api/storages/localfiles", {"project": pid, "title": "frames", "path": str(img_dir),
                                              "use_blob_urls": False})
        api.post(f"/api/projects/{pid}/import", tasks)
        print(f"'{title}': project {pid}, {len(tasks)} frames from {len(avail)} videos")


if __name__ == "__main__":
    main()
