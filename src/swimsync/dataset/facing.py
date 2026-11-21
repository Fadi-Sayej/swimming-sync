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


def main():
    import cv2
    from ultralytics import YOLO

    from swimsync.dataset.export_yolo import project_titles
    from swimsync.waterline.above_water import pick_device

    if sys.argv[1:] == ["export"]:
        return export()
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
