"""Hand labels for the leg height score (milestone 3): toes, knee and water point per leg.

Per frame, for each leg out of the water (leg 1 = the leg on the image's left): click the toes,
the knee (only if it is out of the water) and where the leg enters the water, and answer
knee out / knee under water / not out of the water. Leg 2 can also be "pressed to leg 1" (legs
together, seen as one). The hand-drawn leg boxes are shown as a guide.

    python -m swimsync.dataset.height_labels              # Label Studio "M3 leg height - test"
    python -m swimsync.dataset.height_labels add FOLDER   # add frames from images/FOLDER (Instagram)
"""

import random
import sys
from pathlib import Path

TITLE = "M3 leg height - test"
ROOT = Path("data/raw/frames/m2")
STATUSES = ("knee out", "knee under water", "not out of the water")
LEG_COLORS = {"1": "#2a78d6", "2": "#eb6834"}
POINT_KEYS = {"1 toes": "q", "1 knee": "w", "1 water": "e", "2 toes": "a", "2 knee": "s", "2 water": "d"}


def height_config():
    points = "\n".join(f'    <Label value="{n}" background="{LEG_COLORS[n[0]]}" hotkey="{k}"/>'
                       for n, k in POINT_KEYS.items())
    leg1 = "\n".join(f'    <Choice value="{s}" hotkey="{i + 1}"/>' for i, s in enumerate(STATUSES))
    leg2 = "\n".join(f'    <Choice value="{s}" hotkey="{i + 4}"/>' for i, s in enumerate(STATUSES + ("pressed to leg 1",)))
    return f"""<View>
  <Header value="Leg 1 = the leg on the image's left. Click toes, knee (only if out of the water) and where the leg enters the water."/>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <KeyPointLabels name="points" toName="image" strokeWidth="2" opacity="0.9">
{points}
  </KeyPointLabels>
  <Header value="Leg 1"/>
  <Choices name="leg1" toName="image" choice="single-radio" showInline="true" required="true">
{leg1}
  </Choices>
  <Header value="Leg 2"/>
  <Choices name="leg2" toName="image" choice="single-radio" showInline="true" required="true">
{leg2}
  </Choices>
  <RectangleLabels name="guide" toName="image" strokeWidth="1" opacity="0.1">
    <Label value="leg" background="#999999"/>
  </RectangleLabels>
</View>"""


def height_task(image_url, leg_boxes, w, h):
    """Task with the hand leg boxes as a guide, sorted left to right (leg 1 first)."""
    result = [{"from_name": "guide", "to_name": "image", "type": "rectanglelabels",
               "original_width": w, "original_height": h,
               "value": {"x": 100 * x1 / w, "y": 100 * y1 / h, "width": 100 * (x2 - x1) / w,
                         "height": 100 * (y2 - y1) / h, "rotation": 0, "rectanglelabels": ["leg"]}}
              for x1, y1, x2, y2 in sorted(leg_boxes, key=lambda b: b[0] + b[2])]
    return {"data": {"image": image_url}, "predictions": [{"model_version": "hand leg boxes", "result": result}]}


def parse_height(result):
    """{1: {"status", "toes", "knee", "water"}, 2: {...}}, points in pixels or None."""
    legs = {i: {"status": None, "toes": None, "knee": None, "water": None} for i in (1, 2)}
    for r in result:
        if r.get("type") == "keypointlabels":
            name = r["value"]["keypointlabels"][0]
            leg, part = int(name[0]), name[2:]
            legs[leg][part] = (r["value"]["x"] * r["original_width"] / 100, r["value"]["y"] * r["original_height"] / 100)
        elif r.get("type") == "choices" and r.get("from_name") in ("leg1", "leg2"):
            legs[int(r["from_name"][-1])]["status"] = r["value"]["choices"][0]
    return legs


def _test_frames():
    """[(image path, hand leg boxes, w, h)] for the 4-class test frames with at least one leg box."""
    import cv2

    ds = Path("data/raw/datasets/m2_4cls")
    out = []
    for img in sorted((ds / "images/test").glob("*.jpg")):
        h, w = cv2.imread(str(img)).shape[:2]
        legs = []
        for line in (ds / "labels/test" / (img.stem + ".txt")).read_text().splitlines():
            if line.strip():
                c, cx, cy, bw, bh = map(float, line.split())
                if int(c) in (0, 1):
                    legs.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        if legs:
            out.append((img, legs, w, h))
    return out


def main():
    import cv2

    from swimsync.dataset.export_yolo import project_titles
    from swimsync.dataset.facing import _api

    api = _api()
    titles = project_titles(api)
    if sys.argv[1:2] == ["add"]:
        folder = sys.argv[2]
        img_dir = (ROOT / "images" / folder).resolve()
        pid = titles[TITLE]
        api.post("/api/storages/localfiles", {"project": pid, "title": folder, "path": str(img_dir),
                                              "use_blob_urls": False})
        tasks = []
        for img in sorted(img_dir.glob("*.jpg")):
            h, w = cv2.imread(str(img)).shape[:2]
            tasks.append(height_task(f"/data/local-files/?d=images/{folder}/{img.name}", [], w, h))
        api.post(f"/api/projects/{pid}/import", tasks)
        print(f"'{TITLE}': added {len(tasks)} frames from {folder}")
        return
    if TITLE in titles:
        print(f"'{TITLE}' already exists (id {titles[TITLE]}); not importing again.")
        return
    img_dir = (ROOT / "images" / "test").resolve()
    tasks = [height_task(f"/data/local-files/?d=images/test/{img.name}", legs, w, h)
             for img, legs, w, h in _test_frames()]
    random.Random(0).shuffle(tasks)
    pid = api.post("/api/projects", {"title": TITLE, "label_config": height_config()})["id"]
    api.post("/api/storages/localfiles", {"project": pid, "title": "frames", "path": str(img_dir),
                                          "use_blob_urls": False})
    api.post(f"/api/projects/{pid}/import", tasks)
    print(f"'{TITLE}': project {pid}, {len(tasks)} frames")


if __name__ == "__main__":
    main()
