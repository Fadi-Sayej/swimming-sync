"""How sure are the leg sides? Relabel the sides of 30 test frames without seeing the old ones.

    python experiments/limbs/recheck_sides.py create   # Label Studio project "M2 leg sides - recheck"
    python experiments/limbs/recheck_sides.py score    # after labeling: agreement with the old labels

Each frame has both legs boxed (from the test labels) as "leg?"; change each box to her own
left or right leg. Can't tell? Leave it as "leg?". The agreement is the best a model trained on
these labels can hope for.
"""
import random
import sys
from collections import Counter
from pathlib import Path

from swimsync.dataset.export_yolo import labeled_frames, project_titles
from swimsync.dataset.labelstudio import LabelStudioAPI, make_task

TITLE = "M2 leg sides - recheck"
TEST = "M2 legs+arms - test"
FRAMES = Path("data/raw/frames/m2/images/test")
N = 30


def config():
    return """<View>
  <Header value="Change each box to her OWN left (1) or right (2) leg. Don't move the boxes. Can't tell? Leave it as leg? and Submit."/>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
    <Label value="left_leg" background="#2a78d6" hotkey="1"/>
    <Label value="right_leg" background="#d6402a" hotkey="2"/>
    <Label value="leg?" background="#999999" hotkey="3"/>
  </RectangleLabels>
</View>"""


def legs(result):
    """[(label, (x, y, w, h) in %)] for the leg boxes of a Label Studio result."""
    return [(r["value"]["rectanglelabels"][0], tuple(round(r["value"][k], 3) for k in ("x", "y", "width", "height")))
            for r in result if r.get("type") == "rectanglelabels" and "leg" in r["value"]["rectanglelabels"][0]]


def api():
    cred = dict(line.split(": ", 1) for line in
                Path("data/raw/labelstudio/credentials.txt").read_text().splitlines() if ": " in line)
    return LabelStudioAPI(cred["url"], cred["api_token"])


def create(a):
    if TITLE in project_titles(a):
        print(f"'{TITLE}' already exists")
        return
    lab, _ = labeled_frames(a, TEST)
    two = sorted(n for n, r in lab.items() if sorted(l for l, _ in legs(r)) == ["left_leg", "right_leg"])
    by_video = {}
    for n in two:
        by_video.setdefault(n.rsplit("_", 1)[0], []).append(n)
    rng = random.Random(0)
    picks = []
    while len(picks) < min(N, len(two)):  # round-robin over the videos
        for v in sorted(by_video):
            if by_video[v] and len(picks) < N:
                picks.append(by_video[v].pop(rng.randrange(len(by_video[v]))))
    rng.shuffle(picks)
    tasks = []
    for n in picks:
        r = [x for x in lab[n] if x.get("type") == "rectanglelabels" and "leg" in x["value"]["rectanglelabels"][0]]
        w, h = r[0]["original_width"], r[0]["original_height"]
        boxes = {}
        for i, x in enumerate(r):
            v = x["value"]
            boxes[f"leg?{i}"] = (v["x"] * w / 100, v["y"] * h / 100,
                                 (v["x"] + v["width"]) * w / 100, (v["y"] + v["height"]) * h / 100)
        t = make_task(f"/data/local-files/?d=images/test/{n}", boxes, w, h, "", model_version="test boxes")
        for p in t["predictions"][0]["result"]:
            p["value"]["rectanglelabels"] = ["leg?"]
        del t["data"]["source"]
        tasks.append(t)
    pid = a.post("/api/projects", {"title": TITLE, "label_config": config()})["id"]
    a.post("/api/storages/localfiles", {"project": pid, "title": "frames", "path": str(FRAMES.resolve()),
                                        "use_blob_urls": False})
    a.post(f"/api/projects/{pid}/import", tasks)
    print(f"'{TITLE}': project {pid}, {len(tasks)} frames ({Counter(n.rsplit('_', 1)[0] for n in picks)})")


def score(a):
    old, _ = labeled_frames(a, TEST)
    new, _ = labeled_frames(a, TITLE)
    c = Counter()
    for n, r in new.items():
        before = dict((box, l) for l, box in legs(old[n]))
        for l, box in legs(r):
            # match by position (boxes weren't moved; allow tiny rounding)
            near = min(before, key=lambda b: sum(abs(p - q) for p, q in zip(b, box)))
            c["boxes"] += 1
            if l == "leg?":
                c["unsure"] += 1
            else:
                c["decided"] += 1
                c["same"] += l == before[near]
    print(f"{len(new)} frames, {c['boxes']} leg boxes: {c['unsure']} left unsure; "
          f"same side as before {c['same']} of {c['decided']} ({100 * c['same'] / max(c['decided'], 1):.0f}%)")


if __name__ == "__main__":
    {"create": create, "score": score}[sys.argv[1]](api())
