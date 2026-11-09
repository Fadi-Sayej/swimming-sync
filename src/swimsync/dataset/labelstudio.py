"""Label Studio project config, tasks with pre-filled boxes, and project setup.

Start Label Studio with `scripts/labelstudio.sh`, then (once):
    python -m swimsync.dataset.labelstudio
creates the "M2 legs+arms - train" and "- test" projects, links the image folders
and imports the frames picked by `swimsync.dataset.pick_frames`. Running it again
doesn't import twice.
"""

import json
import sys
from pathlib import Path

LABELS = ["left_leg", "right_leg", "left_arm", "right_arm"]
COLORS = {"left_leg": "#2a78d6", "right_leg": "#eb6834", "left_arm": "#2ca58d", "right_arm": "#c43c8a"}


def label_config():
    labels = "\n".join(
        f'    <Label value="{n}" background="{COLORS[n]}" hotkey="{i + 1}"/>' for i, n in enumerate(LABELS)
    )
    return f"""<View>
  <Header value="Her own left/right. Box only what you can see. Unsure of the side? Skip the frame."/>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <RectangleLabels name="label" toName="image">
{labels}
  </RectangleLabels>
  <Text name="source" value="$source"/>
</View>"""


def make_task(image_url, boxes, img_w, img_h, source, model_version="yolo11m-pose"):
    """One Label Studio task; `boxes` ({label: xyxy in pixels}) become pre-filled boxes."""
    result = [
        {
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "original_width": img_w,
            "original_height": img_h,
            "value": {
                "x": 100 * x1 / img_w,
                "y": 100 * y1 / img_h,
                "width": 100 * (x2 - x1) / img_w,
                "height": 100 * (y2 - y1) / img_h,
                "rotation": 0,
                "rectanglelabels": [label],
            },
        }
        for label, (x1, y1, x2, y2) in boxes.items()
    ]
    task = {"data": {"image": image_url, "source": source}}
    task["predictions"] = [{"model_version": model_version, "result": result}] if result else []
    return task


class LabelStudioAPI:
    def __init__(self, url, token):
        import requests

        self.url, self.s = url.rstrip("/"), requests.Session()
        self.s.headers["Authorization"] = f"Token {token}"

    def get(self, path):
        r = self.s.get(self.url + path, timeout=60)
        r.raise_for_status()
        return r.json()

    def post(self, path, body):
        r = self.s.post(self.url + path, json=body, timeout=600)
        if not r.ok:
            raise RuntimeError(f"{path}: {r.status_code} {r.text[:300]}")
        return r.json()


def setup_project(api, title, tasks, images_dir):
    """Create the project, link `images_dir` as local storage and import `tasks`.

    Does nothing if a project with this title already exists (so labels are never duplicated).
    Returns the project id.
    """
    found = api.get("/api/projects?page_size=1000")
    for p in found.get("results", found) if isinstance(found, dict) else found:
        if p["title"] == title:
            print(f"'{title}' already exists (id {p['id']}); not importing again.")
            return p["id"]
    pid = api.post("/api/projects", {"title": title, "label_config": label_config()})["id"]
    # lets Label Studio serve the images; not synced, the tasks below already point to them
    api.post("/api/storages/localfiles", {
        "project": pid, "title": "frames", "path": str(images_dir), "use_blob_urls": False,
    })
    api.post(f"/api/projects/{pid}/import", tasks)
    print(f"'{title}': project {pid}, imported {len(tasks)} frames")
    return pid


def main():
    root = Path("data/raw/frames/m2").resolve()
    cred = dict(
        line.split(": ", 1) for line in Path("data/raw/labelstudio/credentials.txt").read_text().splitlines() if ": " in line
    )
    api = LabelStudioAPI(cred["url"], cred["api_token"])
    for split in ("train", "test"):
        tasks_file = root / f"tasks_{split}.json"
        if not tasks_file.exists():
            sys.exit(f"{tasks_file} not found: run python -m swimsync.dataset.pick_frames first")
        tasks = json.loads(tasks_file.read_text())
        setup_project(api, f"M2 legs+arms - {split}", tasks, root / "images" / split)
    print(f"Open {cred['url']} and log in with {cred['email']} (password in data/raw/labelstudio/credentials.txt)")


if __name__ == "__main__":
    main()
