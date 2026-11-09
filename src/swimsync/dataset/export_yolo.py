"""Build YOLO training sets from the Label Studio labels (no manual export).

Usage (Label Studio must be running, see scripts/labelstudio.sh):
    python -m swimsync.dataset.export_yolo

Writes data/raw/datasets/m2_4cls (left/right leg and arm) and m2_2cls (leg, arm):
    images/{train,val,test}/   hard links to the picked frames
    labels/{train,val,test}/   one .txt per image: class cx cy w h (0..1)
    data.yaml                  train / val / test = Doha + Paris (new pools)
    data_budapest.yaml         test = Budapest 2024 (same city as training), scored separately

Training frames come from every "M2 legs+arms - train*" project (train, train2, ...).
Only labeled frames are used: skipped frames are left out, frames submitted with no
boxes are kept as "nothing here" examples. val = the last 20% (by time) of each
training video, so its swimmers are not in train.
"""

import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

CLASSES = {
    4: ["left_leg", "right_leg", "left_arm", "right_arm"],
    2: ["leg", "arm"],
}
TO_2 = {"left_leg": "leg", "right_leg": "leg", "left_arm": "arm", "right_arm": "arm"}


def to_yolo_lines(result, n_classes):
    """Label Studio box results (percent, top-left) -> YOLO lines (class cx cy w h, 0..1)."""
    names = CLASSES[n_classes]
    lines = []
    for r in result:
        if r.get("type") != "rectanglelabels":
            continue
        v = r["value"]
        label = v["rectanglelabels"][0]
        if n_classes == 2:
            label = TO_2[label]
        x1, y1 = max(v["x"], 0) / 100, max(v["y"], 0) / 100
        x2, y2 = min(v["x"] + v["width"], 100) / 100, min(v["y"] + v["height"], 100) / 100
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"{names.index(label)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}")
    return lines


def frame_time(name):
    """'jr2026_m_solo_tech_final_00072.0s.jpg' -> ('jr2026_m_solo_tech_final', 72.0)."""
    m = re.match(r"(.+)_(\d+(?:\.\d+)?)s\.jpg$", Path(name).name)
    return m.group(1), float(m.group(2))


def split_train_val(names, val_share=0.2):
    """The last `val_share` of each video's frames (by time) go to val."""
    by_video = defaultdict(list)
    for n in names:
        by_video[frame_time(n)[0]].append(n)
    train, val = [], []
    for frames in by_video.values():
        frames.sort(key=lambda n: frame_time(n)[1])
        k = round(len(frames) * (1 - val_share))
        train += frames[:k]
        val += frames[k:]
    return train, val


def project_titles(api):
    projects = api.get("/api/projects?page_size=1000")
    projects = projects.get("results", projects) if isinstance(projects, dict) else projects
    return {p["title"]: p["id"] for p in projects}


def labeled_frames(api, project_title):
    """{image file name: box results} for every frame with a submitted (not skipped) label."""
    pid = project_titles(api)[project_title]
    tasks = api.get(f"/api/tasks?project={pid}&page_size=10000&fields=all")["tasks"]
    out = {}
    for t in tasks:
        done = [a for a in t.get("annotations", []) if not a.get("was_cancelled")]
        if done:
            out[t["data"]["image"].split("/")[-1]] = done[-1]["result"]
    return out, len(tasks)


def _link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def write_dataset(out, n_classes, splits, frames_root):
    """splits: {split: {file name: (source split folder, results)}}."""
    if out.exists():
        shutil.rmtree(out)
    lists = defaultdict(list)
    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
        for name, (src_split, result) in items.items():
            img = out / "images" / split / name
            _link(frames_root / "images" / src_split / name, img)
            (out / "labels" / split / name).with_suffix(".txt").write_text(
                "\n".join(to_yolo_lines(result, n_classes)))
            if split == "test":
                group = "budapest" if name.startswith("budapest") else "newpool"
                lists[group].append(str(img.resolve()))
    for group, paths in lists.items():
        (out / f"test_{group}.txt").write_text("\n".join(sorted(paths)) + "\n")
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES[n_classes]))
    base = f"path: {out.resolve()}\ntrain: images/train\nval: images/val\n"
    (out / "data.yaml").write_text(base + (
        "test: test_newpool.txt\n" if lists["newpool"] else "") + f"names:\n{names}\n")
    (out / "data_budapest.yaml").write_text(base + (
        "test: test_budapest.txt\n" if lists["budapest"] else "") + f"names:\n{names}\n")


def main():
    from .labelstudio import LabelStudioAPI

    frames_root = Path("data/raw/frames/m2")
    cred = dict(line.split(": ", 1) for line in
                Path("data/raw/labelstudio/credentials.txt").read_text().splitlines() if ": " in line)
    api = LabelStudioAPI(cred["url"], cred["api_token"])

    prefix = "M2 legs+arms - "
    train_lab, n_train = {}, 0  # file name -> (image folder, results)
    for title in sorted(project_titles(api)):
        if title.startswith(prefix + "train"):
            folder = title[len(prefix):]
            lab, n = labeled_frames(api, title)
            train_lab.update({name: (folder, r) for name, r in lab.items()})
            n_train += n
            print(f"{title}: {len(lab)} labeled of {n}")
    test_lab, n_test = labeled_frames(api, prefix + "test")
    train, val = split_train_val(list(train_lab))
    splits = {
        "train": {n: train_lab[n] for n in train},
        "val": {n: train_lab[n] for n in val},
        "test": {n: ("test", r) for n, r in test_lab.items()},
    }
    for n_classes in (4, 2):
        write_dataset(Path(f"data/raw/datasets/m2_{n_classes}cls"), n_classes, splits, frames_root)

    print(f"train {len(train)} + val {len(val)} labeled frames (of {n_train}); test {len(test_lab)} of {n_test}")
    if len(test_lab) < n_test:
        print("Test set not fully labeled yet: re-run this after labeling it.")
    print("Saved to data/raw/datasets/m2_4cls and m2_2cls")


if __name__ == "__main__":
    main()
