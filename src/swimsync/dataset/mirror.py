"""Mirrored training copies with her left and right swapped (4-class boxes).

Mirroring an image swaps her left and right, so training normally skips it (fliplr=0). Mirroring
and swapping the labels too gives twice the training frames, and the model can't learn "the left
leg is usually on the image's left": it has to use her body.

    python -m swimsync.dataset.mirror     # data/raw/datasets/m2_4cls -> m2_4cls_mirror
"""

import os
import shutil
from pathlib import Path

import cv2

SWAP = {0: 1, 1: 0, 2: 3, 3: 2}  # left_leg <-> right_leg, left_arm <-> right_arm


def mirror_line(line):
    c, x, y, w, h = line.split()
    return f"{SWAP[int(c)]} {1 - float(x):.6f} {float(y):.6f} {float(w):.6f} {float(h):.6f}"


def _link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def mirror_dataset(src, out):
    src, out = Path(src), Path(out)
    if out.exists():
        shutil.rmtree(out)
    for kind in ("images", "labels"):
        for split_dir in sorted((src / kind).iterdir()):
            if not split_dir.is_dir():
                continue
            dst = out / kind / split_dir.name
            dst.mkdir(parents=True)
            for f in split_dir.iterdir():
                if f.suffix == ".cache":
                    continue
                _link(f, dst / f.name)
                if split_dir.name != "train":
                    continue
                m = dst / f"{f.stem}_mirror{f.suffix}"
                if kind == "images":
                    cv2.imwrite(str(m), cv2.flip(cv2.imread(str(f)), 1))
                else:
                    m.write_text("\n".join(mirror_line(l) for l in f.read_text().splitlines() if l.strip()))
    for f in src.iterdir():
        if f.suffix == ".yaml":
            (out / f.name).write_text(f.read_text().replace(f"path: {src.resolve()}", f"path: {out.resolve()}")
                                      .replace(f"path: {src}", f"path: {out.resolve()}"))
        elif f.suffix == ".txt":
            shutil.copy2(f, out / f.name)  # test lists: absolute paths to the original images


if __name__ == "__main__":
    mirror_dataset("data/raw/datasets/m2_4cls", "data/raw/datasets/m2_4cls_mirror")
    print("Saved to data/raw/datasets/m2_4cls_mirror")
