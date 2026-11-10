"""Score the detector and the outline (seg) model's boxes against the hand-drawn boxes.

The seg model's own val scores its boxes against boxes around the SAM outlines, which are tighter
than hand-drawn ones, so the two models can't be compared that way. This scores both against
the same hand-drawn boxes (data/raw/datasets/m2_2cls), mAP50 per class.

Usage: python experiments/limbs/box_vs_hand.py [val] [newpool] [budapest]
"""
import sys
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

from swimsync.waterline.above_water import pick_device

ds = Path("data/raw/datasets/m2_2cls")
def images(split):
    if split == "val":
        return sorted((ds / "images/val").glob("*.jpg"))
    return [Path(p) for p in (ds / f"test_{split}.txt").read_text().split()]

def score(weights, split):
    m = YOLO(weights)
    tp, conf, pcls, tcls = [], [], [], []
    for img in images(split):
        r = m.predict(str(img), imgsz=960, conf=0.001, iou=0.7, verbose=False, device=pick_device())[0]
        h, w = r.orig_shape
        lab = ds / "labels" / img.parent.name / (img.stem + ".txt")
        g = np.array([list(map(float, l.split())) for l in lab.read_text().splitlines() if l.strip()]).reshape(-1, 5)
        gb = np.c_[(g[:,1]-g[:,3]/2)*w, (g[:,2]-g[:,4]/2)*h, (g[:,1]+g[:,3]/2)*w, (g[:,2]+g[:,4]/2)*h]
        pb, pc, pk = r.boxes.xyxy.cpu(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()
        t = np.zeros(len(pb), bool)
        if len(g) and len(pb):
            iou = box_iou(torch.tensor(gb, dtype=torch.float32), pb).numpy()
            iou[g[:,0][:,None] != pk[None,:]] = 0
            used = set()
            for j in np.argsort(-pc):
                i = int(iou[:, j].argmax())
                if iou[i, j] >= 0.5 and i not in used:
                    used.add(i)
                    t[j] = True
        tp.append(t); conf.append(pc); pcls.append(pk); tcls.append(g[:,0])
    tp = np.concatenate(tp)[:, None]
    res = ap_per_class(tp, np.concatenate(conf), np.concatenate(pcls), np.concatenate(tcls))
    ap50 = res[5][:, 0]
    return ap50.mean(), ap50

for name, w in [("detector", "experiments/limbs/output/2cls/weights/best.pt"),
                ("seg", "experiments/limbs/output/seg/weights/best.pt")]:
    for split in sys.argv[1:] or ["val"]:
        mean, per = score(w, split)
        print(f"{name:9s} {split:9s} box mAP50 vs hand boxes {mean:.3f}  (leg {per[0]:.3f}, arm {per[1]:.3f})")
