"""Leg height on one frame: leg outlines (outline model) + knee (a knee finder) + the score rules."""
import cv2
import numpy as np

from swimsync.height.score import judge_leg, leg_ends, toes_cut

SEG = "experiments/limbs/output/seg_b3/weights/best.pt"
KNEES = (13, 14)  # COCO left / right knee


def inside_leg(point, mask, grow=15):
    """Is the point on this leg: inside its outline grown by `grow` px?"""
    x, y = int(point[0]), int(point[1])
    h, w = mask.shape
    if not (0 <= x < w and 0 <= y < h):
        return False
    y1, y2, x1, x2 = max(y - grow, 0), min(y + grow + 1, h), max(x - grow, 0), min(x + grow + 1, w)
    return bool(mask[y1:y2, x1:x2].any())


def pick_knee(people, mask, conf_min=0.3, grow=15):
    """(x, y, conf) of the surest COCO knee inside the leg outline (grown by `grow` px), or None."""
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)).astype(bool)
    h, w = mask.shape
    found = [(float(kp[i, 0]), float(kp[i, 1]), float(kp[i, 2])) for kp in people for i in KNEES
             if kp[i, 2] >= conf_min and 0 <= int(kp[i, 0]) < w and 0 <= int(kp[i, 1]) < h
             and grown[int(kp[i, 1]), int(kp[i, 0])]]
    return max(found, key=lambda k: k[2], default=None)


class PoseKnees:
    """Knees from the ready-made pose model (weak on legs-only frames: experiments/height/results.md)."""

    def __init__(self, weights="yolo11m-pose.pt", device=None, zoom=False):
        from ultralytics import YOLO

        self.model, self.device, self.zoom = YOLO(weights), device, zoom

    def knees(self, img, legs):
        from swimsync.limbs.sides import pose_people

        people = pose_people(self.model, img, [b for b, _ in legs] if self.zoom else None, self.device)
        return [pick_knee(people, m) for _, m in legs]


def _covered(a, b):
    """Share of the smaller of two outlines that lies inside the other."""
    small = min(a.sum(), b.sum())
    return float(np.logical_and(a, b).sum() / small) if small else 0.0


def drop_repeats(legs, share=0.5):
    """[(box, mask, conf)] without the legs mostly (> `share`) inside a surer leg's outline.

    The outline model sometimes outlines one leg twice (the whole leg and its lower part), or a
    pressed pair and one of its legs; their boxes overlap too little for YOLO's own check (box
    overlap 0.7), so the video showed a third leg (problems #39).
    """
    kept = []
    for leg in sorted(legs, key=lambda leg: -leg[2]):
        if all(_covered(leg[1], k[1]) <= share for k in kept):
            kept.append(leg)
    return kept


def leg_outlines(img, seg_model, device, conf=0.25):
    """[(box, mask)] of the outline model's legs, masks at image size, sorted left to right."""
    r = seg_model.predict(img, imgsz=960, conf=conf, retina_masks=True, verbose=False, device=device)[0]
    if r.masks is None:
        return []
    legs = [(tuple(b), m.astype(bool), s) for c, b, m, s in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist(),
                                                               r.masks.data.cpu().numpy(), r.boxes.conf.tolist())
            if seg_model.names[int(c)] == "leg"]
    return sorted(((b, m) for b, m, _ in drop_repeats(legs)), key=lambda bm: bm[0][0] + bm[0][2])


def measure_frame(img, seg_model, knee_finder, device, conf=0.25):
    """LegHeight per leg the outline model finds, left to right (with its box and outline)."""
    legs = leg_outlines(img, seg_model, device, conf)
    out = []
    for (box, mask), knee in zip(legs, knee_finder.knees(img, legs) if legs else []):
        ends = leg_ends(mask)
        toes, water = ends if ends else (None, None)
        cut = ends is not None and toes_cut(mask, toes, water)
        leg = judge_leg(toes, water, knee[:2] if knee else None, knee[2] if knee else None, toes_cut=cut)
        leg.box, leg.mask = box, mask
        out.append(leg)
    return out


class OurKnees:
    """Knees from our knee model, one crop per leg box (plan Task 8). knees17c: the M2 knee crops +
    Fadi's "M3 leg height - train" legs + "no leg here" crops (arms, bodies); experiments/height/results.md."""

    def __init__(self, weights="experiments/limbs/output/knees17c/weights/best.pt", device=None, conf_min=0.5):
        from ultralytics import YOLO

        self.model, self.device, self.conf_min = YOLO(weights), device, conf_min

    def knees(self, img, legs):
        from swimsync.dataset.export_knees import leg_crop

        h, w = img.shape[:2]
        out = []
        for box, mask in legs:
            x1, y1, x2, y2 = leg_crop(box, w, h)
            r = self.model.predict(img[y1:y2, x1:x2], imgsz=320, conf=0.1, verbose=False, device=self.device)[0]
            if r.keypoints is None or len(r.boxes) == 0:
                out.append(None)
                continue
            kps = r.keypoints.data[int(r.boxes.conf.argmax())]
            kx, ky, kc = (max((kps[i].tolist() for i in KNEES), key=lambda k: k[2]) if len(kps) == 17
                          else kps[0].tolist())
            knee = (kx + x1, ky + y1, kc)
            # the crop may show the other leg too: keep the knee only if it is on this leg
            grow = max(15, round(0.05 * max(box[2] - box[0], box[3] - box[1])))  # 5% of the leg, at least 15 px
            out.append(knee if kc >= self.conf_min and (mask is None or inside_leg(knee, mask, grow)) else None)
        return out
