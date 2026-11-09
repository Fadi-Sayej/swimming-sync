"""Frame picking and pre-labels for milestone 2 (leg / arm boxes)."""

import numpy as np
import pytest

from swimsync.dataset.limbs import limb_boxes
from swimsync.dataset.labelstudio import LABELS, label_config, make_task
from swimsync.dataset.pick_frames import ring_share, spread_pick, swimmer_index

W, H = 1920, 1080


def keypoints(**points):
    """17x3 COCO keypoints; unnamed points get confidence 0."""
    names = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder",
             "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip",
             "l_knee", "r_knee", "l_ankle", "r_ankle"]
    kp = np.zeros((17, 3))
    for name, (x, y, c) in points.items():
        kp[names.index(name)] = (x, y, c)
    return kp


# --- limb boxes from pose keypoints -------------------------------------------

def test_vertical_leg_box_covers_hip_to_pointed_toes():
    # upside down, leg straight up out of the water: hip low, ankle high
    kp = keypoints(l_hip=(500, 600, .9), l_knee=(500, 500, .9), l_ankle=(500, 400, .9))
    x1, y1, x2, y2 = limb_boxes(kp, W, H)["left_leg"]
    assert y2 >= 600                 # reaches the hip
    assert y1 <= 400 - 0.5 * 100     # goes past the ankle for the pointed foot
    assert x1 < 500 < x2 and x2 - x1 < 80  # stays narrow around a straight leg


def test_arm_box_reaches_past_the_wrist_for_the_hand():
    kp = keypoints(r_shoulder=(800, 300, .9), r_elbow=(900, 300, .9), r_wrist=(1000, 300, .9))
    x1, _, x2, _ = limb_boxes(kp, W, H)["right_arm"]
    assert x1 <= 800 and x2 >= 1000 + 0.5 * 100


def test_limb_needs_two_confident_points():
    kp = keypoints(l_hip=(500, 600, .9), l_knee=(500, 500, .1), l_ankle=(500, 400, .1))
    assert "left_leg" not in limb_boxes(kp, W, H)


def test_box_from_hip_and_ankle_when_knee_is_missing():
    kp = keypoints(r_hip=(500, 600, .9), r_ankle=(500, 400, .9))
    x1, y1, x2, y2 = limb_boxes(kp, W, H)["right_leg"]
    assert y1 <= 400 and y2 >= 600


def test_boxes_are_clipped_to_the_image():
    kp = keypoints(l_shoulder=(10, 10, .9), l_elbow=(5, 5, .9), l_wrist=(0, 0, .9))
    x1, y1, x2, y2 = limb_boxes(kp, W, H)["left_arm"]
    assert x1 >= 0 and y1 >= 0 and x2 <= W and y2 <= H


def test_sides_follow_the_keypoint_names_not_the_image_side():
    # her left leg on the right of the image (facing the camera)
    kp = keypoints(l_hip=(1200, 600, .9), l_knee=(1200, 500, .9),
                   r_hip=(800, 600, .9), r_knee=(800, 500, .9))
    boxes = limb_boxes(kp, W, H)
    assert boxes["left_leg"][0] > boxes["right_leg"][2]


# --- is this person in the pool? -------------------------------------------------

def pool_mask(water_left_of=None):
    m = np.ones((H, W), bool)
    if water_left_of is not None:
        m[:, water_left_of:] = False
    return m


def test_ring_share_is_high_for_a_swimmer_surrounded_by_water():
    assert ring_share(pool_mask(), (900, 500, 1000, 600)) > 0.95


def test_ring_share_is_low_on_the_deck():
    assert ring_share(pool_mask(water_left_of=300), (900, 500, 1000, 600)) < 0.05


def test_swimmer_is_the_biggest_person_in_the_water():
    mask = pool_mask(water_left_of=1200)  # water on the left, deck on the right
    boxes = np.array([
        [1400, 200, 1800, 900],  # coach on the deck: biggest, but not in the water
        [300, 400, 400, 500],    # small person in the water
        [500, 300, 700, 600],    # bigger person in the water -> the swimmer
    ], float)
    assert swimmer_index(boxes, mask) == 2


def test_no_swimmer_when_nobody_is_in_the_water():
    boxes = np.array([[1400, 200, 1800, 900]], float)
    assert swimmer_index(boxes, pool_mask(water_left_of=1200)) is None
    assert swimmer_index(np.zeros((0, 4)), pool_mask()) is None


# --- spreading picks through a video ---------------------------------------------

def test_spread_pick_spreads_over_the_whole_video():
    times = np.arange(0, 1000, 2.0)
    picked = times[spread_pick(times, 10, min_gap=5)]
    assert len(picked) == 10
    assert picked.min() < 100 and picked.max() > 850


def test_spread_pick_keeps_the_minimum_gap():
    times = np.array([0, 1, 2, 3, 4, 10, 11, 12, 30.0])
    picked = times[spread_pick(times, 9, min_gap=5)]
    assert np.all(np.diff(np.sort(picked)) >= 5)


def test_spread_pick_returns_fewer_when_there_are_not_enough():
    assert len(spread_pick(np.array([0, 10, 20.0]), 5, min_gap=5)) == 3
    assert len(spread_pick(np.array([]), 5, min_gap=5)) == 0


# --- Label Studio ------------------------------------------------------------------

def test_label_config_has_the_four_labels():
    xml = label_config()
    for name in LABELS:
        assert f'value="{name}"' in xml


def test_task_boxes_are_in_percent_of_the_image():
    t = make_task("/img/a.jpg", {"left_leg": (192, 108, 384, 540)}, W, H, source="v.mp4@12.0s")
    assert t["data"]["image"] == "/img/a.jpg"
    (r,) = t["predictions"][0]["result"]
    v = r["value"]
    assert v["rectanglelabels"] == ["left_leg"]
    assert v["x"] == pytest.approx(10) and v["y"] == pytest.approx(10)
    assert v["width"] == pytest.approx(10) and v["height"] == pytest.approx(40)
    assert r["original_width"] == W and r["original_height"] == H


def test_task_without_boxes_has_no_prediction():
    t = make_task("/img/b.jpg", {}, W, H, source="v.mp4@1.0s")
    assert t.get("predictions", []) == []


# --- pool water colour (HSV values measured on the real videos) ---------------------

def hsv_patch(h, s, v):
    import cv2
    return cv2.cvtColor(np.full((20, 20, 3), (h, s, v), np.uint8), cv2.COLOR_HSV2BGR)


@pytest.mark.parametrize("name,hsv", [
    ("Junior Worlds Budapest, bright cyan", (94, 239, 218)),
    ("Doha, bright blue", (98, 227, 242)),
    ("Paris, darker teal", (91, 66, 125)),
    ("Budapest World Cup, teal", (88, 155, 144)),
])
def test_pool_water_of_every_pool_counts(name, hsv):
    from swimsync.dataset.pick_frames import pool_mask
    assert pool_mask(hsv_patch(*hsv)).all(), name


def test_navy_sofa_backdrop_is_not_pool_water():
    from swimsync.dataset.pick_frames import pool_mask
    assert not pool_mask(hsv_patch(106, 228, 129)).any()


# --- "is this a pool shot?" (CLIP) and using it when picking ----------------------

def test_pool_probability_adds_up_the_pool_prompts():
    from swimsync.dataset.scene import pool_probability
    kinds = ["pool", "pool", "other"]
    logits = np.log(np.array([0.2, 0.3, 0.5]))  # softmax gives back 0.2, 0.3, 0.5
    assert pool_probability(logits, kinds) == pytest.approx(0.5)


def test_choose_skips_frames_that_are_not_pool_shots():
    from swimsync.dataset.pick_frames import choose
    cands = [{"t": float(t), "kind": "swimmer", "kp": None, "pool": 0.9 if t < 50 else 0.1}
             for t in range(0, 100, 5)]
    picks = choose([{"video": "v.mp4", "candidates": cands}], 20, min_gap=5, min_pool=0.3)
    assert picks and all(c["t"] < 50 for _, c in picks)


# --- Label Studio project setup (with a fake server) --------------------------------

class FakeLabelStudio:
    def __init__(self):
        self.projects, self.storages, self.imported = [], [], {}

    def get(self, path):
        assert path == "/api/projects?page_size=1000"
        return {"results": self.projects}

    def post(self, path, body):
        if path == "/api/projects":
            p = {"id": len(self.projects) + 1, "title": body["title"]}
            self.projects.append(p)
            return p
        if path == "/api/storages/localfiles":
            self.storages.append(body)
            return {"id": len(self.storages)}
        if path.startswith("/api/projects/") and path.endswith("/import"):
            pid = int(path.split("/")[3])
            self.imported[pid] = self.imported.get(pid, 0) + len(body)
            return {"task_count": len(body)}
        raise AssertionError(path)


def test_setup_creates_project_storage_and_imports_tasks():
    from swimsync.dataset.labelstudio import setup_project
    api = FakeLabelStudio()
    pid = setup_project(api, "M2 train", [{"data": {}}] * 3, "/abs/frames/images/train")
    assert api.projects[0]["title"] == "M2 train"
    assert api.storages[0]["path"] == "/abs/frames/images/train" and api.storages[0]["project"] == pid
    assert api.imported[pid] == 3


def test_setup_twice_does_not_import_again():
    from swimsync.dataset.labelstudio import setup_project
    api = FakeLabelStudio()
    setup_project(api, "M2 train", [{"data": {}}] * 3, "/abs/images/train")
    setup_project(api, "M2 train", [{"data": {}}] * 3, "/abs/images/train")
    assert len(api.projects) == 1 and api.imported[1] == 3
