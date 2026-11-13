import numpy as np

from swimsync.dataset.keypoints import POINTS, fix_sides, keypoint_task, kp_config, prefill_points


def person(**joints):
    """17 COCO keypoints, all hidden except the given ones: name=(x, y)."""
    idx = {"l_sh": 5, "r_sh": 6, "l_el": 7, "r_el": 8, "l_wr": 9, "r_wr": 10,
           "l_hip": 11, "r_hip": 12, "l_kn": 13, "r_kn": 14, "l_an": 15, "r_an": 16}
    kp = np.zeros((17, 3))
    for name, (x, y) in joints.items():
        kp[idx[name]] = (x, y, 0.9)
    return kp


# pose model: "left" leg at x=100, "right" leg at x=300
LEGS = person(l_hip=(100, 100), l_kn=(100, 200), l_an=(100, 300),
              r_hip=(300, 100), r_kn=(300, 200), r_an=(300, 300))


def test_points_are_the_12_limb_joints():
    assert len(POINTS) == 12
    assert POINTS["left_knee"] == 13 and POINTS["right_wrist"] == 10


def test_config_has_a_point_label_per_joint():
    c = kp_config()
    assert c.count("<Label value=") == 12 + 4  # 12 points + 4 guide boxes
    assert "KeyPointLabels" in c


def test_fix_sides_keeps_legs_that_match_the_hand_boxes():
    boxes = {"left_leg": (80, 80, 120, 360), "right_leg": (280, 80, 320, 360)}
    assert np.array_equal(fix_sides(LEGS, boxes), LEGS)


def test_fix_sides_swaps_legs_when_the_hand_boxes_say_so():
    boxes = {"right_leg": (80, 80, 120, 360), "left_leg": (280, 80, 320, 360)}
    fixed = fix_sides(LEGS, boxes)
    assert tuple(fixed[13, :2]) == (300, 200)  # left knee now on the leg in the left_leg box
    assert tuple(fixed[14, :2]) == (100, 200)


def test_fix_sides_with_one_hand_box_uses_that_one():
    fixed = fix_sides(LEGS, {"left_leg": (280, 80, 320, 360)})
    assert tuple(fixed[15, :2]) == (300, 300)


def test_fix_sides_swaps_arms_and_legs_separately():
    kp = person(l_hip=(100, 100), l_kn=(100, 200), r_hip=(300, 100), r_kn=(300, 200),
                l_sh=(100, 500), l_el=(100, 600), r_sh=(300, 500), r_el=(300, 600))
    boxes = {"left_leg": (80, 80, 120, 220), "right_arm": (80, 480, 120, 620)}
    fixed = fix_sides(kp, boxes)
    assert tuple(fixed[13, :2]) == (100, 200)  # legs kept
    assert tuple(fixed[8, :2]) == (100, 600)  # arms swapped: right elbow on the right_arm box


def test_prefill_keeps_only_points_inside_their_own_box():
    boxes = {"left_leg": (80, 80, 120, 360)}  # right leg not boxed: not visible, no points
    pts = prefill_points(LEGS, boxes)
    assert sorted(pts) == ["left_ankle", "left_hip", "left_knee"]
    assert pts["left_knee"] == (100.0, 200.0)


def test_prefill_drops_a_point_far_outside_its_box():
    kp = LEGS.copy()
    kp[15, :2] = (100, 900)  # invented ankle far below the box
    assert "left_ankle" not in prefill_points(kp, {"left_leg": (80, 80, 120, 360)})


def test_prefill_drops_unsure_points():
    kp = LEGS.copy()
    kp[13, 2] = 0.1
    assert "left_knee" not in prefill_points(kp, {"left_leg": (80, 80, 120, 360)})


def test_keypoint_task_has_points_in_percent_and_guide_boxes():
    t = keypoint_task("/data/x.jpg", {"left_knee": (100, 50)}, {"left_leg": (80, 20, 120, 90)}, 400, 200)
    pts = [r for r in t["predictions"][0]["result"] if r["type"] == "keypointlabels"]
    assert pts[0]["value"]["x"] == 25.0 and pts[0]["value"]["y"] == 25.0
    assert pts[0]["value"]["keypointlabels"] == ["left_knee"]
    boxes = [r for r in t["predictions"][0]["result"] if r["type"] == "rectanglelabels"]
    assert boxes[0]["from_name"] == "guide" and boxes[0]["value"]["rectanglelabels"] == ["left_leg"]
