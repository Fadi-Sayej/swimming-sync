from swimsync.dataset.export_pose import FLIP_IDX, labeled_points, pose_line


def kp(name, x, y, w=400, h=200):
    return {"type": "keypointlabels", "original_width": w, "original_height": h,
            "value": {"x": 100 * x / w, "y": 100 * y / h, "width": 0.6, "keypointlabels": [name]}}


def box(name, x1, y1, x2, y2, w=400, h=200):
    return {"type": "rectanglelabels", "from_name": "guide", "original_width": w, "original_height": h,
            "value": {"x": 100 * x1 / w, "y": 100 * y1 / h, "width": 100 * (x2 - x1) / w,
                      "height": 100 * (y2 - y1) / h, "rectanglelabels": [name]}}


def test_labeled_points_in_pixels():
    pts = labeled_points([kp("left_knee", 100, 50), kp("right_ankle", 300, 150), box("left_leg", 0, 0, 10, 10)])
    assert pts == {"left_knee": (100.0, 50.0), "right_ankle": (300.0, 150.0)}


def test_a_point_given_twice_makes_the_frame_unusable():
    assert labeled_points([kp("left_knee", 100, 50), kp("left_knee", 300, 50)]) is None


def test_pose_line_is_box_plus_17_points_in_the_crop():
    # crop (100, 0)-(300, 200); knee at (200, 50) -> (0.5, 0.25); limb box (150, 20)-(250, 180)
    line = pose_line({"left_knee": (200, 50)}, [(150, 20, 250, 180)], (100, 0, 300, 200)).split()
    assert line[0] == "0"
    assert [float(v) for v in line[1:5]] == [0.5, 0.5, 0.5, 0.8]  # person box = the limb box
    kps = [float(v) for v in line[5:]]
    assert len(kps) == 17 * 3
    assert kps[13 * 3:13 * 3 + 3] == [0.5, 0.25, 2.0]  # left knee (COCO 13), visible
    assert kps[0:3] == [0.0, 0.0, 0.0]  # nose: not labeled


def test_pose_line_box_also_covers_points_outside_the_limb_boxes():
    line = pose_line({"left_wrist": (290, 190)}, [(150, 20, 250, 180)], (100, 0, 300, 200)).split()
    cx, cy, w, h = map(float, line[1:5])
    assert abs((cx + w / 2) - 0.95) < 1e-6 and abs((cy + h / 2) - 0.95) < 1e-6


def test_flip_idx_swaps_left_and_right():
    assert FLIP_IDX[13] == 14 and FLIP_IDX[5] == 6 and FLIP_IDX[0] == 0
