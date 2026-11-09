"""Turning Label Studio labels into a YOLO training set."""

import pytest

from swimsync.dataset.export_yolo import CLASSES, frame_time, split_train_val, to_yolo_lines


def box(label, x, y, w, h):
    return {"type": "rectanglelabels",
            "value": {"x": x, "y": y, "width": w, "height": h, "rectanglelabels": [label]}}


def test_four_classes_keep_the_sides():
    assert CLASSES[4] == ["left_leg", "right_leg", "left_arm", "right_arm"]


def test_box_in_percent_becomes_yolo_center_format():
    (line,) = to_yolo_lines([box("right_leg", 10, 20, 30, 40)], 4)
    cls, cx, cy, w, h = line.split()
    assert cls == "1"
    assert float(cx) == pytest.approx(0.25) and float(cy) == pytest.approx(0.40)
    assert float(w) == pytest.approx(0.30) and float(h) == pytest.approx(0.40)


def test_two_classes_merge_the_sides():
    lines = to_yolo_lines([box("left_leg", 0, 0, 10, 10), box("right_arm", 0, 0, 10, 10)], 2)
    assert [ln.split()[0] for ln in lines] == ["0", "1"]  # leg, arm


def test_boxes_hanging_off_the_image_are_clipped():
    (line,) = to_yolo_lines([box("left_arm", 95, -5, 10, 20)], 4)
    _, cx, cy, w, h = map(float, line.split())
    assert cx + w / 2 <= 1.0 and cy - h / 2 >= 0.0


def test_non_box_results_are_ignored():
    assert to_yolo_lines([{"type": "textarea", "value": {}}], 4) == []


def test_frame_time_comes_from_the_file_name():
    assert frame_time("jr2026_m_solo_tech_final_00072.0s.jpg") == ("jr2026_m_solo_tech_final", 72.0)


def test_val_is_the_last_part_of_each_video():
    names = [f"vidA_{t:07.1f}s.jpg" for t in range(0, 100, 10)] + [f"vidB_{t:07.1f}s.jpg" for t in range(0, 50, 10)]
    train, val = split_train_val(names, val_share=0.2)
    assert set(val) == {"vidA_00080.0s.jpg", "vidA_00090.0s.jpg", "vidB_00040.0s.jpg"}
    assert set(train) | set(val) == set(names) and not set(train) & set(val)
