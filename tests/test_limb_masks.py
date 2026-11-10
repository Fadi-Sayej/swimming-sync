"""Limb outlines from SAM 2: quality check and format conversion."""

import numpy as np
import pytest

from swimsync.dataset.limb_masks import ls_polygon, mask_ok, mask_polygon, yolo_seg_line

H, W = 100, 200


def mask_with(*rects):
    m = np.zeros((H, W), bool)
    for x1, y1, x2, y2 in rects:
        m[y1:y2, x1:x2] = True
    return m


BOX = (50, 20, 100, 80)  # x1, y1, x2, y2: 50 x 60 px


def test_a_limb_filling_part_of_its_box_is_ok():
    assert mask_ok(mask_with((65, 20, 85, 80)), BOX)  # a leg: 40% of the box


def test_an_empty_mask_is_rejected():
    assert not mask_ok(mask_with(), BOX)


def test_a_mask_filling_the_whole_box_is_rejected():
    assert not mask_ok(mask_with((50, 20, 100, 80)), BOX)  # SAM took the water


def test_a_mask_spilling_far_outside_the_box_is_rejected():
    assert not mask_ok(mask_with((65, 20, 85, 80), (0, 0, 200, 15)), BOX)


def test_polygon_of_a_rectangle_in_0_to_1():
    poly = mask_polygon(mask_with((50, 20, 100, 80)))
    xs, ys = poly[:, 0], poly[:, 1]
    assert xs.min() == pytest.approx(50 / W, abs=0.01) and xs.max() == pytest.approx(100 / W, abs=0.01)
    assert ys.min() == pytest.approx(20 / H, abs=0.02) and ys.max() == pytest.approx(80 / H, abs=0.02)


def test_polygon_drops_a_tiny_speck():
    poly = mask_polygon(mask_with((10, 10, 14, 14), (100, 20, 160, 90)))
    assert poly[:, 0].min() > 0.45


def test_polygon_keeps_both_pieces_of_a_limb_cut_in_two():
    # e.g. the face in front of the arm: upper arm on the left, hand on the right
    poly = mask_polygon(mask_with((20, 40, 60, 60), (120, 40, 160, 60)))
    assert poly[:, 0].min() == pytest.approx(20 / W, abs=0.01)
    assert poly[:, 0].max() == pytest.approx(160 / W, abs=0.01)


def test_polygon_of_an_empty_mask_is_none():
    assert mask_polygon(mask_with()) is None


def test_yolo_seg_line():
    line = yolo_seg_line(1, np.array([[0.1, 0.2], [0.3, 0.2], [0.3, 0.4]]))
    assert line == "1 0.100000 0.200000 0.300000 0.200000 0.300000 0.400000"


def test_label_studio_polygon_is_in_percent():
    r = ls_polygon(np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]), "leg", W, H)
    assert r["type"] == "polygonlabels" and r["value"]["polygonlabels"] == ["leg"]
    assert r["value"]["points"][0] == pytest.approx([10, 20])
