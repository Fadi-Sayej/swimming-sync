"""Waterline detection and swimmer selection on synthetic frames.

The synthetic frame mimics an underwater side view (like the Worlds 2025 solo
video): dark-blue wall high up, a row of pink ad banners, and the true water
surface at the bottom edge of the banners.
"""

import numpy as np

from swimsync.waterline.above_water import pick_swimmer
from swimsync.waterline.detect import detect_waterline, line_from_points

H, W = 1080, 1920
TRUE_Y = 418

DARK_GREY = (30, 30, 30)
DARK_BLUE_WALL = (110, 50, 30)  # BGR, reads as "water" by color alone
PINK_BANNER = (180, 80, 240)
NAVY_BANNER = (120, 40, 20)  # blue ad banner, also reads as "water" by color
POOL_BLUE = (200, 120, 40)


def underwater_frame(blue_banner_cols=None):
    f = np.zeros((H, W, 3), np.uint8)
    f[:] = DARK_GREY
    f[170:300] = DARK_BLUE_WALL
    f[300:TRUE_Y] = PINK_BANNER
    f[TRUE_Y:] = POOL_BLUE
    if blue_banner_cols:
        x1, x2 = blue_banner_cols
        f[300:TRUE_Y, x1:x2] = NAVY_BANNER
    return f


def test_waterline_skips_blue_wall_above_the_banners():
    line = detect_waterline(underwater_frame())
    assert line is not None
    assert abs(float(line.y_at(W / 2)) - TRUE_Y) < 6


def test_waterline_stays_level_when_a_blue_banner_touches_the_water():
    # 35% of columns see the navy banner merge with the water, 118 px too high
    line = detect_waterline(underwater_frame(blue_banner_cols=(600, 1272)))
    assert line is not None
    for x in (0, W / 2, W - 1):
        assert abs(float(line.y_at(x)) - TRUE_Y) < 6


LEVEL = line_from_points(0, TRUE_Y, W, TRUE_Y)


def test_pick_swimmer_ignores_bigger_person_above_the_water():
    boxes = np.array([
        [100, 50, 500, 400],     # spectator / judge: big, fully above water
        [900, 380, 1000, 700],   # swimmer: crosses the waterline
    ], dtype=float)
    assert pick_swimmer(boxes, LEVEL) == 1


def test_pick_swimmer_takes_biggest_of_the_people_in_the_water():
    boxes = np.array([
        [900, 380, 1000, 700],   # 100 x 320
        [100, 430, 400, 900],    # 300 x 470, fully underwater
    ], dtype=float)
    assert pick_swimmer(boxes, LEVEL) == 1


def test_pick_swimmer_returns_none_when_everyone_is_above_the_water():
    boxes = np.array([[100, 50, 500, 400], [600, 100, 700, 300]], dtype=float)
    assert pick_swimmer(boxes, LEVEL) is None


def test_pick_swimmer_without_a_waterline_takes_biggest_box():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 50, 50]], dtype=float)
    assert pick_swimmer(boxes, None) == 1


def test_pick_swimmer_with_no_boxes():
    assert pick_swimmer(np.zeros((0, 4)), LEVEL) is None


def test_smoothing_restarts_after_a_scene_cut():
    from swimsync.waterline.detect import SmoothedWaterline, Waterline

    s = SmoothedWaterline()
    for _ in range(50):
        s.update(Waterline(0.0, 5.0, 1.0))  # e.g. logo screen
    for _ in range(3):  # cut back to the pool, held for 3 frames
        line = s.update(Waterline(0.0, 426.0, 1.0))
    assert abs(float(line.y_at(960)) - 426) < 1


def test_smoothing_ignores_a_one_frame_outlier():
    from swimsync.waterline.detect import SmoothedWaterline, Waterline

    s = SmoothedWaterline()
    for _ in range(10):
        s.update(Waterline(0.0, 420.0, 1.0))
    line = s.update(Waterline(0.0, 300.0, 0.2))  # splash: few columns agree
    assert abs(float(line.y_at(960)) - 420) < 1
    line = s.update(Waterline(0.0, 420.0, 1.0))
    assert abs(float(line.y_at(960)) - 420) < 1


def test_smoothing_still_damps_small_jitter():
    from swimsync.waterline.detect import SmoothedWaterline, Waterline

    s = SmoothedWaterline()
    s.update(Waterline(0.0, 420.0, 1.0))
    line = s.update(Waterline(0.0, 430.0, 1.0))
    assert 420 < float(line.y_at(960)) < 430


def test_confident_one_frame_cut_is_taken_at_once():
    # frame 4051 of the solo video: logo screen, then one clear pool frame
    from swimsync.waterline.detect import SmoothedWaterline, Waterline

    s = SmoothedWaterline()
    for _ in range(50):
        s.update(Waterline(0.0, 5.0, 0.9))
    line = s.update(Waterline(0.0, 426.0, 0.62))
    assert abs(float(line.y_at(960)) - 426) < 1


def test_unsure_far_lines_are_held_off_through_a_splash():
    # frames 84-94 of the solo video: splash gives ~443 px with low confidence
    from swimsync.waterline.detect import SmoothedWaterline, Waterline

    s = SmoothedWaterline()
    for _ in range(10):
        s.update(Waterline(0.0, 426.0, 0.25))
    for _ in range(5):
        line = s.update(Waterline(0.0, 470.0, 0.2))
    assert abs(float(line.y_at(960)) - 426) < 1


def test_unsure_far_line_is_taken_if_it_lasts():
    from swimsync.waterline.detect import SmoothedWaterline, Waterline

    s = SmoothedWaterline()
    for _ in range(10):
        s.update(Waterline(0.0, 426.0, 0.25))
    for _ in range(10):
        line = s.update(Waterline(0.0, 470.0, 0.2))
    assert abs(float(line.y_at(960)) - 470) < 1
