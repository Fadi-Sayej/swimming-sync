import numpy as np

from swimsync.height.measure import pick_knee


def person(knees):
    """COCO keypoints (17, 3) with the given (x, y, conf) at the knees (13, 14)."""
    kp = np.zeros((17, 3))
    for i, k in zip((13, 14), knees):
        kp[i] = k
    return kp


def leg_mask():
    m = np.zeros((300, 300), bool)
    m[50:250, 100:140] = True
    return m


def test_pick_knee_takes_the_surest_knee_inside_the_leg():
    people = [person([(120, 150, 0.5)]), person([(118, 160, 0.9)])]
    assert pick_knee(people, leg_mask()) == (118, 160, 0.9)


def test_pick_knee_ignores_knees_outside_the_leg():
    assert pick_knee([person([(250, 150, 0.9)])], leg_mask()) is None


def test_pick_knee_ignores_unsure_knees():
    assert pick_knee([person([(120, 150, 0.2)])], leg_mask()) is None


from swimsync.height.measure import inside_leg  # noqa: E402


def test_a_knee_on_the_other_leg_is_not_this_legs_knee():
    m = leg_mask()  # columns 100-140
    assert inside_leg((120, 150), m)
    assert not inside_leg((200, 150), m)       # the other leg, 60 px away
    assert inside_leg((150, 150), m, grow=15)  # just outside the outline: still this leg


from swimsync.height.measure import drop_repeats  # noqa: E402


def _rect_mask(x1, y1, x2, y2, h=100, w=100):
    m = np.zeros((h, w), bool)
    m[y1:y2, x1:x2] = True
    return m


def test_a_leg_outlined_twice_keeps_the_surer_outline():
    whole = ((10, 10, 30, 90), _rect_mask(10, 10, 30, 90), 0.9)
    lower_part = ((10, 50, 30, 90), _rect_mask(10, 50, 30, 90), 0.7)   # inside the whole leg
    other_leg = ((60, 10, 80, 90), _rect_mask(60, 10, 80, 90), 0.8)
    kept = drop_repeats([lower_part, whole, other_leg])
    assert [conf for _, _, conf in kept] == [0.9, 0.8]


def test_pressed_pair_and_one_of_its_legs_keeps_two_legs_when_a_single_leg_is_surer():
    pair = ((10, 10, 50, 90), _rect_mask(10, 10, 50, 90), 0.6)
    left = ((10, 10, 30, 90), _rect_mask(10, 10, 30, 90), 0.9)
    right = ((30, 10, 50, 90), _rect_mask(30, 10, 50, 90), 0.8)
    assert sorted(conf for _, _, conf in drop_repeats([pair, left, right])) == [0.8, 0.9]


def test_legs_that_only_touch_are_both_kept():
    a = ((10, 10, 32, 90), _rect_mask(10, 10, 32, 90), 0.9)
    b = ((28, 10, 50, 90), _rect_mask(28, 10, 50, 90), 0.8)   # 4 px of 22 overlap
    assert len(drop_repeats([a, b])) == 2
