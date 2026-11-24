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
