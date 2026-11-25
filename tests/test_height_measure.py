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
