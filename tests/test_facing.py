import numpy as np

from swimsync.dataset.facing import facing, facing_config, facing_task, feet_dir, pick_spread, sides_from_facing


def kp(**pts):
    """COCO keypoints; given ones visible: l_hip, r_hip, l_kn, r_kn, l_an, r_an = (x, y)."""
    idx = {"l_hip": 11, "r_hip": 12, "l_kn": 13, "r_kn": 14, "l_an": 15, "r_an": 16}
    k = np.zeros((17, 3))
    for n, (x, y) in pts.items():
        k[idx[n]] = (x, y, 1)
    return k


DOWN = np.array([0.0, 1.0])  # image y grows downwards
UP = np.array([0.0, -1.0])


def test_feet_dir_points_from_hip_toward_the_ankle():
    assert np.allclose(feet_dir(kp(l_kn=(100, 100), l_an=(100, 200))), DOWN)
    assert np.allclose(feet_dir(kp(l_hip=(100, 300), r_kn=(200, 200), r_an=(200, 100))), UP)  # one leg enough


def test_feet_dir_needs_two_points_on_a_leg():
    assert feet_dir(kp(l_kn=(100, 100), r_an=(200, 200))) is None


def test_standing_facing_the_camera_is_front():
    # her left leg appears on the image's right
    assert facing((300, 0), (100, 0), DOWN) == "front"
    assert facing((100, 0), (300, 0), DOWN) == "back"


def test_upside_down_vertical_facing_the_camera_is_front():
    # feet up; facing the camera her left leg is on the image's left
    assert facing((100, 0), (300, 0), UP) == "front"


def test_mirroring_doesnt_change_front_or_back():
    # mirror x and swap her left/right (a mirrored person): same answer
    left, right = np.array([300.0, 10.0]), np.array([100.0, 0.0])
    m = lambda p: np.array([-p[0], p[1]])  # noqa: E731
    a = np.array([0.2, 1.0])
    assert facing(left, right, a) == facing(m(right), m(left), m(a))


def test_legs_lined_up_along_the_body_give_no_answer():
    assert facing((100, 300), (100, 100), DOWN) is None


def test_sides_from_facing_undoes_facing():
    a_box, b_box = (250, 0, 350, 100), (50, 0, 150, 100)  # a on the image's right
    assert sides_from_facing("front", DOWN, a_box, b_box) == ("left", "right")
    assert sides_from_facing("back", DOWN, a_box, b_box) == ("right", "left")
    assert sides_from_facing("front", UP, a_box, b_box) == ("right", "left")


def test_config_has_three_choices_with_keys():
    c = facing_config()
    assert 'value="front"' in c and 'value="back"' in c and "can't tell" in c
    assert c.count("hotkey=") == 3


def test_facing_task_shows_the_two_leg_boxes():
    t = facing_task("/data/x.jpg", [(40, 20, 80, 100), (100, 20, 140, 100)], 400, 200)
    r = t["predictions"][0]["result"]
    assert len(r) == 2 and r[0]["value"]["x"] == 10.0 and r[0]["from_name"] == "legs"


def test_pick_spread_keeps_gap_and_count():
    times = [float(t) for t in range(100)]
    picks = pick_spread(times, 10, min_gap=2.0)
    assert len(picks) == 10
    assert all(b - a >= 2.0 for a, b in zip(picks, picks[1:]))
    assert pick_spread([1.0, 1.5, 2.0], 5, min_gap=1.0) == [1.0, 2.0]
