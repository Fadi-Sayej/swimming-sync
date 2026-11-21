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


def test_square_crop_is_square_and_holds_both_legs():
    from swimsync.dataset.facing import square_crop
    x1, y1, x2, y2 = square_crop([(400, 100, 440, 400), (460, 120, 500, 380)], 1920, 1080, pad=0.2)
    assert x2 - x1 == y2 - y1 == round(300 * 1.4)
    assert x1 <= 400 and y1 <= 100 and x2 >= 500 and y2 >= 400


def test_square_crop_slides_back_inside_the_image():
    from swimsync.dataset.facing import square_crop
    assert square_crop([(0, 900, 100, 1080)], 1920, 1080, pad=0.2) == (0, 828, 252, 1080)


def test_square_crop_shrinks_only_when_bigger_than_the_image():
    from swimsync.dataset.facing import square_crop
    x1, y1, x2, y2 = square_crop([(10, 10, 300, 190)], 320, 200, pad=0.5)
    assert (x2 - x1, y2 - y1) == (200, 200) and 0 <= x1 and x2 <= 320


def test_facing_choice_reads_the_answer():
    from swimsync.dataset.facing import facing_choice
    legs = {"type": "rectanglelabels", "value": {"rectanglelabels": ["leg"]}}
    assert facing_choice([legs, {"type": "choices", "value": {"choices": ["back"]}}]) == "back"
    assert facing_choice([legs, {"type": "choices", "value": {"choices": ["can't tell"]}}]) == "unsure"
    assert facing_choice([legs]) is None


def test_write_facing_dataset_sorts_square_crops_into_class_folders(tmp_path):
    import cv2

    from swimsync.dataset.facing import write_facing_dataset
    img = np.zeros((200, 400, 3), np.uint8)
    cv2.imwrite(str(tmp_path / "a.jpg"), img)
    out = tmp_path / "ds"
    write_facing_dataset(out, {"train": [(tmp_path / "a.jpg", [(100, 50, 120, 150)], "front")],
                               "val": [(tmp_path / "a.jpg", [(100, 50, 120, 150)], "back")]}, max_size=64)
    crop = cv2.imread(str(out / "train/front/a.jpg"))
    assert crop.shape[:2] == (64, 64)
    assert (out / "val/back/a.jpg").exists() and not (out / "train/back").exists()


def test_feet_from_people_uses_the_person_in_the_leg_boxes():
    from swimsync.dataset.facing import feet_from_people
    legs = [(90, 0, 110, 200), (140, 0, 160, 200)]
    other = kp(l_kn=(500, 100), l_an=(500, 200))         # someone else, feet down
    her = kp(l_kn=(100, 150), l_an=(100, 20), r_kn=(150, 150), r_an=(150, 20))  # feet up
    assert np.allclose(feet_from_people([other, her], legs), UP)


def test_feet_from_people_ignores_unsure_points():
    from swimsync.dataset.facing import feet_from_people
    her = kp(l_kn=(100, 150), l_an=(100, 20))
    her[15, 2] = 0.1  # ankle unsure: one point left, no direction
    assert feet_from_people([her], [(90, 0, 110, 200)]) is None
    assert feet_from_people([], [(90, 0, 110, 200)]) is None
