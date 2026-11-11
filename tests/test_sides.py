import numpy as np

from swimsync.limbs.sides import assign_sides, crop_around, to_image


def person(**joints):
    """17 COCO keypoints, all hidden except the given ones: name=(x, y)."""
    idx = {"l_sh": 5, "r_sh": 6, "l_el": 7, "r_el": 8, "l_wr": 9, "r_wr": 10,
           "l_hip": 11, "r_hip": 12, "l_kn": 13, "r_kn": 14, "l_an": 15, "r_an": 16}
    kp = np.zeros((17, 3))
    for name, (x, y) in joints.items():
        kp[idx[name]] = (x, y, 0.9)
    return kp


# left leg straight down at x=100, right leg at x=300
LEGS = person(l_hip=(100, 100), l_kn=(100, 200), l_an=(100, 300),
              r_hip=(300, 100), r_kn=(300, 200), r_an=(300, 300))


def test_one_leg_box_gets_the_side_of_the_leg_it_covers():
    assert assign_sides([("leg", (80, 80, 120, 360))], [LEGS])[0] == ["left"]
    assert assign_sides([("leg", (280, 80, 320, 360))], [LEGS])[0] == ["right"]


def test_two_leg_boxes_get_one_side_each():
    sides, _ = assign_sides([("leg", (280, 80, 320, 360)), ("leg", (80, 80, 120, 360))], [LEGS])
    assert sides == ["right", "left"]


def test_two_boxes_use_the_best_overall_match_not_the_first_greedy_one():
    # box A overlaps the left leg a bit more than the right one, box B overlaps only the left one:
    # the best pairing is A=right, B=left
    legs = person(l_hip=(100, 100), l_kn=(100, 200), l_an=(100, 300),
                  r_hip=(160, 100), r_kn=(160, 200), r_an=(160, 300))
    sides, _ = assign_sides([("leg", (85, 80, 170, 360)), ("leg", (70, 80, 125, 360))], [legs])
    assert sides == ["right", "left"]


def test_no_side_when_the_pose_has_no_limb_there():
    sides, _ = assign_sides([("arm", (500, 500, 600, 600))], [LEGS])
    assert sides == [None]


def test_no_side_without_any_person():
    assert assign_sides([("leg", (80, 80, 120, 360))], [])[0] == [None]


def test_picks_the_person_whose_limbs_match_the_boxes():
    elsewhere = person(l_hip=(900, 100), l_kn=(900, 200), l_an=(900, 300),
                       r_hip=(700, 100), r_kn=(700, 200), r_an=(700, 300))
    sides, kp = assign_sides([("leg", (80, 80, 120, 360))], [elsewhere, LEGS])
    assert sides == ["left"]
    assert kp is LEGS


def test_arms_and_legs_are_matched_separately():
    body = person(l_sh=(100, 0), l_el=(100, 40), l_wr=(100, 80),
                  r_sh=(300, 0), r_el=(300, 40), r_wr=(300, 80),
                  l_hip=(100, 100), l_kn=(100, 200), l_an=(100, 300),
                  r_hip=(300, 100), r_kn=(300, 200), r_an=(300, 300))
    sides, _ = assign_sides([("arm", (280, -10, 320, 110)), ("leg", (280, 80, 320, 360))], [body])
    assert sides == ["right", "right"]


def test_crop_around_pads_the_boxes_and_stays_in_the_image():
    x1, y1, x2, y2 = crop_around([(100, 100, 200, 300)], 1000, 1000, pad=0.5)
    assert (x1, y1, x2, y2) == (0, 0, 300, 400)  # 200 tall -> 100 padding on each side
    assert crop_around([(900, 900, 1000, 1000)], 1000, 1000, pad=0.5)[2:] == (1000, 1000)


def test_to_image_moves_crop_keypoints_back():
    kp = np.array([[10.0, 20.0, 0.9]])
    assert to_image(kp, (100, 50, 400, 300)).tolist() == [[110.0, 70.0, 0.9]]
