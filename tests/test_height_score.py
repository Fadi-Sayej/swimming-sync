import numpy as np
import pytest

from swimsync.height.score import height_score, judge_leg, knee_is_out, leg_ends


def test_water_at_knee_is_half():
    assert height_score((0, 0), (0, 100), (0, 100)) == pytest.approx(0.5)


def test_water_mid_thigh_is_three_quarters():
    assert height_score((0, 0), (0, 100), (0, 150)) == pytest.approx(0.75)


def test_whole_leg_out_is_capped_at_full():
    assert height_score((0, 0), (0, 100), (0, 260)) == pytest.approx(1.0)


def test_bent_leg_measures_along_the_leg():
    assert height_score((0, 0), (100, 0), (100, 50)) == pytest.approx(0.75)


def test_zero_shin_gives_no_score():
    assert height_score((5, 5), (5, 5), (5, 100)) is None


def test_knee_past_the_water_is_under():
    assert not knee_is_out((0, 0), (0, 120), (0, 100))
    assert judge_leg((0, 0), (0, 100), (0, 120), 0.9).status == "fail"


def test_knee_between_toes_and_water_is_out():
    leg = judge_leg((0, 0), (0, 150), (0, 100), 0.9)
    assert knee_is_out((0, 0), (0, 100), (0, 150))
    assert leg.status == "pass" and leg.score == pytest.approx(0.75)


def test_no_knee_found_is_a_fail():
    leg = judge_leg((0, 0), (0, 100), None, None)
    assert leg.status == "fail" and leg.knee is None


def test_no_ends_is_no_score():
    assert judge_leg(None, None, (0, 50), 0.9).status == "no score"


def test_vertical_mask_ends():
    m = np.zeros((300, 100), bool)
    m[50:250, 40:60] = True
    toes, water = leg_ends(m)
    assert toes[1] < 60 and water[1] > 240


def test_flat_leg_takes_lower_end_as_water():
    m = np.zeros((300, 400), bool)
    for x in range(50, 350):  # bar rising to the right, about 10 degrees
        y = int(200 - (x - 50) * 0.18)
        m[y - 8:y + 8, x] = True
    toes, water = leg_ends(m)
    assert water[0] < toes[0]  # the left end is lower in the image


def test_tiny_mask_gives_no_ends():
    m = np.zeros((100, 100), bool)
    m[10:15, 10:20] = True
    assert leg_ends(m) is None


from swimsync.height.score import toes_cut  # noqa: E402


def test_toes_cut_by_the_frame_edge():
    m = np.zeros((300, 100), bool)
    m[0:200, 40:60] = True  # leg runs off the top of the picture
    toes, water = leg_ends(m)
    assert toes_cut(m, toes, water)


def test_toes_inside_the_frame():
    m = np.zeros((300, 100), bool)
    m[50:250, 40:60] = True
    toes, water = leg_ends(m)
    assert not toes_cut(m, toes, water)


def test_cut_toes_still_pass_but_without_a_score():
    leg = judge_leg((0, 0), (0, 150), (0, 100), 0.9, toes_cut=True)
    assert leg.status == "pass" and leg.score is None


from swimsync.height.score import LegHeight, with_unseen_legs  # noqa: E402


def test_a_leg_not_seen_out_of_the_water_is_a_fail():
    legs = with_unseen_legs([LegHeight("pass", 0.7, (0, 0), (0, 100), (0, 150), 0.9)])
    assert [leg.status for leg in legs] == ["pass", "fail"]
    assert legs[1].toes is None and legs[1].knee is None


def test_no_leg_seen_is_two_fails():
    assert [leg.status for leg in with_unseen_legs([])] == ["fail", "fail"]


def test_legs_without_a_score_do_not_count_as_seen():
    legs = with_unseen_legs([LegHeight("no score")])
    assert [leg.status for leg in legs] == ["no score", "fail", "fail"]
