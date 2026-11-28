from swimsync.dataset.export_knees import knee_line, leg_crop


def test_knee_line_normalises_to_the_crop():
    # crop (100, 100)-(300, 300); leg box fills its middle; knee at the crop's centre
    line = knee_line((200, 200), (150, 120, 250, 280), (100, 100, 300, 300))
    v = line.split()
    assert v[0] == "0"
    assert [float(x) for x in v[1:5]] == [0.5, 0.5, 0.5, 0.8]
    assert [float(x) for x in v[5:]] == [0.5, 0.5, 2.0]


def test_no_knee_gives_invisible_point():
    v = knee_line(None, (150, 120, 250, 280), (100, 100, 300, 300)).split()
    assert [float(x) for x in v[5:]] == [0.0, 0.0, 0.0]


def test_leg_crop_pads_and_stays_inside_the_image():
    x1, y1, x2, y2 = leg_crop((0, 50, 100, 250), 1000, 500, pad=0.3)
    assert (x1, y1) == (0, 0)          # moved inside the image
    assert x2 - x1 == y2 - y1 == 320   # square: longer side 200 + 2 x 30%


from swimsync.dataset.export_knees import leg_line17  # noqa: E402


def test_leg_line17_puts_each_joint_in_both_side_slots():
    v = [float(x) for x in leg_line17({"knee": (200, 200), "ankle": None, "hip": (150, 300)},
                                      (150, 120, 250, 280), (100, 100, 300, 300)).split()]
    kp = v[5:]
    assert len(kp) == 17 * 3
    assert kp[13 * 3:13 * 3 + 3] == kp[14 * 3:14 * 3 + 3] == [0.5, 0.5, 2.0]   # knee in both slots
    assert kp[15 * 3:15 * 3 + 3] == [0.0, 0.0, 0.0]                             # no ankle
    assert kp[11 * 3:11 * 3 + 3] == [0.25, 1.0, 2.0]                            # hip


from swimsync.dataset.export_knees import height_leg_box, height_legs  # noqa: E402


def _pt(name, x, y, w=1000, h=500):
    return {"type": "keypointlabels", "original_width": w, "original_height": h,
            "value": {"x": 100 * x / w, "y": 100 * y / h, "keypointlabels": [name]}}


def _ans(leg, answer):
    return {"type": "choices", "from_name": leg, "value": {"choices": [answer]}}


def test_height_legs_keep_knee_out_with_its_knee_and_knee_under_without():
    r = [_pt("1 toes", 100, 50), _pt("1 knee", 120, 150), _pt("1 water", 140, 250), _ans("leg1", "knee out"),
         _pt("2 toes", 400, 100), _pt("2 water", 420, 200), _ans("leg2", "knee under water")]
    assert height_legs(r) == [((100, 50), (120, 150), (140, 250)), ((400, 100), None, (420, 200))]


def test_height_legs_skip_legs_a_crop_cannot_use():
    r = [_pt("1 toes", 100, 50), _pt("1 water", 140, 250), _ans("leg1", "knee out"),      # no knee point
         _pt("2 water", 420, 200), _ans("leg2", "knee under water")]                       # no toes
    assert height_legs(r) == []
    r = [_pt("1 toes", 100, 50), _pt("1 water", 140, 250), _ans("leg1", "not out of the water"),
         _ans("leg2", "pressed to leg 1")]
    assert height_legs(r) == []


def test_height_leg_box_uses_the_model_box_holding_the_leg():
    arm, leg = (500, 0, 600, 100), (90, 40, 150, 260)
    assert height_leg_box((100, 50), (120, 150), (140, 250), [arm, leg]) == leg


def test_height_leg_box_falls_back_to_a_box_around_the_points():
    arm = (500, 0, 600, 100)
    x1, y1, x2, y2 = height_leg_box((100, 50), None, (100, 250), [arm])
    assert (x1, x2) == (76, 124) and (y1, y2) == (26, 274)   # padded 12% of the 200 px leg


from swimsync.dataset.export_knees import negatives_allowed, unmatched_boxes  # noqa: E402


def test_unmatched_boxes_leave_out_used_and_overlapping_boxes():
    used = (100, 100, 200, 300)
    same_leg_again = (110, 120, 210, 310)
    arm = (500, 100, 600, 200)
    assert unmatched_boxes([used, same_leg_again, arm], [used]) == [arm]


def test_negatives_only_when_every_leg_is_accounted_for():
    out = [_pt("1 toes", 100, 50), _pt("1 knee", 120, 150), _pt("1 water", 140, 250), _ans("leg1", "knee out")]
    assert negatives_allowed(out + [_ans("leg2", "not out of the water")])
    assert negatives_allowed(out + [_ans("leg2", "knee under water")])          # no points: not out
    assert not negatives_allowed(out + [_ans("leg2", "pressed to leg 1")])     # the 2nd box is a leg
    no_knee = [_pt("1 toes", 100, 50), _pt("1 water", 140, 250), _ans("leg1", "knee out")]
    assert not negatives_allowed(no_knee + [_ans("leg2", "not out of the water")])
    assert not negatives_allowed(out)                                          # leg 2 not answered


def test_no_leg_crops_skip_a_model_box_over_a_clicked_leg(tmp_path):
    import cv2
    import numpy as np

    from swimsync.dataset.export_knees import add_height_crops

    cv2.imwrite(str(tmp_path / "v_00001.0s.jpg"), np.zeros((500, 1000, 3), np.uint8))
    for d in ("images/train", "labels/train"):
        (tmp_path / "out" / d).mkdir(parents=True)
    r = [_pt("1 toes", 100, 50), _pt("1 knee", 120, 150), _pt("1 water", 140, 250), _ans("leg1", "knee out"),
         _ans("leg2", "not out of the water")]
    short_box = (60, 30, 200, 120)    # the model framed the leg short of its water point
    arm = (600, 100, 700, 200)
    counts = add_height_crops({"train": {"v_00001.0s.jpg": r}}, tmp_path, tmp_path / "out",
                              lambda img: [short_box, arm], no_leg=True)
    assert counts["train"] == (1, 1, 1)   # one leg crop with its knee, one "no leg" crop (the arm)
