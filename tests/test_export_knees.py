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
