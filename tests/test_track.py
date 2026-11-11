import numpy as np

from swimsync.limbs.track import is_anchor, is_cut, pair, track_sides


def body(dx=0.0):
    """A standing person facing the camera: her left side on the image right. Torso and legs visible."""
    kp = np.zeros((17, 3))
    pts = {5: (220, 100), 6: (180, 100), 11: (215, 200), 12: (185, 200),
           13: (215, 280), 14: (185, 280), 15: (215, 360), 16: (185, 360)}
    for i, (x, y) in pts.items():
        kp[i] = (x + dx, y, 0.9)
    return kp


LEFT_BOX, RIGHT_BOX = (205, 190, 225, 410), (175, 190, 195, 410)


def frame(legs, people=(), cut=False):
    return {"legs": list(legs), "people": list(people), "cut": cut}


def test_anchor_needs_shoulders_and_hips():
    assert is_anchor(body())
    kp = body()
    kp[5, 2] = 0.1  # left shoulder unsure
    assert not is_anchor(kp)


def test_pair_keeps_each_leg_on_its_nearest_box():
    tracks = {"left": (0, 0, 10, 100), "right": (50, 0, 60, 100)}
    assert pair(tracks, [(52, 5, 62, 105), (2, 5, 12, 105)]) == {"left": 1, "right": 0}


def test_pair_with_one_box_gives_it_to_the_nearest_leg_only():
    tracks = {"left": (0, 0, 10, 100), "right": (50, 0, 60, 100)}
    assert pair(tracks, [(48, 0, 58, 100)]) == {"right": 0}


def test_sides_carry_over_from_an_anchor_through_frames_without_a_body():
    # frame 0: whole body visible; frames 1-3: only the legs, drifting right; target = frame 3
    frames = [frame([LEFT_BOX, RIGHT_BOX], [body()])]
    for dx in (10, 20, 30):
        frames.append(frame([(x1 + dx, y1, x2 + dx, y2) for x1, y1, x2, y2 in (RIGHT_BOX, LEFT_BOX)]))
    target = [(175 + 30, 190, 195 + 30, 410), (205 + 30, 190, 225 + 30, 410)]
    sides, gap = track_sides(frames, 3, target)
    assert sides == ["right", "left"]
    assert gap == 3


def test_anchor_after_the_target_works_too():
    frames = [frame([RIGHT_BOX, LEFT_BOX]), frame([LEFT_BOX, RIGHT_BOX], [body()])]
    assert track_sides(frames, 0, [LEFT_BOX, RIGHT_BOX])[0] == ["left", "right"]


def test_no_sides_across_a_scene_cut():
    frames = [frame([LEFT_BOX, RIGHT_BOX], [body()]), frame([LEFT_BOX, RIGHT_BOX], cut=True)]
    assert track_sides(frames, 1, [LEFT_BOX, RIGHT_BOX]) == ([None, None], None)


def test_no_sides_without_an_anchor():
    frames = [frame([LEFT_BOX, RIGHT_BOX]), frame([LEFT_BOX, RIGHT_BOX])]
    assert track_sides(frames, 1, [LEFT_BOX, RIGHT_BOX]) == ([None, None], None)


def test_nearest_anchor_wins():
    swapped = body()
    swapped[[11, 12, 13, 14, 15, 16]] = swapped[[12, 11, 14, 13, 16, 15]]  # this frame's pose swaps the legs
    frames = [frame([LEFT_BOX, RIGHT_BOX], [swapped]), frame([LEFT_BOX, RIGHT_BOX]),
              frame([LEFT_BOX, RIGHT_BOX], [body()]), frame([LEFT_BOX, RIGHT_BOX])]
    assert track_sides(frames, 3, [LEFT_BOX, RIGHT_BOX])[0] == ["left", "right"]


def test_cut_between_two_different_pictures_but_not_two_similar_ones():
    blue = np.zeros((90, 160, 3), np.uint8)
    blue[:] = (200, 150, 30)
    blue2 = blue.copy()
    blue2[40:50, 70:90] = (180, 180, 180)  # a swimmer moved in
    red = np.zeros_like(blue)
    red[:] = (20, 30, 200)
    assert not is_cut(blue, blue2)
    assert is_cut(blue, red)
