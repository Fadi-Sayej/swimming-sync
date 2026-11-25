from swimsync.dataset.height_labels import height_config, height_task, parse_height


def point(name, x, y, w=1000, h=500):
    return {"type": "keypointlabels", "original_width": w, "original_height": h,
            "value": {"x": x, "y": y, "keypointlabels": [name]}}


def choice(leg, answer):
    return {"type": "choices", "from_name": leg, "value": {"choices": [answer]}}


def test_config_has_six_points_and_two_leg_questions():
    c = height_config()
    for s in ('value="1 toes"', 'value="1 knee"', 'value="1 water"', 'value="2 water"',
              'name="leg1"', 'name="leg2"', "pressed to leg 1"):
        assert s in c


def test_task_orders_legs_left_to_right():
    task = height_task("/x.jpg", [(500, 0, 600, 100), (100, 0, 200, 100)], 1000, 500)
    boxes = task["predictions"][0]["result"]
    assert boxes[0]["value"]["x"] == 10.0
    assert boxes[1]["value"]["x"] == 50.0


def test_parse_reads_points_in_pixels_and_status():
    r = [point("1 toes", 50, 20), choice("leg1", "knee under water")]
    assert parse_height(r)[1] == {"status": "knee under water", "toes": (500.0, 100.0), "knee": None, "water": None}


def test_parse_missing_leg_answer_is_none_status():
    r = [choice("leg1", "knee out")]
    assert parse_height(r)[2]["status"] is None


from swimsync.dataset.height_labels import prefill_result  # noqa: E402
from swimsync.height.score import LegHeight  # noqa: E402


def test_prefill_puts_model_points_on_legs_left_to_right():
    right = LegHeight("pass", 0.6, toes=(800, 100), knee=(800, 300), water=(800, 400), box=(750, 50, 850, 420))
    left = LegHeight("fail", toes=(200, 100), water=(200, 400), box=(150, 50, 250, 420))
    r = prefill_result([right, left], 1000, 500)
    names = {x["value"]["keypointlabels"][0]: (x["value"]["x"], x["value"]["y"]) for x in r if x["type"] == "keypointlabels"}
    assert names["1 toes"] == (20.0, 20.0) and names["2 knee"] == (80.0, 60.0)
    assert "1 knee" not in names
    choices = {x["from_name"]: x["value"]["choices"][0] for x in r if x["type"] == "choices"}
    assert choices == {"leg1": "knee under water", "leg2": "knee out"}


def test_prefill_with_one_leg_marks_leg_2_under_water():
    leg = LegHeight("pass", 0.6, toes=(800, 100), knee=(800, 300), water=(800, 400), box=(750, 50, 850, 420))
    choices = {x["from_name"]: x["value"]["choices"][0] for x in prefill_result([leg], 1000, 500) if x["type"] == "choices"}
    assert choices == {"leg1": "knee out", "leg2": "knee under water"}


from swimsync.dataset.height_labels import near_boundary  # noqa: E402


def test_near_boundary_when_the_knee_is_close_to_the_water():
    assert near_boundary(LegHeight("pass", 0.55, toes=(0, 0), knee=(0, 90), water=(0, 100)))
    assert not near_boundary(LegHeight("pass", 0.9, toes=(0, 0), knee=(0, 40), water=(0, 100)))


def test_near_boundary_when_the_leg_fails():
    assert near_boundary(LegHeight("fail", toes=(0, 0), water=(0, 100)))
    assert not near_boundary(LegHeight("no score"))
