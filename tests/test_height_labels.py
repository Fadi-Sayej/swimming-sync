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
