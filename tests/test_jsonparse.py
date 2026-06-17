from clipopedia.jsonparse import extract_json


def test_plain_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_json_with_preamble():
    assert extract_json('Sure! Here you go: {"a": [1, 2]} hope that helps') == {"a": [1, 2]}


def test_braces_inside_strings_are_ignored():
    assert extract_json('{"text": "a } brace"}') == {"text": "a } brace"}


def test_garbage_returns_none():
    assert extract_json("no json here") is None
    assert extract_json("") is None
    assert extract_json(None) is None
