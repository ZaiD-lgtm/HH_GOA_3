from hhg3.hashing import canonical_json, sha256_json


def test_canonical_json_is_key_order_independent():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_json_has_no_incidental_whitespace():
    assert canonical_json({"a": 1, "b": [1, 2]}) == b'{"a":1,"b":[1,2]}'


def test_hash_changes_when_any_field_changes():
    base = {"match": {"page_url": "https://x.com/a/status/1"}}
    tampered = {"match": {"page_url": "https://x.com/a/status/2"}}
    assert sha256_json(base) != sha256_json(tampered)


def test_hash_is_stable_across_calls():
    obj = {"schema": "hhg3/face-match-record/v1", "n": 3}
    assert sha256_json(obj) == sha256_json(obj)
