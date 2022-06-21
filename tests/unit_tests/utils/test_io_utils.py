import os

from causica.utils.io_utils import (
    flatten_keys,
    format_dict_for_console,
    get_nth_parent_dir,
    read_json_as,
    read_pickle,
    save_json,
    save_pickle,
    unflatten_keys,
)


def test_get_nth_parent_dir():
    path = os.path.join("the", "quick", "brown", "fox", "jumped")
    assert get_nth_parent_dir(path, 0) == os.path.join("the", "quick", "brown", "fox", "jumped")
    assert get_nth_parent_dir(path, 1) == os.path.join("the", "quick", "brown", "fox")
    assert get_nth_parent_dir(path, 2) == os.path.join("the", "quick", "brown")
    assert get_nth_parent_dir(path, 3) == os.path.join("the", "quick")


def test_read_load_json(tmpdir):
    d = {"foo": "bar", "foo2": [1, 2, 3], "foo3": {"foo4": [1, "b", ["c", 3]]}}
    path = os.path.join(tmpdir, "dict.json")
    save_json(d, path)
    d2 = read_json_as(path, dict)
    assert d == d2


def test_read_load_pickle(tmpdir):
    d = {"foo": "bar", "foo2": [1, 2, 3], 3: {"foo4": [1, "b", ["c", 3]]}}
    path = os.path.join(tmpdir, "dict.pkl")
    save_pickle(d, path)
    d2 = read_pickle(path)
    assert d == d2


def test_format_dict_for_console():
    d = {"a": {"c": 2, "b": 3}}
    s = format_dict_for_console(d)
    assert (
        s
        == """{
  "a.b": 3,
  "a.c": 2
}"""
    )


def test_flatten_keys():
    d = {"a": {"c": 2, "b": {"z": 3, "h": "elephant"}}}
    flattened = flatten_keys(d)
    assert flattened == {"a.c": 2, "a.b.z": 3, "a.b.h": "elephant"}


def test_unflatten_keys():
    flattened = {"a.c": 2, "a.b.z": 3, "a.b.h": "elephant"}
    d = unflatten_keys(flattened)
    print("d is", d)
    assert d == {"a": {"c": 2, "b": {"z": 3, "h": "elephant"}}}
