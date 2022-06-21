import json
import os
import pickle
from functools import partial
from typing import Any, Callable, Dict, TextIO, Type, TypeVar

T = TypeVar("T")


def read(path: str, expected_ext: str, read_func: Callable[[TextIO], T]) -> T:
    if not os.path.isfile(path):
        raise IOError(f"File {path} does not exist.")

    _, ext = os.path.splitext(path)
    if ext != expected_ext:
        raise IOError(f"Expected extension for file {path} to be a {expected_ext}. Extension is {ext}.")

    with open(path, "r", encoding="utf-8") as file:
        data = read_func(file)
    return data


def save(data: T, path: str, expected_ext: str, write_func: Callable[[T, TextIO], None]) -> None:
    _, ext = os.path.splitext(path)
    if ext != expected_ext:
        raise IOError(f"Expected extension for file {path} to be a {expected_ext}. Extension is {ext}.")

    with open(path, "w", encoding="utf-8") as file:
        write_func(data, file)


def read_json_as(path: str, t: Type[T]) -> T:
    """
    Reads a json file from disk enabling specification of return type for type checking.
    Args:
        path (str): Path to json file
        t (type): Expected return type of python object.
    """
    data = read(path, ".json", json.load)
    assert isinstance(data, t)
    return data


def save_json(data: Any, path: str) -> None:
    save(data, path, ".json", partial(json.dump, indent=4, sort_keys=True))


def read_txt(path: str) -> str:
    def read_func(file):
        return file.read()

    return read(path, ".txt", read_func)


def save_txt(data: str, path: str) -> None:
    def write_func(data, file):
        file.write(data)

    save(data, path, ".txt", write_func)


def save_pickle(data: Any, path: str) -> None:
    with open(path, "wb") as file:
        pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)


def read_pickle(path: str) -> Any:
    with open(path, "rb") as file:
        output = pickle.load(file)
    return output


def get_nth_parent_dir(path: str, n: int) -> str:
    """
    Get the nth parent directory of a path. E.g. get_nth_parent_dir('/foo/bar/file.txt',2) would return '/foo/'.
    Args:
        path: Path to find parent directory of.
        n: Number of levels of parent directories to traverse.
    Returns:
        path: Path to nth parent directory.
    """
    for _ in range(n):
        path = os.path.dirname(path)
    return path


def format_dict_for_console(d: dict) -> str:
    """
    Format {'a': {'c': 2, 'b': 3}} as
        "a.b": 2
        "a.c": 3

    Args:
        d (dict): dict to format

    Returns:
        str: string representation of flattened dict
    """
    return json.dumps(flatten_keys(d), sort_keys=True, indent=2)


def flatten_keys(d: Dict[str, Any], separator: str = ".") -> Dict[str, Any]:
    # Turn a nested dict into a flat dict
    # Example:
    # flatten_keys({'a': {'b': 3, 'c': {'f': 4}}})
    # returns {'a.b': 3, 'a.c.f': 4}
    flatter = {}
    for k, v in d.items():
        if isinstance(v, dict):
            flatter.update(
                flatten_keys({f"{k}{separator}{subk}": subv for subk, subv in v.items()}, separator=separator)
            )
        else:
            flatter.update({k: v})
    return flatter


def unflatten_keys(d: Dict[str, Any], separator: str = ".") -> dict:
    """
    Turn a flattened dict back into a nested dict. Inverse of flatten_keys, provided the input to flatten_keys
    does not have 'separator' in any of the keys.

    Beware - modifies input dict in-place.

    Example:
    unflatten_keys({'a.b': 3, 'a.c.f': 4})
    returns {'a': {'b': 3, 'c': {'f': 4}}}

    Args:
        d (Dict[str, Any]): dictionary to unflatten
        separator (str, optional): Defaults to ".".

    Returns:
        Unflattened dict
    """

    all_keys = list(d.keys())
    for k in all_keys:
        if separator in k:
            val = d.pop(k)
            head, tail = k.split(separator, maxsplit=1)
            unflattened_val = unflatten_keys({tail: val})
            recursive_update(d, {head: unflattened_val})
        else:
            val = d[k]
            if isinstance(val, dict):
                d[k] = unflatten_keys(val)
    return d


def recursive_update(old: dict, new: dict) -> dict:
    """
    Update the dictionary `old`, which may contain arbitrarily nested dictionaries, with the values from `new`.
    Args:
        old: Base dictionary to update.
        new: New dictionary to overwrite values of `old` with.
    Returns:
        updated: New dictionary containing the values of `old`, updated with values from `new` when applicable.
    """
    for key, val in new.items():
        if isinstance(val, dict):
            if key in old:
                old[key] = recursive_update(old[key], val)
            else:
                old[key] = val
        else:
            old[key] = val
    return old
