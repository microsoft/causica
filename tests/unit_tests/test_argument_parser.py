import argparse

import pytest

from causica.argument_parser import PathSplitSetter


@pytest.fixture(name="path_split_parser")
def fixture_path_split_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", action=PathSplitSetter, dirname_dest="dirname", basename_dest="basename")
    return parser


@pytest.mark.parametrize("path", ["./foo", "foo/bar", "foo/bar/baz", "../."])
def test_path_split_setter_valid(path: str, path_split_parser: argparse.ArgumentParser):
    namespace = path_split_parser.parse_args(["--path", path])
    assert namespace.path and namespace.dirname and namespace.basename


@pytest.mark.parametrize("path", [".", "", " ", ".."])
def test_path_split_setter_invalid(path: str, path_split_parser: argparse.ArgumentParser):
    with pytest.raises(AssertionError):
        path_split_parser.parse_args(["--path", path])
