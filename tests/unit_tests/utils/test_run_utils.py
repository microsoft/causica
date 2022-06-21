import os

import pytest

from causica.utils.run_utils import find_all_model_dirs, find_local_model_dir


def test_missing_model(tmpdir):
    with pytest.raises(FileNotFoundError):
        find_local_model_dir(tmpdir)


def test_find_local_model_dir(tmpdir):
    model_file = os.path.join(tmpdir, "model.pt")
    with open(model_file, "w", encoding="utf-8"):
        # Create empty file
        pass
    d, mid = find_local_model_dir(tmpdir)
    assert d == tmpdir
    assert mid == os.path.basename(tmpdir)


def test_ignore_sub_models(tmpdir):
    with open(os.path.join(tmpdir, "model.pt"), "w", encoding="utf-8"):
        pass
    sub_model_dir = os.path.join(tmpdir, "junk")
    os.mkdir(sub_model_dir)
    with open(os.path.join(sub_model_dir, "model.pt"), "w", encoding="utf-8"):
        # Create empty file
        pass
    d, mid = find_local_model_dir(tmpdir)
    assert d == tmpdir
    assert mid == os.path.basename(tmpdir)


def test_find_two_model_dirs(tmpdir):
    sub_model_dir = os.path.join(tmpdir, "junk")
    os.mkdir(sub_model_dir)
    with open(os.path.join(sub_model_dir, "model.pt"), "w", encoding="utf-8"):
        pass
    second_sub_model_dir = os.path.join(tmpdir, "junk2")
    os.mkdir(second_sub_model_dir)
    with open(os.path.join(second_sub_model_dir, "model.pt"), "w", encoding="utf-8"):
        pass
    assert len(find_all_model_dirs([tmpdir])) == 2
    assert {second_sub_model_dir, sub_model_dir} == set(find_all_model_dirs([tmpdir]))
    with pytest.raises(ValueError, match="There are multiple model directories here, so the request is ambiguous"):
        find_local_model_dir(tmpdir)
