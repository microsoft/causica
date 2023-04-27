from typing import Any

import fsspec


def get_storage_options_for_path(path: str) -> dict[str, Any]:
    """Get storage options for a given path.

    This checks whether a path uses the abfs protocol and if so returns storage options for the Azure Blob Storage.

    Args:
        path: fsspec compatible path

    Returns:
        Dictionary of storage options for the given path.
    """
    fs, _ = fsspec.core.url_to_fs(path)
    if fs.protocol == "abfs":
        return {"az": {"anon": False}}
    return {}
