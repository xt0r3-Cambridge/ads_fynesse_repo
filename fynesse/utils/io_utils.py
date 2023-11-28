import pickle
from pathlib import Path

import pandas as pd

from ..config import *

ROOT_DIR = Path(".")
MODULE_ROOT = Path(__file__).parent.parent.resolve()


def save(
    data,
    name,
    cache_dir="../cache",
    is_pandas=False,
    use_module_root=False,
):
    """Saves data to a file.
    @param data: The data to be saved.
    @param name: The name of the file to be saved.
    @param cache_dir: The directory to save the file in.
    @param is_pandas: Whether the data is a pandas dataframe.
    @param use_module_root: Whether to use the module root as the cache directory.
    @return: None
    """
    if use_module_root:
        cache_dir = MODULE_ROOT / cache_dir

    filepath = Path(cache_dir) / name
    filepath.parents[0].mkdir(parents=True, exist_ok=True)

    if is_pandas:
        data.to_pickle(filepath)
    else:
        with open(filepath, "wb") as file:
            pickle.dump(data, file)


def load(
    name,
    cache_dir="../cache",
    is_pandas=False,
    use_module_root=False,
    force_reload=False,
):
    """
    Loads data from a file.
    @param name: The name of the file to be loaded.
    @param cache_dir: The directory to load the file from.
    @param is_pandas: Whether the data is a pandas dataframe.
    @param use_module_root: Whether to use the module root as the cache directory.
    @param force_reload: Whether to force reload the data.
    @return: The loaded data.
    """
    if use_module_root:
        cache_dir = MODULE_ROOT / cache_dir
    filepath = Path(cache_dir) / name
    data = None
    if is_pandas:
        data = pd.read_pickle(filepath)
    else:
        with open(filepath, "rb") as file:
            data = pickle.load(file)
    return data


def get_or_load(
    name,
    loader,
    *args,
    cache_dir="../cache",
    is_pandas=False,
    use_module_root=False,
    force_reload=False,
    **kwargs
):
    """
    Gets or loads data from a file.
    @param name: The name of the file to be loaded.
    @param loader: The function to load the data.
    @param args: The arguments to pass to the loader function.
    @param cache_dir: The directory to load the file from.
    @param is_pandas: Whether the data is a pandas dataframe.
    @param use_module_root: Whether to use the module root as the cache directory.
    @param force_reload: Whether to force reload the data.
    @param kwargs: The keyword arguments to pass to the loader function.
    @return: The loaded data.
    """
    if use_module_root:
        cache_dir = MODULE_ROOT / cache_dir
    filepath = Path(cache_dir) / name

    data = None
    if not filepath.exists() or force_reload or config["global_force_reload"]:
        data = loader(*args, **kwargs)
        save(
            data=data,
            name=name,
            cache_dir=cache_dir,
            is_pandas=is_pandas,
        )
    else:
        data = load(
            name=name,
            cache_dir=cache_dir,
            is_pandas=is_pandas,
            force_reload=force_reload,
        )

    return data
