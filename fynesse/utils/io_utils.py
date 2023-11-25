import pickle
import pandas as pd
from pathlib import Path
from ..config import *

ROOT_DIR=Path('.')
MODULE_ROOT=Path(__file__).parent.parent.resolve()

def save(
    data,
    name,
    cache_dir="../cache",
    is_pandas=False,
    use_module_root=False,
):
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
    if use_module_root:
        cache_dir = MODULE_ROOT / cache_dir
    filepath = Path(cache_dir) / name
    
    data = None
    if not filepath.exists() or force_reload or config['global_force_reload']:
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