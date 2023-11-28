from datetime import datetime
from decimal import Decimal

import pandas as pd


def hash_recur(d):
    """
    Recursively hashes a dictionary.
    @param d: The dictionary to hash.
    @return: The hash of the dictionary.
    """
    try:
        return hash(d)
    except TypeError:
        try:
            return hash(frozenset(d))
        except TypeError:
            return hash(frozenset({(k, hash_recur(v)) for (k, v) in sorted(d.items())}))


def coerce_args(func):
    """
    Coerces arguments to a function to the correct type.
    The order of the coersions is as follows:
    1. Numeric
    2. Datetime
    @param func: The function to coerce arguments for.
    @return: The wrapped function.
    """

    def is_datetime(arg):
        try:
            pd.Timestamp(arg)
            return True
        except Exception as e:
            return False

    def coerce_datetime(arg):
        if is_datetime(arg):
            return pd.Timestamp(arg).date(), True
        return arg, False

    def is_numeric(arg):
        try:
            if isinstance(arg, str) and arg.isnumeric():
                return True
            elif isinstance(arg, float):
                return True
            elif isinstance(arg, Decimal):
                return True
            elif isinstance(arg, int):
                return True
            else:
                return False
        except ValueError:
            return False

    def coerce_numeric(arg):
        if is_numeric(arg):
            return float(arg), True
        return arg, False

    coersions = [
        coerce_numeric,
        coerce_datetime,
    ]

    def coerce(arg):
        for coersion in coersions:
            arg, changed = coersion(arg)
            if changed:
                break
        return arg

    def wrapper(*args, **kwargs):
        args = [coerce(arg) for arg in args]
        kwargs = {k: coerce(v) for k, v in kwargs.items()}

        # Call the original function
        result = func(*args, **kwargs)
        return result

    return wrapper
