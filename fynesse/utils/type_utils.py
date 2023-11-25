from datetime import datetime
import pandas as pd
from decimal import Decimal

def coerce_args(func):
    def is_datetime(arg):
        try:
            pd.Timestamp(arg)
            return True
        except Exception as e:
            return False

    def coerce_datetime(arg):
        if is_datetime(arg):
            pd.Timestamp(arg).date, True
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
        coerce_datetime,
        coerce_numeric,
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