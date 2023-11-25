import pandas as pd


def aligned_concat(*args):
    for i, ser in enumerate(args):
        assert ser.index.equals(
            args[0].index
        ), f"""Index mismatch between series #{
        i
} and series #0."""
    return pd.concat(args, axis=1)
