import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from plotly import express as px
from tqdm import tqdm

from utils.pandas_utils import aligned_concat
from utils.stats_utils import r2_score


class SubplotsIter:
    "A class that allows adding new subplots to a figure without having to specify their number in advance."

    def __init__(self, n_cols, max_subplots, subplot_width, subplot_height):
        self.n_cols = n_cols
        self.subplot_width = subplot_width
        self.subplot_height = subplot_height
        self.max_subplots = max_subplots

        self._count = 0

        self.max_rows = (max_subplots - 1) // n_cols + 1
        self.fig = plt.figure(
            1,
            figsize=(
                self.n_cols * self.subplot_width,
                self.max_rows * self.subplot_height,
            ),
        )

    def __iter__(self):
        return self

    def __next__(self):
        "Generates next subplot if we're not over the allowed subplot limit"
        if self._count > self.max_subplots:
            raise StopIteration(
                "Reached the maximum number of subplots. Stopping generation"
            )
        self._count += 1
        return self.fig.add_subplot(self.max_rows, self.n_cols, self._count)


def subplots_iter(
    n_cols=2,  # Number of subplots in a single row
    max_subplots=100,  # The number of subplots we can iterate over before stopping
    subplot_width=6,  # The width of each individual subplot in inches(?)
    subplot_height=4,  # The height of each individual subplot in inches(?)
):
    "Returns an iterator to an axis that dynamically generates subplots"
    return SubplotsIter(
        n_cols=n_cols,
        max_subplots=max_subplots,
        subplot_width=subplot_width,
        subplot_height=subplot_height,
    )


def bin_plot(
    data,
    x,
    y,
    col=None,
    col_wrap=None,
    facet_kws=dict(sharex=False, sharey=False),
    line_kws={"color": "red"},
    x_bins=30,
    **kwargs,
):
    if col is None:
        return sns.regplot(
            data=data,
            x=x,
            y=y,
            x_bins=x_bins,
            line_kws=line_kws,
            **kwargs,
        )

    return sns.lmplot(
        data=data,
        x=x,
        y=y,
        col=col,
        x_bins=x_bins,
        col_wrap=col_wrap,
        facet_kws=facet_kws,
        line_kws=line_kws,
        **kwargs,
    )


def plot_rolling_r2(
    y_true: pd.Series,
    y_pred: pd.Series,
    window_length: pd.Timedelta = pd.Timedelta("50 days"),
    min_nobs=300,
):
    if not isinstance(window_length, pd.Timedelta):
        window_length = pd.Timedelta(window_length)

    data = aligned_concat(y_true, y_pred).copy().dropna()
    data.columns = ["y_true", "y_pred"]

    r2_scores = []

    for win in tqdm(list(data.rolling(window_length))):
        if len(win) < min_nobs:
            r2_scores.append(np.nan)
        else:
            r2_scores.append(
                r2_score(y_true=win.y_true, y_pred=win.y_pred, raise_na=False)
            )

    plot_data = pd.DataFrame(r2_scores, index=data.index, columns=["r2_score"])

    return px.line(plot_data, title="R2 scores per rolling refit")
