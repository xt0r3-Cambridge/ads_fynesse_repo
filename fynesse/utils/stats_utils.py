import datetime as dt
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import metrics
from tqdm import tqdm

from .pandas_utils import aligned_concat

def standardise(data, X=None):
    if X is None:
        X = data
    return (data - X.mean()) / X.std()

def parse_preds(y_true, y_pred):
    """
    Combine true and predicted values into a DataFrame with standardized column names.

    Parameters:
    - y_true: True values of the dependent variable.
    - y_pred: Predicted values of the dependent variable.

    Returns:
    - pd.DataFrame: A DataFrame with columns 'y_true' and 'y_pred'.
    """
    y_true = y_true.copy()
    y_pred = y_pred.copy()

    if not isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_frame()
    if not isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_frame()

    y_true.columns = ["y_true"]
    y_pred.columns = ["y_pred"]

    data = pd.concat(
        [
            y_true,
            y_pred,
        ],
        axis=1,
    )

    return data


def r2_score(y_true, y_pred, raise_na=True):
    """
    Calculate the R-squared (R2) score for the given true and predicted values.

    Parameters:
    - y_true: True values of the dependent variable.
    - y_pred: Predicted values of the dependent variable.
    - raise_na (bool): Whether to raise an error if NaN values are detected in the data.

    Returns:
    - float: The R-squared (R2) score for the true and predicted values.
    """
    assert y_true.index.equals(y_pred.index), "Index mismatch"

    data = parse_preds(y_true, y_pred)

    if data.isna().any().any():
        if raise_na:
            print(
                f"Detected NaNs ({round(100 - len(data.dropna()) / len(data) * 100, 2)}% of elements)... dropping"
            )

        data = data.dropna()

    return metrics.r2_score(
        y_true=data.y_true,
        y_pred=data.y_pred,
    )


def male(y_true, y_pred):
    """
    Returns the mean absolute log error.
    """
    y_true = pd.Series(y_true).pipe(np.log10)
    y_pred = pd.Series(y_pred).pipe(np.log10)

    return y_true.sub(y_pred).abs().mean()

def mape(y_true, y_pred): 
    """
    Returns the mean absolute percentage error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) 


def train(df, train_pct = 0.8):
    return df.iloc[: int(len(df) * train_pct)].copy()


def test(df, test_pct=0.2):
    return df.iloc[int(len(df) * (1 - test_pct)) :].copy()


def fit_regression(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    raise_na: str = "raise",
    fill_value=0,
) -> pd.Series:
    """
    This method finds an approximation, β' for the
    statistical model `Y = Xβ + ε`, where
    - X is a matrix of random "error" variables for which
      the following statements are true:
      - `X.shape == (n, d)`
    - Y is a matrix of random "error" variables for which
      the following statements are true:
      - `Y.shape == (n, 1)`
    - β is a coefficient matrix for which
      the following statements are true:
      - `β.shape == (d, 1)
    - ε is a matrix of `n` random variables for which
      the following statements are true:
      - `e.shape == (n, 1)`
      - E[ε] = 0
      - Cov(ε, ε) = σ^2*I, where
        - I is the (n * n) identity matrix
        - Cov(ε, ε) is the covariance matrix of ε,
          where Cov(ε, ε)[i][j] = Cov(ε[i], ε[j])

    Note, how we only know X and Y, and the true underlying value
    of β and the concrete sample of ε in our observed dataset are
    hidden to us.

    If the assumptions above are all met, the returned β' should
    be the best linear unbiased estimator of β, meaning:
    - Best: It minimises E[||w*(b - β)||^2] (the expected weighted
            squared difference from the true value of β) for all
            diagonal weight matrices w of shape `(d * d)`
    - Linear: It is a linear transformation of the matrix Y
    - Unbiased estimator: E[β'] = β
    (Note that all expected values are with respect to ε)

    @param X: A matrix of shape (n * d) of the feature variables
    @param Y: A matrix of shape (n * 1) of the response variables
    @returns: β', the fitted set of coefficients for the model
    @param raise_na: What to do when NA values are encountered in the
                     input. One of {'raise', 'error' 'fill'}
    """
    if not isinstance(Y, pd.DataFrame):
        Y = Y.to_frame()

    assert X.index.equals(Y.index), "Error: Index mismatch between X and Y"

    if raise_na == "error":
        assert not (
            X.isna().any().any() or Y.isna().any().any()
        ), "NaN values encountered in the input, exiting..."
    elif raise_na == "raise":
        if X.isna().any().any() or Y.isna().any().any():
            nans = X.isna().any() | Y.isna().any()
            print(
                f"NaNs encountered in the data ({round(len(nans) / len(X) * 100, 2)}% of {len(X)} elements), therefore result is NaNs"
            )
    elif raise_na == "drop":
        data = aligned_concat(X, Y)
        data = data.dropna()
        X = data[X.columns]
        Y = data[Y.columns]
    elif raise_na == "fill":
        X = X.fillna(fill_value)
        Y = Y.fillna(fill_value)
    else:
        assert False, "Invalid value provided for variable `fill`"

    # Solve the OLS and return the results
    # We use np.linalg.pinv for numerical stability
    return pd.DataFrame(
        np.linalg.pinv((X.T.dot(X))).dot(X.T).dot(Y),
        index=X.columns,
        columns=["beta"],
    )


class RollingRegression:
    """A class for performing rolling regression analysis on time series data."""

    def __init__(
        self,
        X: pd.DataFrame,
        Y: pd.DataFrame,
        window: Union[
            int,
            dt.timedelta,
            str,
        ],
        target_lookahead: Union[
            int,
            dt.timedelta,
            str,
        ],
        expanding: bool = True,
        min_nobs: Union[
            int,
            dt.timedelta,
            str,
        ] = 0,
        refit_freq: Union[
            int,
            dt.timedelta,
            str,
        ] = 1,
        **kwargs,
    ):
        """
        Initialize the RollingRegression object.

        Parameters:
        - X (pd.DataFrame): The independent variable data.
        - Y (pd.DataFrame): The dependent variable data.
        - window (Union[int, dt.timedelta, str]): The rolling window size.
        - target_lookahead (Union[int, dt.timedelta, str]): We are trying to predict data
                                                            `target-lookahead` time in the future.
        - expanding (bool): Whether to use an expanding rolling window (default True).
        - min_nobs (Union[int, dt.timedelta, str]): The minimum number of observations for fitting.
        - refit_freq (Union[int, dt.timedelta, str]): How often the model gets refitted.
        - **kwargs: Additional keyword arguments passed to `fit_regression`
        """
        assert X.index.equals(Y.index), "Error: Index mismatch between X and Y"

        self.X = X.copy()
        self.Y = Y.copy()
        self.window = window
        self.target_lookahead = target_lookahead
        self.expanding = expanding
        self.min_nobs = min_nobs
        self.kwargs = kwargs
        if not self.expanding:
            assert isinstance(
                self.window, int
            ), """Error:
expanding must be set to True and min_nobs must be explicitly provided if the window size is not an integer"""
            self.min_nobs = self.window
        self.refit_freq = refit_freq
        self._params = None
        self._param_ts = "param_ts"
        self._Y_true = "Y_true"

        if not isinstance(self.Y, pd.DataFrame):
            self.Y = self.Y.to_frame()
        if not isinstance(self.window, int):
            self.window = pd.Timedelta(self.window)
        if not isinstance(self.min_nobs, int):
            self.min_nobs = pd.Timedelta(self.min_nobs)
        if not isinstance(self.refit_freq, int):
            self.refit_freq = pd.Timedelta(self.refit_freq)
        if not isinstance(self.target_lookahead, int) and not isinstance(
            self.target_lookahead, pd.Timedelta
        ):
            self.target_lookahead = pd.Timedelta(self.target_lookahead)

    def fit(self):
        """
        Fit rolling regressions to the data and store the parameters.

        Returns:
        - self: The RollingRegression object.
        """
        self._params = {}

        X_rolling = self.X.rolling(window=self.window)
        Y_rolling = self.Y.rolling(window=self.window)

        # We store the last index where we fitted the regression
        # We use this to see if we need to refit
        last_idx = None
        # If we count the items to be fitted by number of data points,
        # we use `skipped_count` to tally the skipped rows
        skipped_count = 0
        if isinstance(self.refit_freq, int):
            skipped_count = self.refit_freq

        for X_win, Y_win in tqdm(list(zip(X_rolling, Y_rolling))):
            skipped_count += 1

            # See if the refit frequency was met
            refit_freq_ok = False
            if isinstance(self.refit_freq, int):
                refit_freq_ok = skipped_count >= self.refit_freq
            elif isinstance(self.refit_freq, pd.Timedelta):
                refit_freq_ok = (
                    last_idx is None or (X_win.index[-1] - last_idx) >= self.refit_freq
                )

            # See if we have the minimum number of observations
            min_nobs_ok = False
            if isinstance(self.min_nobs, int):
                min_nobs_ok = len(X_win) >= self.min_nobs
            elif isinstance(self.min_nobs, pd.Timedelta):
                # The time from the first index in the array is big enough
                min_nobs_ok = (X_win.index[-1] - X_win.index[0]) >= self.min_nobs

            if refit_freq_ok and min_nobs_ok:
                self._params[X_win.index.max()] = fit_regression(
                    X_win, Y_win, **self.kwargs
                ).beta
                skipped_count = 0
                last_idx = X_win.index.max()

        return self

    @property
    def params(self):
        """
        Get the fitted regression parameters.

        Returns:
        - dict: A dictionary of regression parameters with the index of self.X as keys (usually timestamps).
        """
        if self._params is None:
            self.fit()
        return self._params

    def predict(
        self, X: pd.DataFrame, Y: Optional[pd.DataFrame] = None, detailed_preds=False
    ):
        """
        Make predictions using the fitted parameters.

        Parameters:
        - X (pd.DataFrame): The data for making predictions.
        - Y (Optional[pd.DataFrame]): The true dependent variable data (required for detailed predictions).
        - detailed_preds (bool): Whether to return detailed prediction information including timestamp
                                 of the regression parameters used and the true values for the predictions.

        Returns:
        - pd.DataFrame: Predicted values or detailed prediction information.
        """
        if detailed_preds and Y is None:
            assert (
                False
            ), "The dependent variable, `Y` must be defined to get detailed predictions."
        if Y is None:
            Y = pd.DataFrame(np.zeros(len(X)), index=X.index)

        assert X.columns.equals(
            self.X.columns
        ), "Columns mismatch between data used to train model and data used to predict"
        keys = np.sort(np.array(list(self.params.keys())))

        # Find the appropriate parameters for inference
        # If the target_lookahead is x, we need to make sure that at least
        # x time elapsed after fitting the parameters
        latest_param_idx = []
        if isinstance(self.target_lookahead, int):
            latest_param_idx = (
                np.searchsorted(keys, X.index) - 1 - self.target_lookahead
            )
        elif isinstance(self.target_lookahead, pd.Timedelta):
            latest_param_idx = (
                np.searchsorted(keys, X.index - self.target_lookahead) - 1
            )
        else:
            assert (
                False
            ), "the target_lookahead parameter must be datetime-like or an int"

        y = []
        y_extra = []

        for idx, (ts, row), (y_ts, y_values) in tqdm(
            list(zip(latest_param_idx, X.iterrows(), Y.iterrows()))
        ):
            if idx >= 0:
                assert (
                    keys[idx] < ts
                ), f"""Lookahead error on date {ts}:
Attepmted to use parameters from {keys[idx]}
"""

                # print(keys[idx], ts)
                y.append(self.params[keys[idx]].T.dot(row))
                if detailed_preds:
                    y_extra.append([y[-1], keys[idx], y_values.iloc[0]])
            else:
                y.append(np.nan)
                if detailed_preds:
                    y_extra.append([np.nan for i in range(3)])

        if detailed_preds:
            return pd.DataFrame(
                y_extra,
                index=X.index,
                columns=self.Y.columns.tolist() + [self._param_ts, self._Y_true],
            )
        return pd.DataFrame(y, index=X.index, columns=self.Y.columns)

    def plot_pred_error(
        self,
        X: Optional[pd.DataFrame] = None,
        Y: Optional[pd.DataFrame] = None,
        smoothing: Union[
            int,
            dt.timedelta,
            str,
        ] = 60,
        debug=False,
        mse_window=pd.Timedelta("30 days"),
    ):
        """
        Plot prediction errors and related metrics.

        Parameters:
        - X (Optional[pd.DataFrame]): The data used for plotting (default is X during initialization).
        - Y (Optional[pd.DataFrame]): The true dependent variable data (default is Y during initialization).
        - smoothing (Union[int, dt.timedelta, str]): Smoothing window for metrics.
        - debug (bool): Whether to return additional debug information.
        - mse_window (pd.Timedelta): Window for calculating mean squared error.

        Returns:
        - Plotly Figure or Tuple: Prediction errors plot and, if debug is True, additional debug information.
        """
        assert (X is None) == (
            Y is None
        ), "Either both or neither of X and Y should be specified"

        if X is None:
            X = self.X.copy()
        if Y is None:
            Y = self.Y.copy()

        Y_pred = self.predict(X)

        MSE = Y_pred.sub(Y).pow(2).rolling(smoothing).mean()
        true_rolling_MSE = (
            Y.sub(Y.rolling(mse_window).mean())
            .pow(2)
            .rolling(mse_window)
            .mean()
            .iloc(axis=1)[0]
            .rename("rolling MSE of Y")
        )

        TSE = Y.sub(Y.mean()).pow(2).rolling(smoothing).mean()

        R2 = MSE.div(TSE).mul(-1).add(1)

        if debug:
            return Y_pred, Y, R2

        # Note that the metric we should care about is MSE, not R2, as
        # the TSE is going to be really close to 0 if we have a small
        # prediction horizon
        fig = px.line(aligned_concat(MSE, true_rolling_MSE), title="Prediction Errors")

        fig.add_hline(
            Y.sub(Y.mean()).pow(2).mean().iloc[0],
            line_dash="dash",
            annotation_text="response variable MSE",
            annotation_position="top left",
        )

        return fig

    def plot_r2(
        self,
        X: Optional[pd.DataFrame] = None,
        Y: Optional[pd.DataFrame] = None,
        plot_count=False,
    ):
        """
        Plot R-squared (R2) scores for rolling refits.

        Parameters:
        - X (Optional[pd.DataFrame]): The data used for plotting (default is X during initialization).
        - Y (Optional[pd.DataFrame]): The true dependent variable data (default is Y during initialization).
        - plot_count (bool): Whether to include counts in the plot.

        Returns:
        - Plotly Figure: R2 scores plot.
        """
        assert (X is None) == (
            Y is None
        ), "Either both or neither of X and Y should be specified"

        if X is None:
            X = self.X.copy()
        if Y is None:
            Y = self.Y.copy()

        data = self.predict(X, Y, detailed_preds=True).dropna()

        pred_col = Y.columns[0]

        r2_scores = (
            data.groupby(self._param_ts)
            .apply(
                lambda g: np.nan
                if len(g) < 2
                else r2_score(g[self._Y_true], g[pred_col], raise_na=False)
            )
            .sort_index()
        )
        counts = data[self._param_ts].value_counts().sort_index()

        plot_data = aligned_concat(r2_scores, counts).copy()
        plot_data.columns = ["r2_score", "count"]

        if not plot_count:
            plot_data = plot_data[["r2_score"]]

        return px.line(plot_data, title="R2 scores per rolling refit")

    def plot_params(self):
        """
        Plot model parameters over time.

        Returns:
        - Plotly Figure: Model parameters plot.
        """
        return px.line(pd.DataFrame(self.params).T, title="Model Parameters Over Time")
