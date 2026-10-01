"""
Baseline forecasting models for SmartBiz AI V1.0.
Implements simple, interpretable baselines for daily demand forecasting:
    - Seasonal Naive
    - Moving Average
    - Exponential Smoothing

These baselines serve as reference points to evaluate more complex models.
"""

import pandas as pd
import numpy as np
import logging


class SeasonalNaive:
    """
    Seasonal Naive forecaster.

    Repeats the value observed `season_length` periods ago as the forecast.
    For daily data with a weekly season, season_length=7 predicts that
    next Monday's demand equals last Monday's demand.

    Attributes:
        season_length (int): Number of periods in one seasonal cycle.
        history (np.ndarray): Stored training data.
    """

    def __init__(self, season_length=7):
        self.season_length = season_length
        self.history = None

    def fit(self, y):
        """
        Store the training series.

        Args:
            y (array-like): Training target values.
        """
        self.history = np.array(y, dtype=float)
        logging.info(
            f"SeasonalNaive fitted with {len(self.history)} observations "
            f"(season_length={self.season_length})."
        )

    def predict(self, horizon):
        """
        Generate forecasts for the next `horizon` periods.

        Args:
            horizon (int): Number of future periods to forecast.

        Returns:
            np.ndarray: Forecast values.
        """
        if self.history is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        forecasts = []
        n = len(self.history)
        for h in range(horizon):
            # Index into the last full season of the history
            idx = n - self.season_length + (h % self.season_length)
            if idx < 0:
                idx = h % n  # fallback for very short history
            forecasts.append(self.history[idx])

        return np.array(forecasts)


class MovingAverage:
    """
    Simple Moving-Average forecaster.

    Forecasts the mean of the last `window` observations as a flat
    prediction for all future periods.

    Attributes:
        window (int): Number of trailing observations to average.
    """

    def __init__(self, window=7):
        self.window = window
        self.history = None
        self._forecast_value = None

    def fit(self, y):
        """
        Compute the moving-average forecast value from the training series.

        Args:
            y (array-like): Training target values.
        """
        self.history = np.array(y, dtype=float)
        effective_window = min(self.window, len(self.history))
        self._forecast_value = np.mean(self.history[-effective_window:])
        logging.info(
            f"MovingAverage fitted (window={effective_window}). "
            f"Forecast value={self._forecast_value:.2f}"
        )

    def predict(self, horizon):
        """
        Generate a flat forecast for `horizon` periods.

        Args:
            horizon (int): Number of future periods.

        Returns:
            np.ndarray: Constant forecast repeated `horizon` times.
        """
        if self._forecast_value is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        return np.full(horizon, self._forecast_value)


class ExponentialSmoothing:
    """
    Simple Exponential Smoothing (SES) forecaster.

    Produces a single smoothed level that is used as a flat forecast.
    Suitable for data without clear trend or seasonality components,
    or as a lightweight baseline.

    Attributes:
        alpha (float): Smoothing factor in (0, 1). Higher values give
            more weight to recent observations.
    """

    def __init__(self, alpha=0.3):
        if not 0 < alpha < 1:
            raise ValueError("alpha must be between 0 and 1 (exclusive).")
        self.alpha = alpha
        self.history = None
        self._level = None

    def fit(self, y):
        """
        Fit the exponential-smoothing model by computing the final level.

        Args:
            y (array-like): Training target values.
        """
        self.history = np.array(y, dtype=float)
        # Initialize level with the first observation
        level = self.history[0]
        for t in range(1, len(self.history)):
            level = self.alpha * self.history[t] + (1 - self.alpha) * level
        self._level = level
        logging.info(
            f"ExponentialSmoothing fitted (alpha={self.alpha}). "
            f"Final level={self._level:.2f}"
        )

    def predict(self, horizon):
        """
        Generate a flat forecast equal to the final smoothed level.

        Args:
            horizon (int): Number of future periods.

        Returns:
            np.ndarray: Constant forecast repeated `horizon` times.
        """
        if self._level is None:
            raise ValueError("Model has not been fitted. Call fit() first.")

        return np.full(horizon, self._level)


def run_baseline_experiment(train_y, test_y, season_length=7, ma_window=7,
                            ses_alpha=0.3):
    """
    Convenience function: fit all three baselines and return their predictions.

    Args:
        train_y (array-like): Training target values.
        test_y (array-like): Test target values (used only for horizon length).
        season_length (int): Season length for Seasonal Naive.
        ma_window (int): Window size for Moving Average.
        ses_alpha (float): Smoothing parameter for Exponential Smoothing.

    Returns:
        dict: {
            "Seasonal Naive": np.ndarray,
            "Moving Average": np.ndarray,
            "Exponential Smoothing": np.ndarray,
        }
    """
    horizon = len(test_y)
    results = {}

    # --- Seasonal Naive ---
    sn = SeasonalNaive(season_length=season_length)
    sn.fit(train_y)
    results["Seasonal Naive"] = sn.predict(horizon)

    # --- Moving Average ---
    ma = MovingAverage(window=ma_window)
    ma.fit(train_y)
    results["Moving Average"] = ma.predict(horizon)

    # --- Exponential Smoothing ---
    es = ExponentialSmoothing(alpha=ses_alpha)
    es.fit(train_y)
    results["Exponential Smoothing"] = es.predict(horizon)

    logging.info(f"Baseline experiment complete. Horizon={horizon}.")
    return results
