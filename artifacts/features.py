"""Shared code for the notebook AND the Streamlit app.

Keeping feature building in ONE place means the app computes exactly the same
features the model was trained on (no "training/serving skew").
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

# ---------------------------------------------------------------- AQI helpers
# US EPA PM2.5 (24-hour) breakpoints, 2024 revision. Lower bound of each category.
HIGH_SMOG = 125.5  # "Very Unhealthy" starts here
AQI_BREAKPOINTS = [9.1, 35.5, 55.5, 125.5, 225.5]
AQI_CATEGORIES = [
    "Good",
    "Moderate",
    "Unhealthy for Sensitive Groups",
    "Unhealthy",
    "Very Unhealthy",
    "Hazardous",
]


def pm_category_index(value):
    """0..5 category index for a PM2.5 value (works on scalars and arrays)."""
    return np.digitize(value, AQI_BREAKPOINTS)


def pm_category(value):
    return AQI_CATEGORIES[int(pm_category_index(value))]


# ------------------------------------------------------------ data preparation
FEATURES = [
    "pm_today", "pm_lag1", "pm_lag2", "pm_lag3", "pm_lag7",
    "pm_roll3", "pm_roll7", "pm_roll7_std",
    "temperature", "humidity", "wind", "temp_change1", "wind_change1",
    "doy_sin", "doy_cos",
]


def hourly_to_daily(df, min_hours=18):
    """Hourly -> daily mean. A day with fewer than `min_hours` readings becomes NaN
    (so a half-empty day is never silently treated as a real daily average)."""
    g = df.set_index("time").resample("D")
    daily = g.mean().where(g.count() >= min_hours)
    return daily.reset_index()


def fill_small_gaps(daily, cols, limit=3):
    """Interpolate gaps of at most `limit` days. Returns (frame, number_of_values_filled)."""
    d = daily.copy()
    before = int(d[cols].isna().sum().sum())
    d[cols] = d.set_index("time")[cols].interpolate(
        method="time", limit=limit, limit_area="inside"
    ).values
    after = int(d[cols].isna().sum().sum())
    return d, before - after


def add_features(daily):
    """Build model inputs from a DAILY frame (columns: time, pm2_5, temperature,
    humidity, wind). Every feature on row t uses data from day t or earlier only.
    The frame must contain every calendar day (no dropped rows), so shifts are
    real calendar lags."""
    d = daily.sort_values("time").reset_index(drop=True).copy()
    pm = d["pm2_5"]
    d["pm_today"] = pm
    for k in (1, 2, 3, 7):
        d[f"pm_lag{k}"] = pm.shift(k)
    d["pm_roll3"] = pm.rolling(3).mean()
    d["pm_roll7"] = pm.rolling(7).mean()
    d["pm_roll7_std"] = pm.rolling(7).std()
    d["temp_change1"] = d["temperature"].diff()
    d["wind_change1"] = d["wind"].diff()
    # day-of-year as a circle, so 31 Dec and 1 Jan are neighbours
    doy = d["time"].dt.dayofyear
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return d


def build_latest_features(air_hourly, weather_hourly, min_hours=20, today=None, max_age_days=1):
    """For the app: hourly frames -> (all feature rows, the newest usable row or None).

    Returns None for the row when the newest day is incomplete OR older than
    `max_age_days` (so the app never forecasts from stale data).

    air_hourly:     columns time, pm2_5
    weather_hourly: columns time, temperature, humidity, wind
    """
    a = hourly_to_daily(air_hourly, min_hours)
    w = hourly_to_daily(weather_hourly, min_hours)
    daily = a.merge(w, on="time", how="inner")
    daily, _ = fill_small_gaps(daily, ["pm2_5", "temperature", "humidity", "wind"])
    d = add_features(daily)
    ok = d.dropna(subset=FEATURES)
    if ok.empty:
        return d, None
    if today is None:
        today = pd.Timestamp.now(tz="Asia/Karachi").tz_localize(None).normalize()
    last = ok["time"].iloc[-1]
    if last != d["time"].iloc[-1] or (pd.Timestamp(today) - last).days > max_age_days:
        return d, None
    return d, ok.iloc[[-1]]


# ----------------------------------------------------------------------- model
class DeltaRegressor(BaseEstimator, RegressorMixin):
    """Learns the CHANGE from today to tomorrow, then adds today's value back.

    Why: tree models cannot predict above the largest value they saw in training.
    Predicting the change lets them output 'today + change' even in a record smog.
    """

    def __init__(self, base=None, col="pm_today"):
        self.base = base
        self.col = col

    def fit(self, X, y):
        self.model_ = clone(self.base).fit(X, np.asarray(y) - X[self.col].values)
        return self

    def predict(self, X):
        return np.clip(X[self.col].values + self.model_.predict(X), 0, None)
