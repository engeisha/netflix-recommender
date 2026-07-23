"""
Task 5 – Netflix Trend Forecasting
====================================
Aggregates monthly Netflix additions, engineers lag/rolling features,
trains a GradientBoosting regressor per series (overall, Movie, TV Show,
and top genres), evaluates on a held-out test split, and forecasts
12-24 months into the future.

Saved artifacts
---------------
  forecasting_model.pkl  – dict of fitted models keyed by series name
  forecasting_data.pkl   – dict of DataFrames and metadata for the UI
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent
DATA_PATH  = ROOT / "netflix_titles.csv"
MODEL_OUT  = ROOT / "forecasting_model.pkl"
DATA_OUT   = ROOT / "forecasting_data.pkl"

# Only use data from this year onward (sparse before)
SERIES_START = "2016-01"
TOP_GENRES   = [
    "International Movies", "Dramas", "Comedies",
    "International TV Shows", "Documentaries",
    "Action & Adventure", "TV Dramas", "Independent Movies",
]
LAGS         = [1, 2, 3, 6, 12]
ROLLING_WINS = [3, 6, 12]


# ── helpers ────────────────────────────────────────────────────────────────────
def make_features(series: pd.Series) -> pd.DataFrame:
    """
    Given a monthly time-series (PeriodIndex), build a supervised
    feature DataFrame with lag and rolling features.
    Returns (X, y) aligned DataFrame.
    """
    df = pd.DataFrame({"y": series.values}, index=series.index)
    df.index = df.index.to_timestamp()

    for lag in LAGS:
        df[f"lag_{lag}"] = df["y"].shift(lag)
    for win in ROLLING_WINS:
        df[f"roll_mean_{win}"] = df["y"].shift(1).rolling(win).mean()
        df[f"roll_std_{win}"]  = df["y"].shift(1).rolling(win).std()

    df["month"]       = df.index.month
    df["month_sin"]   = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df.index.month / 12)
    df["time_idx"]    = np.arange(len(df))

    df = df.dropna()
    return df


def train_series(series: pd.Series, test_months: int = 12):
    """
    Train / evaluate a GradientBoosting model on one monthly series.
    Returns (model, metrics_dict, train_df, test_df).
    """
    df = make_features(series)
    if len(df) < test_months + 10:
        return None, {}, None, None

    feat_cols = [c for c in df.columns if c != "y"]
    X, y = df[feat_cols], df["y"]

    split = len(df) - test_months
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    )
    model.fit(X_tr, y_tr)

    preds = np.maximum(model.predict(X_te), 0)
    mae   = mean_absolute_error(y_te, preds)
    rmse  = float(np.sqrt(mean_squared_error(y_te, preds)))
    mask  = y_te > 0
    mape  = float(np.mean(np.abs((y_te[mask] - preds[mask]) / y_te[mask])) * 100) if mask.any() else np.nan

    metrics = {"mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}

    train_out = pd.DataFrame({"actual": y_tr, "predicted": np.maximum(model.predict(X_tr), 0)}, index=X_tr.index)
    test_out  = pd.DataFrame({"actual": y_te.values, "predicted": preds}, index=X_te.index)

    return model, metrics, train_out, test_out


def forecast_future(model, series: pd.Series, horizon: int = 24) -> pd.DataFrame:
    """
    Iteratively forecast `horizon` months beyond the last known date.
    Builds each new row from the growing history.
    """
    history = series.copy()
    last_ts = history.index[-1].to_timestamp()
    preds   = []
    dates   = []

    for step in range(1, horizon + 1):
        next_ts = last_ts + pd.DateOffset(months=step)
        tmp     = history.copy()
        # append a placeholder so rolling features align
        tmp_ts  = tmp.copy()
        tmp_ts.index = tmp_ts.index.to_timestamp()

        row: dict = {}
        for lag in LAGS:
            row[f"lag_{lag}"] = float(tmp_ts.iloc[-lag]) if lag <= len(tmp_ts) else 0.0
        for win in ROLLING_WINS:
            tail = tmp_ts.iloc[-win:] if len(tmp_ts) >= win else tmp_ts
            row[f"roll_mean_{win}"] = float(tail.mean())
            row[f"roll_std_{win}"]  = float(tail.std()) if len(tail) > 1 else 0.0

        row["month"]     = next_ts.month
        row["month_sin"] = np.sin(2 * np.pi * next_ts.month / 12)
        row["month_cos"] = np.cos(2 * np.pi * next_ts.month / 12)
        row["time_idx"]  = len(tmp_ts) + step - 1

        feat_cols = [c for c in make_features(series).columns if c != "y"]
        X_new = pd.DataFrame([row])[feat_cols]
        pred  = float(max(model.predict(X_new)[0], 0))
        preds.append(pred)
        dates.append(next_ts)

        # append prediction to rolling history
        new_period = pd.Period(next_ts, freq="M")
        history    = pd.concat([history, pd.Series([pred], index=pd.PeriodIndex([new_period]))])

    return pd.DataFrame({"forecast": preds}, index=pd.DatetimeIndex(dates))


# ── data loading ───────────────────────────────────────────────────────────────
def build_monthly_series(path: Path = DATA_PATH):
    df = pd.read_csv(path)
    df["date_added"] = pd.to_datetime(df["date_added"].str.strip(),
                                      format="%B %d, %Y", errors="coerce")
    df = df.dropna(subset=["date_added"])
    df["ym"] = df["date_added"].dt.to_period("M")

    # full monthly index (fill gaps with 0)
    all_months = pd.period_range(start=SERIES_START, end=df["ym"].max(), freq="M")

    def make_series(mask=None) -> pd.Series:
        sub   = df if mask is None else df[mask]
        cnts  = sub.groupby("ym").size()
        return cnts.reindex(all_months, fill_value=0)

    series: dict[str, pd.Series] = {
        "Overall":  make_series(),
        "Movie":    make_series(df["type"] == "Movie"),
        "TV Show":  make_series(df["type"] == "TV Show"),
    }
    for genre in TOP_GENRES:
        mask = df["listed_in"].str.contains(genre, na=False)
        series[genre] = make_series(mask)

    return series, df


# ── main ───────────────────────────────────────────────────────────────────────
def run():
    print("Loading data …")
    series_dict, raw_df = build_monthly_series()
    print(f"  Series: {list(series_dict.keys())}")
    print(f"  Length: {len(series_dict['Overall'])} months  "
          f"({SERIES_START} → {series_dict['Overall'].index[-1]})")

    models:   dict = {}
    metrics:  dict = {}
    history:  dict = {}   # train+test actuals + fitted values
    forecasts: dict = {}  # future 24-month forecasts

    for name, s in series_dict.items():
        print(f"  Training '{name}' …", end=" ")
        model, met, tr, te = train_series(s, test_months=12)
        if model is None:
            print("skipped (too short)")
            continue
        models[name]   = model
        metrics[name]  = met
        history[name]  = {"train": tr, "test": te, "full_series": s}
        forecasts[name] = forecast_future(model, s, horizon=24)
        print(f"MAE={met['mae']:.1f}  RMSE={met['rmse']:.1f}  MAPE={met['mape']:.1f}%")

    # genre growth: compare first-half vs second-half of known data
    genre_growth = {}
    for genre in TOP_GENRES:
        if genre not in history:
            continue
        s   = history[genre]["full_series"]
        mid = len(s) // 2
        avg_first  = float(s.iloc[:mid].mean())
        avg_second = float(s.iloc[mid:].mean())
        pct = ((avg_second - avg_first) / (avg_first + 1e-9)) * 100
        genre_growth[genre] = round(pct, 1)

    payload = {
        "history":      history,
        "forecasts":    forecasts,
        "metrics":      metrics,
        "genre_growth": genre_growth,
        "series_start": SERIES_START,
        "top_genres":   TOP_GENRES,
        "raw_monthly":  {
            name: h["full_series"].to_timestamp().reset_index()
                    .rename(columns={"index": "date", 0: "count"})
            for name, h in history.items()
        },
    }

    joblib.dump(models,  MODEL_OUT)
    joblib.dump(payload, DATA_OUT)
    print(f"\nSaved → {MODEL_OUT}")
    print(f"Saved → {DATA_OUT}")
    print("Done.")


if __name__ == "__main__":
    run()
