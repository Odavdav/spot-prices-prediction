"""Correction of a model by its recent errors (second stage).

Idea (mimicking Chronos-2's adaptation): the base model makes its out-of-sample forecasts;
a small model learns to predict the error of day D from recent errors:
    - error of D-1 at the same hour;
    - mean error of D-1;
    - mean error of the last 7 days, at the same hour and over all hours.
Corrected forecast = base forecast − predicted error.

Information set: the error of D-1 = forecast of D-1 (made at 12:00 on D-2) − price of D-1
(published around 13:00 on D-2). It is therefore known at 12:00 on D-1, before gate closure for D.
The correction model is a median regression (consistent with the MAE), re-estimated every week
on all the available error history (expanding window).

Usage:
    python -m src.correction --base B2_weekly --period validation --history burn_in
    python -m src.correction --base B2_daily --period test --history validation
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from src.backtest import PREDICTIONS_DIR, save_prediction
from src.config import PERIODS, TZ
from src.data import load_dataset

FEATURES = ["err_d1", "err_d1_mean", "err_7d_hour", "err_7d_mean"]
MIN_HISTORY_DAYS = 90   # no correction while the error history is too short


def _day_hour(time_local: pd.Series):
    t = time_local.dt.tz_convert(TZ)
    return t.dt.tz_localize(None).dt.normalize(), t.dt.hour


def error_features(error: pd.Series, time_local: pd.Series) -> pd.DataFrame:
    """(day, hour) table: error of the day (target) and features of past errors (D-1 to D-7)."""
    day, hour = _day_hour(time_local)
    panel = error.groupby([day, hour]).mean().unstack()            # days × local hours
    panel = panel.reindex(pd.date_range(panel.index.min(), panel.index.max(), freq="D"))
    past = [panel.shift(k) for k in range(1, 8)]                   # errors of D-1 … D-7
    feats = {
        "err_d1": past[0],
        "err_d1_mean": pd.DataFrame(np.repeat(past[0].mean(axis=1).to_numpy()[:, None], panel.shape[1], 1),
                                    index=panel.index, columns=panel.columns),
        "err_7d_hour": sum(past) / 7,
        "err_7d_mean": pd.DataFrame(np.repeat(pd.concat(past, axis=1).mean(axis=1).to_numpy()[:, None],
                                              panel.shape[1], 1), index=panel.index, columns=panel.columns),
        "target": panel,
    }
    long = pd.concat({k: v.stack(future_stack=True) for k, v in feats.items()}, axis=1)
    long.index.names = ["day", "hour"]
    return long


def error_correction(base: pd.Series, price: pd.Series, time_local: pd.Series,
                     start: str, end: str) -> pd.Series:
    """Corrected forecast on [start, end). `base`, `price`, `time_local` cover the history and the period.

    Only the prices of days < D are used to correct day D (through past errors)."""
    error = base - price
    table = error_features(error, time_local)
    days = pd.date_range(start, end, freq="D", inclusive="left")
    fit_dates = list(pd.date_range(start, end, freq="7D", inclusive="left")) + [pd.Timestamp(end)]
    day_idx = table.index.get_level_values("day")
    corr = pd.Series(0.0, index=table.index)
    for a, b in zip(fit_dates[:-1], fit_dates[1:]):
        hist = table[(day_idx < a)].dropna()
        if hist.index.get_level_values("day").nunique() < MIN_HISTORY_DAYS:
            continue
        model = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(hist[FEATURES], hist["target"])
        rows = table[(day_idx >= a) & (day_idx < b)]
        ok = rows[FEATURES].notna().all(axis=1)
        corr.loc[rows.index[ok]] = model.predict(rows.loc[ok, FEATURES])
    # Back onto the hourly index
    day, hour = _day_hour(time_local)
    in_period = (day >= days[0]) & (day <= days[-1])
    key = pd.MultiIndex.from_arrays([day[in_period], hour[in_period]])
    est_error = corr.reindex(key).fillna(0.0).to_numpy()
    return (base[in_period] - est_error).rename(f"{base.name}_corr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--period", choices=list(PERIODS), required=True)
    ap.add_argument("--history", nargs="*", default=[], help="periods whose forecasts serve as history")
    ap.add_argument("--name", default=None, help="output name (default: <base>_corr)")
    args = ap.parse_args()

    parts = [pd.read_parquet(PREDICTIONS_DIR / p / f"{args.base}.parquet").iloc[:, 0]
             for p in args.history + [args.period]]
    base = pd.concat(parts).sort_index()
    base = base[~base.index.duplicated(keep="last")].rename(args.base)
    df = load_dataset().loc[base.index]
    out = error_correction(base, df["price_da"], df["time_local"], *PERIODS[args.period])
    if args.name:
        out = out.rename(args.name)
    save_prediction(args.period, out)
    real = df.loc[out.index, "price_da"]
    print(f"{args.base}: MAE {(base.loc[out.index] - real).abs().mean():.2f} -> "
          f"{out.name}: {(out - real).abs().mean():.2f} €/MWh")


if __name__ == "__main__":
    main()
