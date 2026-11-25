"""Benchmarks: LEAR (linear benchmark from the literature) and Chronos-2.

The naive D-1 and D-7 forecasts are simply the features price_d1 and price_d7 (see backtest.py).

LEAR — Lago, Marcjasz, De Schutter and Weron (2021), Applied Energy 293:116983. Penalised linear
regression (LASSO, penalty chosen by AIC), one model per delivery hour. Inputs for day D: prices of
the 24 hours of D-1, D-2, D-3 and D-7, French load and renewables forecasts for D, D-1 and D-7, day
of the week. Prices and inputs are transformed by asinh after robust scaling (median, MAD).
Forecast = mean of the models calibrated on 56, 84, 1092 and 1456 days.
Same information as step B1 (calendar, price lags, FR fundamentals), without gas or DE.

LEAR-B2 — same model with the information of step B2 added: German price of the 24 hours of D-1,
German load and wind + solar forecasts for D, and the CCGT SRMC for D (TTF and EUA closes from D-2).
The target stays the raw price; the SRMC is an input. Re-estimated daily.

Estimation follows the reference code of the paper (epftoolbox, `LEAR.recalibrate`), which ran on
scikit-learn <= 0.24:
- a window of N days holds N - 7 training days (its first week only provides the D-7 lags);
- the penalty is chosen by `LassoLarsIC(criterion="aic")` with that version's defaults: inputs
  normalised to unit norm, and the noise variance in the AIC taken as the variance of the target
  (later versions estimate it by OLS, which fails when days < inputs, as in the 56- and 84-day
  windows; it is therefore passed explicitly);
- a `Lasso` with that penalty is then fitted on the (asinh-scaled, not normalised) inputs.
`tests/test_lear_reference.py` checks that this reproduces the reference code run under
scikit-learn 0.24.2. Forecasts are not clipped, as in the paper.

Departure from the paper: monthly and weekly versions in addition to the daily one, to compare with
LightGBM at the same re-estimation pace.
"""
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoLarsIC

from src.config import TZ

LEAR_WINDOWS = (1456, 1092, 84, 56)
LEAR_LAG_DAYS = 7     # first week of each window: lags only (as in epftoolbox)


# --------------------------------------------------------------------- helpers
def _local_day_hour(f: pd.DataFrame):
    t = f["time_local"].dt.tz_convert(TZ)
    return t.dt.tz_localize(None).dt.normalize(), t.dt.hour


def _daily_panel(values: pd.Series, day: pd.Series, hour: pd.Series, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Days × 24 local hours table. The missing hour (spring DST change) is interpolated within
    the day, the doubled hour (October) averaged."""
    panel = values.groupby([day, hour]).mean().unstack().reindex(index=days, columns=range(24))
    return panel.interpolate(axis=1, limit_direction="both")


def _to_hourly(panel: pd.DataFrame, day: pd.Series, hour: pd.Series) -> np.ndarray:
    """Map a days × hours table back onto the hourly index."""
    stacked = panel.stack()
    return stacked.reindex(pd.MultiIndex.from_arrays([day, hour])).to_numpy()


def _period_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="D", inclusive="left")


# --------------------------------------------------------------------- LEAR
def _lear_design(f: pd.DataFrame, extended: bool = False):
    """LEAR input matrix (one row per delivery day) and days × 24 hours target.

    `extended`: add the B2 information (German price D-1, German load and renewables for D, SRMC)."""
    day, hour = _local_day_hour(f)
    days = pd.date_range(day.min(), day.max(), freq="D")
    price = _daily_panel(f["price_da"], day, hour, days)
    load = _daily_panel(f["fr_load_fc"], day, hour, days)
    vre = _daily_panel(f["fr_wind_fc"] + f["fr_solar_fc"], day, hour, days)

    blocks = {}
    for k in (1, 2, 3, 7):
        blocks[f"price_d{k}"] = price.shift(k)
    for name, panel in (("load", load), ("vre", vre)):
        for k in (0, 1, 7):
            blocks[f"{name}_d{k}"] = panel.shift(k)
    if extended:
        blocks["de_price_d1"] = _daily_panel(f["de_price_da"], day, hour, days).shift(1)
        blocks["de_load_d0"] = _daily_panel(f["de_load_fc"], day, hour, days)
        blocks["de_vre_d0"] = _daily_panel(f["de_wind_fc"] + f["de_solar_fc"], day, hour, days)
    X = pd.concat(blocks, axis=1)
    X.columns = [f"{b}_h{h}" for b, h in X.columns]
    if extended:   # the SRMC is constant within a day: one input
        X["srmc_ccgt"] = f["srmc_ccgt"].groupby(day.to_numpy()).mean().reindex(days)
    dummies = pd.get_dummies(days.dayofweek, prefix="dow").astype(float)
    dummies.index = days
    X = X.ffill()   # forecast gaps: carry the previous day at the same hour
    return X, dummies, price, day, hour


def _median_mad(a: np.ndarray):
    """Column medians and MADs (normal-consistent, as statsmodels' `mad`); a zero MAD is set to 1."""
    med = np.median(a, axis=0)
    mad = np.median(np.abs(a - med), axis=0) / 0.6744897501960817
    return med, np.where(mad > 0, mad, 1.0)


def lear_one_window(Xt: np.ndarray, Dt: np.ndarray, Yt: np.ndarray,
                    Xp: np.ndarray, Dp: np.ndarray) -> np.ndarray:
    """One LEAR calibration (24 hourly models) as in epftoolbox's `LEAR.recalibrate` and `predict`.

    Xt, Yt: inputs and prices (days × 24) of the training days; Dt, Dp: weekday dummies (not
    scaled); Xp: inputs of the days to forecast. Returns the forecasts (days × 24), in €/MWh."""
    my, sy = _median_mad(Yt)
    mx, sx = _median_mad(Xt)
    Ys = np.arcsinh((Yt - my) / sy)
    Xt = np.hstack([np.arcsinh((Xt - mx) / sx), Dt])
    Xp = np.hstack([np.arcsinh((Xp - mx) / sx), Dp])
    centred = Xt - Xt.mean(axis=0)
    norm = np.sqrt((centred ** 2).sum(axis=0))
    Xn = centred / np.where(norm > 0, norm, 1.0)     # normalize=True of scikit-learn <= 0.24
    out = np.empty((len(Xp), 24))
    for h in range(24):
        y = Ys[:, h]
        alpha = LassoLarsIC(criterion="aic", max_iter=2500, noise_variance=np.var(y)).fit(Xn, y).alpha_
        out[:, h] = Lasso(alpha=alpha, max_iter=2500).fit(Xt, y).predict(Xp)
    return np.sinh(out) * sy + my


def _lear_fit_predict(X, dummies, price, fit_day, predict_days):
    """Calibrate the 4 windows × 24 hours on days < fit_day, forecast `predict_days` (days × 24)."""
    valid = X.notna().all(axis=1) & price.notna().all(axis=1)
    hist = valid.index[(valid.index < fit_day) & valid.to_numpy()]
    preds = []
    for w in LEAR_WINDOWS:
        tr = hist[-(w - LEAR_LAG_DAYS):]
        preds.append(lear_one_window(X.loc[tr].to_numpy(), dummies.loc[tr].to_numpy(),
                                     price.loc[tr].to_numpy(), X.loc[predict_days].to_numpy(),
                                     dummies.loc[predict_days].to_numpy()))
    return pd.DataFrame(np.mean(preds, axis=0), index=predict_days, columns=range(24))


LEAR_NAMES = {"daily": "LEAR_daily", "weekly": "LEAR_weekly", "monthly": "LEAR_monthly"}


def lear_forecasts(f: pd.DataFrame, start: str, end: str, refit: str = "daily",
                   extended: bool = False) -> pd.Series:
    """LEAR forecasts on [start, end), re-estimated every day, week (7 days) or month."""
    X, dummies, price, day, hour = _lear_design(f, extended)
    days = _period_days(start, end)
    t0 = time.time()
    if refit == "daily":
        groups = [(d, pd.DatetimeIndex([d])) for d in days]
    else:
        freq = "7D" if refit == "weekly" else "MS"
        starts = list(pd.date_range(start, end, freq=freq, inclusive="left")) + [pd.Timestamp(end)]
        groups = [(a, days[(days >= a) & (days < b)]) for a, b in zip(starts[:-1], starts[1:])]
    panels = []
    for i, (fit_day, predict_days) in enumerate(groups, 1):
        panels.append(_lear_fit_predict(X, dummies, price, fit_day, predict_days))
        if refit == "daily" and i % 30 == 0:
            print(f"  LEAR{'_B2' if extended else ''} daily: {i}/{len(groups)} days ({time.time() - t0:.0f} s)")
    name = LEAR_NAMES[refit].replace("LEAR", "LEAR_B2") if extended else LEAR_NAMES[refit]
    print(f"  {name}: {len(groups)} re-estimations in {time.time() - t0:.0f} s")
    hourly = pd.Series(_to_hourly(pd.concat(panels), day, hour), index=f.index)
    return hourly.loc[hourly.notna()].rename(name)


# --------------------------------------------------------------------- Chronos-2
# Zero-shot foundation model (Amazon, technical report arXiv 2510.15821, 17 Oct. 2025).
# Released before the start of the test year (2025-11-01): it cannot have seen the test. It may have
# seen the validation year (Nov. 2024 -> Oct. 2025): it is therefore only run on the test year.
CHRONOS_MODEL = "amazon/chronos-2"
CHRONOS_CONTEXT = 2048     # hours of history (~12 weeks)
# Variables known in advance for D (past and future), as in step B2
CHRONOS_KNOWN = ["fr_load_fc", "fr_wind_fc", "fr_solar_fc", "de_load_fc", "de_wind_fc",
                 "de_solar_fc", "srmc_ccgt", "is_holiday"]
CHRONOS_PAST_ONLY = ["de_price_da"]   # only known until D-1

_chronos_pipeline = None


def _chronos():
    global _chronos_pipeline
    if _chronos_pipeline is None:
        from chronos import Chronos2Pipeline   # late import: PyTorch only when it is used
        _chronos_pipeline = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL, device_map="cpu")
    return _chronos_pipeline


def _fill_from_past(cov: pd.DataFrame) -> pd.DataFrame:
    """Fill covariate gaps with past values only (Chronos-2 does not accept missing covariates):
    the same hour one day earlier, then the last known value. The few leading gaps at the very start
    of the dataset (Oct. 2018, far outside every context window) are filled backwards."""
    return cov.fillna(cov.shift(24)).ffill().bfill().astype(float)


def chronos_forecasts(f: pd.DataFrame, start: str, end: str, covariates: bool) -> pd.Series:
    """Median forecast of Chronos-2 for each day of [start, end), from the last CHRONOS_CONTEXT
    hours known at the end of D-1 (and the variables known for D)."""
    name = "Chronos2_fund" if covariates else "Chronos2_prices"
    local = f["time_local"].dt.tz_convert(TZ)
    cols = CHRONOS_KNOWN + CHRONOS_PAST_ONLY
    cov = _fill_from_past(f[cols]) if covariates else None
    price = f["price_da"].to_numpy()
    tasks, targets = [], []
    for d in _period_days(start, end):
        day_start = pd.Timestamp(d, tz=TZ)
        hist = np.flatnonzero((local < day_start).to_numpy())[-CHRONOS_CONTEXT:]
        horizon = np.flatnonzero(((local >= day_start) & (local < day_start + pd.Timedelta(days=1))).to_numpy())
        task = {"target": price[hist]}
        if covariates:
            task["past_covariates"] = {c: cov[c].to_numpy()[hist] for c in cols}
            task["future_covariates"] = {c: cov[c].to_numpy()[horizon] for c in CHRONOS_KNOWN}
        tasks.append(task)
        targets.append(horizon)

    out = pd.Series(np.nan, index=f.index)
    t0 = time.time()
    for length in sorted({len(h) for h in targets}):   # 23, 24 or 25 h (DST changes)
        idx = [i for i, h in enumerate(targets) if len(h) == length]
        quantiles, _ = _chronos().predict_quantiles([tasks[i] for i in idx], prediction_length=length,
                                                    quantile_levels=[0.5], batch_size=32)
        for i, q in zip(idx, quantiles):
            out.iloc[targets[i]] = q[0, :, 0].numpy()
    print(f"  {name}: {len(tasks)} days in {time.time() - t0:.0f} s")
    return out.loc[out.notna()].rename(name)


BENCHMARKS = {
    "Chronos2_prices": lambda f, s, e: chronos_forecasts(f, s, e, covariates=False),
    "Chronos2_fund": lambda f, s, e: chronos_forecasts(f, s, e, covariates=True),
    "LEAR_monthly": lambda f, s, e: lear_forecasts(f, s, e, refit="monthly"),
    "LEAR_weekly": lambda f, s, e: lear_forecasts(f, s, e, refit="weekly"),
    "LEAR_daily": lambda f, s, e: lear_forecasts(f, s, e, refit="daily"),
    "LEAR_B2_daily": lambda f, s, e: lear_forecasts(f, s, e, refit="daily", extended=True),
}
