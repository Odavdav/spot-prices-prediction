"""Walk-forward backtest and metrics.

Protocol: for each refit date of the evaluated period (by default the 1st of each month), the model
is trained on every delivery before that date (expanding window), then forecasts every day until
the next refit date. Weekly and daily retraining are variants (REFIT_VARIANTS).

Metrics: MAE and rMAE (MAE / MAE of the naive D-7 forecast on the same hours), by regime.

Usage:
    python -m src.backtest                    # validation (Nov. 2024 -> Oct. 2025)
    python -m src.backtest --period test      # test: only run at the end of the project
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.benchmarks import BENCHMARKS
from src.config import PERIODS, PREDICTIONS_DIR, RESULTS, SEED, TZ
from src.data import load_dataset
from src.features import STEPS, add_features, get_X, get_y, to_price

# Fixed hyperparameters, reasonable for ~50,000 rows and ~30 features (no tuning)
LGB_PARAMS = dict(
    objective="l1",          # consistent with the MAE; robust to price spikes
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=SEED,
    verbose=-1,
)

NAIVE = {"naive_d1": "price_d1", "naive_d7": "price_d7"}
EVENING_PEAK = range(18, 22)  # 18:00-21:59, local time


# --------------------------------------------------------------------- walk-forward
REFIT_FREQ = {"monthly": "MS", "weekly": "7D", "daily": "D"}
REFIT_SUFFIX = {"weekly": "_weekly", "daily": "_daily"}
# LightGBM variants: name -> (step, retraining pace, half-life of the recency weights in days)
REFIT_VARIANTS = {"A1_weekly": ("A1", "weekly", None), "A2_weekly": ("A2", "weekly", None),
                  "B1_weekly": ("B1", "weekly", None), "B2_weekly": ("B2", "weekly", None),
                  "B2_daily": ("B2", "daily", None), "B2_rl_weekly": ("B2_rl", "weekly", None),
                  "B2_raw_weekly": ("B2_raw", "weekly", None)}
# Recency weights: a training row H days old weighs half as much (H chosen on validation)
HALF_LIVES = (90, 180, 365)
REFIT_VARIANTS.update({f"B2_weekly_hl{h}": ("B2", "weekly", h) for h in HALF_LIVES})
# B3 = B2 + shape of the day + current regime
REFIT_VARIANTS.update({"B3_weekly": ("B3", "weekly", None), "B3_weekly_hl90": ("B3", "weekly", 90),
                       "B3_daily": ("B3", "daily", None), "B3_daily_hl90": ("B3", "daily", 90)})


def refit_dates(start: str, end: str, refit: str = "monthly") -> list:
    """Retraining dates: the 1st of each month, every 7 days, or every day from `start`."""
    return list(pd.date_range(start, end, freq=REFIT_FREQ[refit], tz=TZ, inclusive="left"))


def recency_weights(times: pd.Series, refit_date: pd.Timestamp, half_life: float) -> np.ndarray:
    """Weights 0.5^(age / half-life), the age being counted in days before the retraining date."""
    age_days = (refit_date - times).dt.total_seconds().to_numpy() / 86400
    return 0.5 ** (age_days / half_life)


def walk_forward(f: pd.DataFrame, step: str, start: str, end: str, refit: str = "monthly",
                 half_life: float | None = None) -> pd.Series:
    """Forecasts (in €/MWh) of step `step` on [start, end), retrained at pace `refit`,
    with recency weights if `half_life` is given."""
    y = get_y(f, step)
    X = get_X(f, step)
    dates = refit_dates(start, end, refit) + [pd.Timestamp(end, tz=TZ)]
    preds = []
    for m, m_end in zip(dates[:-1], dates[1:]):
        train = (f["time_local"] < m) & y.notna()
        pred = (f["time_local"] >= m) & (f["time_local"] < m_end)
        w = recency_weights(f.loc[train, "time_local"], m, half_life) if half_life else None
        model = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[train], y[train], sample_weight=w)
        p = to_price(model.predict(X[pred]), f[pred], step)
        preds.append(pd.Series(p, index=f.index[pred]))
        if refit != "daily" or m.day == 1:
            print(f"  {step} ({refit}) {m:%Y-%m-%d}  train={train.sum():6d}  pred={pred.sum():4d}")
    return pd.concat(preds).rename(step + REFIT_SUFFIX.get(refit, ""))


# --------------------------------------------------------------------- Diebold-Mariano test
# Compared pairs (A, B): chosen in advance to limit multiple testing.
DM_PAIRS = [
    ("B2", "naive_d7"),             # does the full model beat the benchmark?
    ("B2", "B2_raw"),               # does the SRMC normalisation change anything?
    ("B2", "LEAR_daily"),           # best LightGBM vs literature benchmark
    ("B1", "LEAR_daily"),           # same information
    ("B1_weekly", "LEAR_weekly"),   # same information and same retraining pace
    ("B2_weekly", "B2"),            # retraining weekly rather than monthly
    ("B2_weekly", "LEAR_daily"),    # best LightGBM vs best LEAR (added after the weekly runs)
    ("B2_daily", "B2_weekly"),      # does daily retraining still help? (fixed before the run)
    ("B2_daily", "LEAR_daily"),     # same pace as the LEAR of the paper (fixed before the run)
    ("B2_rl_weekly", "B2_weekly"),  # residual-load-weighted normalisation vs full
    ("B2_rl_weekly", "B2_raw_weekly"),  # ... vs no normalisation
    ("B2_rl", "B2"),                # same at the monthly pace (crisis period)
    ("B2_rl", "B2_raw"),
    # Post-test additions (developed and chosen on validation): mimic Chronos-2's adaptation
    ("B2_weekly_hl90", "B2_weekly"),  # recency weights, 90-day half-life
    ("B2_weekly_hl180", "B2_weekly"),
    ("B2_weekly_hl365", "B2_weekly"),
    ("B2_weekly_corr", "B2_weekly"),  # correction by recent errors
    ("B2_weekly_hl90_corr", "B2_weekly"),       # both combined (chosen half-life: 90 days)
    ("B2_weekly_hl90_corr", "B2_weekly_hl90"),  # does the correction add to the recency weights?
    # Fixed before running the variants chosen on validation on the test year
    ("B2_daily_corr", "B2_daily"),
    ("B2_daily_corr", "Chronos2_fund"),
    ("B2_weekly_hl90_corr", "Chronos2_fund"),
    # B3: shape of the day and current regime (fixed before the runs)
    ("B3_weekly", "B2_weekly"),
    ("B3_weekly_corr", "B2_weekly_corr"),
    ("B3_weekly_hl90_corr", "B2_weekly_hl90_corr"),
    ("B3_weekly_corr", "Chronos2_fund"),
    ("B3_weekly_hl90_corr", "Chronos2_fund"),
    # Daily versions (fixed before the runs)
    ("B3_daily_corr", "B2_daily"),
    ("B3_daily_hl90_corr", "B2_daily"),
    ("B3_daily_corr", "Chronos2_fund"),
    ("B3_daily_hl90_corr", "Chronos2_fund"),
    ("Chronos2_fund", "Chronos2_prices"),  # do fundamentals help a foundation model?
    ("B2_weekly", "Chronos2_fund"),        # LightGBM vs foundation model, same information
    ("LEAR_daily", "LEAR_monthly"),
    # LEAR with the B2 information (fixed before the runs)
    # Weekly ablation, one block at a time (fixed before the A1/A2 weekly runs)
    ("A2_weekly", "A1_weekly"),
    ("B1_weekly", "A2_weekly"),
    ("B2_weekly", "B1_weekly"),
    ("B2_daily", "LEAR_B2_daily"),  # same information and same pace: model effect only
    ("LEAR_B2_daily", "LEAR_daily"),  # do gas, CO2 and Germany help a linear model?
    ("B2", "MSTL"),
]


def diebold_mariano(err_a: pd.Series, err_b: pd.Series, day: pd.Series) -> dict:
    """Diebold-Mariano test on the absolute loss, with a HAC (Newey-West) variance.

    As in Lago et al. (2021), the loss differential is first averaged by day (the 24 hours of a day
    are strongly correlated), then the autocorrelation between days is handled by Newey-West.
    t < 0: A has a lower error than B.
    """
    d = (err_a.abs() - err_b.abs()).groupby(day.to_numpy()).mean().dropna().to_numpy()
    T = len(d)
    lags = int(np.floor(4 * (T / 100) ** (2 / 9)))      # usual Newey-West rule
    u = d - d.mean()
    lrv = u @ u / T
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / T
    if lrv <= 0:   # constant differential (e.g. two identical forecasts)
        return {"MAE_gap": d.mean(), "t_HAC": np.nan, "p_value": np.nan, "days": T, "lags_NW": lags}
    t = d.mean() / np.sqrt(lrv / T)
    p = 2 * (1 - _normal_cdf(abs(t)))
    return {"MAE_gap": d.mean(), "t_HAC": t, "p_value": p, "days": T, "lags_NW": lags}


def _normal_cdf(x: float) -> float:
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def dm_table(preds: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    rows = f.loc[preds.index]
    day = rows["time_local"].dt.strftime("%Y-%m-%d")
    out = {}
    for a, b in DM_PAIRS:
        if a in preds and b in preds:
            ok = preds[[a, b]].notna().all(axis=1) & rows["price_da"].notna()
            err = preds.loc[ok, [a, b]].sub(rows.loc[ok, "price_da"], axis=0)
            out[f"{a} vs {b}"] = diebold_mariano(err[a], err[b], day[ok])
    return pd.DataFrame(out).T


# --------------------------------------------------------------------- metrics
def regime(f: pd.DataFrame) -> pd.Series:
    """Regime of each hour: negative price, evening peak, other (on the actual price)."""
    r = pd.Series("other", index=f.index)
    r[f["time_local"].dt.hour.isin(EVENING_PEAK)] = "evening_peak"
    r[f["price_da"] < 0] = "negative"   # takes priority: a negative peak hour stays negative
    return r


def metrics(preds: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """MAE and rMAE (vs naive D-7) by model, on all hours and by regime.

    Only the hours where every model has a forecast are counted, so that models are compared on
    the same hours.
    """
    rows = f.loc[preds.index]
    ok = preds.notna().all(axis=1) & rows["price_da"].notna()
    err = preds[ok].sub(rows.loc[ok, "price_da"], axis=0).abs()
    reg = regime(rows[ok])
    out = {}
    for name, mask in [("all", slice(None))] + [(r, reg == r) for r in ("negative", "evening_peak", "other")]:
        e = err[mask]
        out[(name, "MAE")] = e.mean()
        out[(name, "rMAE")] = e.mean() / e["naive_d7"].mean()
        out[(name, "hours")] = pd.Series(len(e), index=e.columns)
    table = pd.DataFrame(out)
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    return table


# --------------------------------------------------------------------- main
def load_predictions(period: str) -> pd.DataFrame:
    """Read back the forecasts already computed: one file per model, so that several runs in
    parallel do not overwrite each other's results."""
    folder = PREDICTIONS_DIR / period
    files = sorted(folder.glob("*.parquet")) if folder.exists() else []
    return pd.concat([pd.read_parquet(p).iloc[:, 0].rename(p.stem) for p in files], axis=1) if files else None


def save_prediction(period: str, pred: pd.Series) -> None:
    folder = PREDICTIONS_DIR / period
    folder.mkdir(parents=True, exist_ok=True)
    pred.to_frame().to_parquet(folder / f"{pred.name}.parquet")


def main():
    models = list(STEPS) + list(REFIT_VARIANTS) + list(BENCHMARKS)
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=list(PERIODS), default="validation")
    ap.add_argument("--models", nargs="*", choices=models, default=models,
                    help="models to (re)compute; the others are read from the saved forecasts "
                         "(--models with no argument: only recompute the tables)")
    args = ap.parse_args()
    if args.period == "test":
        print("WARNING: test period. Only run it once the choices are frozen on validation.")

    start, end = PERIODS[args.period]
    f = add_features(load_dataset())
    in_period = (f["time_local"] >= pd.Timestamp(start, tz=TZ)) & (f["time_local"] < pd.Timestamp(end, tz=TZ))

    for name in args.models:
        if name in BENCHMARKS:
            pred = BENCHMARKS[name](f, start, end)
        elif name in STEPS:
            pred = walk_forward(f, name, start, end)
        else:
            step, refit, half_life = REFIT_VARIANTS[name]
            pred = walk_forward(f, step, start, end, refit=refit, half_life=half_life)
        save_prediction(args.period, pred.rename(name))   # saved at once: a crash loses nothing

    # Tables on everything available (including models from other runs)
    preds = pd.DataFrame({name: f.loc[in_period, col] for name, col in NAIVE.items()})
    saved = load_predictions(args.period)
    if saved is not None:
        preds = preds.join(saved, how="left")
    preds = preds[list(NAIVE) + [m for m in models if m in preds]
                  + [m for m in preds if m not in models and m not in NAIVE]]

    table = metrics(preds, f)
    RESULTS.mkdir(exist_ok=True)
    table.round(3).to_csv(RESULTS / f"metrics_{args.period}.csv")
    dm = dm_table(preds, f)
    dm.round(4).to_csv(RESULTS / f"diebold_mariano_{args.period}.csv")

    with pd.option_context("display.width", 140, "display.float_format", "{:.2f}".format):
        print(f"\n=== {args.period}: {start} -> {end} ===")
        print(table.drop(columns="hours", level=1))
        print("hours by regime:", table.xs("hours", axis=1, level=1).iloc[0].astype(int).to_dict())
        print("\nDiebold-Mariano (absolute loss, daily mean, HAC Newey-West):")
        print(dm)


if __name__ == "__main__":
    main()
