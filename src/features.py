"""Feature construction and publication register (leakage control).

Golden rule: for delivery day D, a feature must be known before SDAC gate closure, at 12:00
(Paris time) on D-1.

Two safeguards, checked by tests/test_no_leakage.py:
1. The register declares, for each raw column, when a value is published relative to its delivery
   day (RAW_PUBLICATION), and for each feature, which columns and which day lag it uses
   (FEATURE_REGISTRY). The test checks that every declaration complies with the rule.
2. The code is checked against the register: every value published after gate closure is erased
   (mask_unpublished) and the features of day D must not change.

All transformations are stateless (nothing is learnt from the data): they can be applied to the
whole dataset before splitting it into periods, without leakage.
"""
import holidays
import numpy as np
import pandas as pd

TZ = "Europe/Paris"
GATE_CLOSURE_HOUR = 12  # SDAC gate closure: 12:00 Paris time, on D-1

# --------------------------------------------------------------------- source register
# Raw column -> (days before delivery, local hour) at which a value delivered on day d is
# published. Example: a day-ahead price for day d is published around 12:45 on d-1.
RAW_PUBLICATION = {
    "price_da": (1, 13),     # SDAC results published around 12:45-13:00 on d-1
    "de_price_da": (1, 13),
    # D-1 load forecast: published at the latest 2 h before gate closure (EU reg. 543/2013, art. 6.1).
    "fr_load_fc": (1, 10),
    "de_load_fc": (1, 10),
    # D-1 wind/solar forecasts: ASSUMPTION. The regulatory deadline is 18:00 on D-1 (art. 14.1),
    # but TSOs (RTE, 50Hertz…) publish them before gate closure in practice, and ENTSO-E only keeps
    # the latest version. Standard practice in price forecasting (see README limitations).
    "fr_wind_fc": (1, 12),
    "fr_solar_fc": (1, 12),
    "de_wind_fc": (1, 12),
    "de_solar_fc": (1, 12),
    # Gas and carbon: already aligned in get_data on the last close dated D-2 or earlier.
    "ttf_d2": (2, 18),
    "eua_d2": (2, 18),
}

# Feature -> (raw columns used, lag in days of the delivery used)
# Lag 0: value attached to day D itself (D-1 forecast for D, TTF already aligned…).
FEATURE_REGISTRY = {
    # Calendar: known in advance, no source
    "hour": ([], 0),
    "dayofweek": ([], 0),
    "month": ([], 0),
    "is_holiday": ([], 0),
    "is_bridge": ([], 0),
    "is_dst_change": ([], 0),
    # French price lags
    "price_d1": (["price_da"], 1),
    "price_d2": (["price_da"], 2),
    "price_d7": (["price_da"], 7),
    "price_d1_mean": (["price_da"], 1),
    "price_d1_min": (["price_da"], 1),
    "price_d1_max": (["price_da"], 1),
    # French fundamentals (D-1 forecasts)
    "fr_load_fc": (["fr_load_fc"], 0),
    "fr_wind_fc": (["fr_wind_fc"], 0),
    "fr_solar_fc": (["fr_solar_fc"], 0),
    "fr_residual_fc": (["fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 0),
    "fr_vre_share": (["fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 0),
    # Fuels and neighbours
    "srmc_ccgt": (["ttf_d2", "eua_d2"], 0),
    "de_price_d1": (["de_price_da"], 1),
    "de_price_d1_mean": (["de_price_da"], 1),
    "de_load_fc": (["de_load_fc"], 0),
    "de_vre_fc": (["de_wind_fc", "de_solar_fc"], 0),
    "de_residual_fc": (["de_load_fc", "de_wind_fc", "de_solar_fc"], 0),
    # Shape of day D (computed on the 24 hours of forecasts for D, known before gate closure)
    "fr_rl_day_min": (["fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 0),
    "fr_rl_day_max": (["fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 0),
    "fr_rl_day_mean": (["fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 0),
    "fr_rl_rel": (["fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 0),
    "fr_rl_rank": (["fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 0),
    "fr_rl_min_hour": (["fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 0),
    "fr_solar_day_max": (["fr_solar_fc"], 0),
    "de_rl_day_mean": (["de_load_fc", "de_wind_fc", "de_solar_fc"], 0),
    "price_d1_rel": (["price_da"], 1),
    # Current regime (rolling windows ending on D-1)
    "spread_14d": (["price_da", "ttf_d2", "eua_d2"], 1),
    "vol_14d": (["price_da"], 1),
    "neg_share_14d": (["price_da"], 1),
    "rl_slope_28d": (["price_da", "fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 1),
}


def is_known_at_gate_closure(column: str, lag_days: int) -> bool:
    """Is a value of `column`, delivered `lag_days` days before D, published before 12:00 on D-1?"""
    days_before, hour = RAW_PUBLICATION[column]
    # Published on (D - lag_days - days_before) at `hour`; gate closure on (D - 1) at 12:00.
    offset = lag_days + days_before
    return offset > 1 or (offset == 1 and hour <= GATE_CLOSURE_HOUR)


# --------------------------------------------------------------------- ablation steps
CALENDAR = ["hour", "dayofweek", "month", "is_holiday", "is_bridge", "is_dst_change"]
PRICE_LAGS = ["price_d1", "price_d2", "price_d7", "price_d1_mean", "price_d1_min", "price_d1_max"]
FUNDAMENTALS = ["fr_load_fc", "fr_wind_fc", "fr_solar_fc", "fr_residual_fc", "fr_vre_share"]
FUELS_NEIGHBOURS = ["srmc_ccgt", "de_price_d1", "de_price_d1_mean",
                    "de_load_fc", "de_vre_fc", "de_residual_fc"]

STEPS = {
    "A1": CALENDAR,
    "A2": CALENDAR + PRICE_LAGS,
    "B1": CALENDAR + PRICE_LAGS + FUNDAMENTALS,
    "B2": CALENDAR + PRICE_LAGS + FUNDAMENTALS + FUELS_NEIGHBOURS,
}
# Post-test extension (inspired by Chronos-2): shape of the day and current regime
DAY_SHAPE = ["fr_rl_day_min", "fr_rl_day_max", "fr_rl_day_mean", "fr_rl_rel", "fr_rl_rank",
             "fr_rl_min_hour", "fr_solar_day_max", "de_rl_day_mean", "price_d1_rel"]
REGIME = ["spread_14d", "vol_14d", "neg_share_14d", "rl_slope_28d"]
STEPS["B3"] = STEPS["B2"] + DAY_SHAPE + REGIME
ALL_FEATURES = STEPS["B3"]
# Control variant: same features as B2, raw-price target (measures what the normalisation brings)
STEPS["B2_raw"] = STEPS["B2"]
# Variant: normalisation weighted by residual load (target price − w(RL) × SRMC)
STEPS["B2_rl"] = STEPS["B2"]

# Normalisation of the target and of €/MWh features, by step:
#   "full": price − SRMC;  "rl": price − w(RL) × SRMC;  absent: raw price.
NORMALIZATION = {"B2": "full", "B2_rl": "rl", "B3": "full"}
NORMALIZED_STEPS = set(NORMALIZATION)
PRICE_LEVEL_FEATURES = ["price_d1", "price_d2", "price_d7", "price_d1_mean", "price_d1_min",
                        "price_d1_max", "de_price_d1", "de_price_d1_mean"]

# Pass-through of the gas cost to the French price by forecast residual load (FR, MW):
# slope of the price on the SRMC by residual-load decile, with hour and month fixed effects,
# estimated on data before the validation year (Oct. 2018 -> Oct. 2024), hence without leakage.
# Reading: below ~35 GW gas rarely sets the price but still matters (0.7: coupling with
# neighbours, water value); above ~40 GW the pass-through is about 1.1-1.2.
PASS_THROUGH_RL = [28261, 33174, 36055, 38614, 41098, 43760, 47089, 51881, 58135, 68107]
PASS_THROUGH = [0.71, 0.86, 0.95, 1.04, 1.13, 1.20, 1.18, 1.05, 1.18, 1.47]

# SRMC of a CCGT, in €/MWh of electricity
CCGT_EFFICIENCY = 0.55   # electrical efficiency
CCGT_EMISSIONS = 0.37    # tCO2 per MWh of electricity
CCGT_VOM = 2.0           # variable costs other than fuel, €/MWh


# --------------------------------------------------------------------- calendar
def _calendar_by_day(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily calendar features (public holidays, bridge days, DST changes).

    `days`: naive local dates (midnight), unique.
    """
    years = range(days.year.min() - 1, days.year.max() + 2)  # margin for bridge days at year ends
    fr_holidays = holidays.France(years=years)

    def is_off(d: pd.Timestamp) -> bool:
        return d.dayofweek >= 5 or d.date() in fr_holidays

    one_day = pd.Timedelta(days=1)
    is_holiday = [d.date() in fr_holidays for d in days]
    # Bridge day: working day between two days off (e.g. the Friday after Ascension Day)
    is_bridge = [not is_off(d) and is_off(d - one_day) and is_off(d + one_day) for d in days]
    # DST change: the local day does not last 24 h
    start = days.tz_localize(TZ)
    end = (days + one_day).tz_localize(TZ)
    is_dst_change = (end - start) != pd.Timedelta(hours=24)

    return pd.DataFrame({"is_holiday": is_holiday, "is_bridge": is_bridge,
                         "is_dst_change": is_dst_change}, index=days).astype("int8")


# --------------------------------------------------------------------- lags
def _same_hour_lag(values: pd.Series, day: pd.Series, hour: pd.Series, k: int) -> np.ndarray:
    """Value at the same local hour, k days earlier.

    Indexing by (local day, local hour) rather than shifting by 24 h: a 24-h shift would be one hour
    off around DST changes. The hour that does not exist (02:00 on the spring change) gives NaN;
    the doubled hour (October) is averaged.
    """
    by_day_hour = values.groupby([day, hour]).mean()
    target = pd.MultiIndex.from_arrays([day - pd.Timedelta(days=k), hour])
    return by_day_hour.reindex(target).to_numpy()


def _daily_stats_lag(values: pd.Series, day: pd.Series, k: int, stats=("mean", "min", "max")):
    """Statistics of local day D-k (mean, min, max), copied onto every hour of D."""
    daily = values.groupby(day).agg(list(stats))
    daily.index = daily.index + pd.Timedelta(days=k)
    return daily.reindex(day.to_numpy())


# --------------------------------------------------------------------- construction
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add every feature (ALL_FEATURES) to a copy of the DataFrame.

    Expects an hourly index and the RAW_PUBLICATION columns, plus `time_local`.
    Lags assume a continuous hourly history: apply to the full dataset, not to pieces of it.
    """
    missing = [c for c in list(RAW_PUBLICATION) + ["time_local"] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    out = df.copy()
    t = out["time_local"].dt.tz_convert(TZ)
    day = t.dt.tz_localize(None).dt.normalize()   # local day, naive
    hour = t.dt.hour

    # --- Calendar
    out["hour"] = hour.astype("int8")
    out["dayofweek"] = t.dt.dayofweek.astype("int8")
    out["month"] = t.dt.month.astype("int8")
    cal = _calendar_by_day(pd.DatetimeIndex(day.unique()))
    for c in cal.columns:
        out[c] = cal[c].reindex(day.to_numpy()).to_numpy()

    # --- French price lags
    for k in (1, 2, 7):
        out[f"price_d{k}"] = _same_hour_lag(out["price_da"], day, hour, k)
    stats = _daily_stats_lag(out["price_da"], day, 1)
    for s in ("mean", "min", "max"):
        out[f"price_d1_{s}"] = stats[s].to_numpy()

    # --- French fundamentals: residual load = what variable renewables leave to be covered
    fr_vre = out["fr_wind_fc"] + out["fr_solar_fc"]
    out["fr_residual_fc"] = out["fr_load_fc"] - fr_vre
    out["fr_vre_share"] = fr_vre / out["fr_load_fc"].where(out["fr_load_fc"] > 0)

    # --- Fuels: marginal cost of a CCGT
    out["srmc_ccgt"] = (out["ttf_d2"] / CCGT_EFFICIENCY
                        + out["eua_d2"] * CCGT_EMISSIONS + CCGT_VOM)

    # --- Germany
    out["de_price_d1"] = _same_hour_lag(out["de_price_da"], day, hour, 1)
    out["de_price_d1_mean"] = _daily_stats_lag(out["de_price_da"], day, 1, ("mean",))["mean"].to_numpy()
    out["de_vre_fc"] = out["de_wind_fc"] + out["de_solar_fc"]
    out["de_residual_fc"] = out["de_load_fc"] - out["de_vre_fc"]

    # --- Shape of day D: every hour "sees" the forecast profile of the whole day
    keys = day.to_numpy()
    rl = out["fr_residual_fc"]
    g = rl.groupby(keys)
    out["fr_rl_day_min"] = g.transform("min").to_numpy()
    out["fr_rl_day_max"] = g.transform("max").to_numpy()
    out["fr_rl_day_mean"] = g.transform("mean").to_numpy()
    out["fr_rl_rel"] = rl - out["fr_rl_day_mean"]                 # position of the hour within the day
    out["fr_rl_rank"] = g.rank(pct=True).to_numpy()
    tmp = pd.DataFrame({"rl": rl.to_numpy(), "hour": hour.to_numpy(), "day": keys}).dropna()
    min_hour = tmp.loc[tmp.groupby("day")["rl"].idxmin()].set_index("day")["hour"]
    out["fr_rl_min_hour"] = min_hour.reindex(keys).to_numpy()    # hour of the trough (often solar noon)
    out["fr_solar_day_max"] = out["fr_solar_fc"].groupby(keys).transform("max").to_numpy()
    out["de_rl_day_mean"] = out["de_residual_fc"].groupby(keys).transform("mean").to_numpy()
    out["price_d1_rel"] = out["price_d1"] - out["price_d1_mean"]  # shape of yesterday's price curve

    # --- Current regime: statistics of days D-1 to D-k (one-day shift: nothing from D itself)
    for c, v in _recent_regime(out, day).items():
        out[c] = v.reindex(keys).to_numpy()

    return out


def _recent_regime(out: pd.DataFrame, day: pd.Series) -> dict:
    """Daily indicators of the current regime, computed on strictly earlier days."""
    price = out["price_da"]
    x = out["fr_residual_fc"] / 1000                     # GW
    both = price.notna() & x.notna()
    xb, yb = x.where(both), price.where(both)
    daily = pd.DataFrame({
        "n": both.astype(float), "sx": xb, "sy": yb, "sxy": xb * yb, "sxx": xb * xb,
        "spread": price - out["srmc_ccgt"], "price": price,
        "neg": (price < 0).astype(float).where(price.notna()), "hours": price.notna().astype(float),
    }).groupby(day.to_numpy()).agg({"n": "sum", "sx": "sum", "sy": "sum", "sxy": "sum", "sxx": "sum",
                                    "spread": "mean", "price": "mean", "neg": "sum", "hours": "sum"})
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))

    r28 = daily[["n", "sx", "sy", "sxy", "sxx"]].rolling(28, min_periods=14).sum().shift(1)
    den = r28["n"] * r28["sxx"] - r28["sx"] ** 2
    slope = (r28["n"] * r28["sxy"] - r28["sx"] * r28["sy"]) / den.where(den > 0)
    r14 = daily.rolling(14, min_periods=7)
    return {
        "spread_14d": r14["spread"].mean().shift(1),     # mean price − SRMC gap: does gas set the price?
        "vol_14d": r14["price"].std().shift(1),          # volatility of daily mean prices
        "neg_share_14d": (r14["neg"].sum() / r14["hours"].sum()).shift(1),
        "rl_slope_28d": slope,                           # €/MWh per GW of residual load, last 28 days
    }


def gas_weight(df: pd.DataFrame) -> pd.Series:
    """SRMC weight w(RL): estimated pass-through of the gas cost, interpolated on the forecast
    residual load (constant beyond the extreme deciles; 1 if the forecast is missing)."""
    rl = df["fr_residual_fc"]
    w = np.interp(rl.fillna(np.nanmedian(PASS_THROUGH_RL)), PASS_THROUGH_RL, PASS_THROUGH)
    return pd.Series(np.where(rl.notna(), w, 1.0), index=df.index)


def srmc_offset(df: pd.DataFrame, step: str) -> pd.Series:
    """Amount subtracted from the price (target and €/MWh features) by the step's normalisation."""
    mode = NORMALIZATION.get(step)
    if mode == "full":
        return df["srmc_ccgt"]
    if mode == "rl":
        return gas_weight(df) * df["srmc_ccgt"]
    return pd.Series(0.0, index=df.index)


def get_X(df: pd.DataFrame, step: str) -> pd.DataFrame:
    """Feature matrix of an ablation step (DataFrame already passed through add_features).

    On normalised steps, €/MWh features are expressed relative to the same amount as the target:
    otherwise the inputs would keep the level problem that the target no longer has.
    """
    X = df[STEPS[step]].copy()
    if step in NORMALIZED_STEPS:
        offset = srmc_offset(df, step)
        for c in PRICE_LEVEL_FEATURES:
            if c in X:
                X[c] = X[c] - offset
    return X


def get_y(df: pd.DataFrame, step: str) -> pd.Series:
    """Target: day-ahead price, minus the (possibly weighted) SRMC on normalised steps."""
    return df["price_da"] - srmc_offset(df, step)


def to_price(pred: np.ndarray, df: pd.DataFrame, step: str) -> np.ndarray:
    """Convert a target prediction back to a price (€/MWh)."""
    return pred + srmc_offset(df, step).to_numpy()


# --------------------------------------------------------------------- leakage control
def gate_closure(delivery_day) -> pd.Timestamp:
    """Auction gate closure for delivery on `delivery_day`: 12:00 on D-1, Paris time."""
    d = pd.Timestamp(delivery_day).normalize()
    if d.tzinfo is not None:
        d = d.tz_localize(None)
    return (d - pd.Timedelta(days=1) + pd.Timedelta(hours=GATE_CLOSURE_HOUR)).tz_localize(TZ)


def mask_unpublished(df: pd.DataFrame, delivery_day) -> pd.DataFrame:
    """Copy in which every value published after the gate closure for `delivery_day` is NaN.

    Rebuilds what a forecaster could see at 12:00 on D-1, according to RAW_PUBLICATION.
    """
    cutoff = gate_closure(delivery_day)
    out = df.copy()
    day = out["time_local"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    for col, (days_before, hour) in RAW_PUBLICATION.items():
        published = (day - pd.Timedelta(days=days_before) + pd.Timedelta(hours=hour)).dt.tz_localize(TZ)
        out.loc[(published > cutoff).to_numpy(), col] = np.nan
    return out
