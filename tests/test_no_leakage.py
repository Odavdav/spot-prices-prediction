"""Leakage tests: no feature of day D may use information published after SDAC gate closure
(12:00 on D-1).

The tests run on a synthetic dataset: they do not depend on data/ (not versioned).
Run from the repository root: python -m pytest
"""
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.features import (ALL_FEATURES, FEATURE_REGISTRY, RAW_PUBLICATION, STEPS, TZ, add_features,
                          get_X, is_known_at_gate_closure, mask_unpublished)


@pytest.fixture(scope="module")
def synthetic() -> pd.DataFrame:
    """Random hourly data around the spring DST change (2024-03-31) and Easter, with enough history
    (6 weeks) for the 14- and 28-day rolling windows to be filled."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-02-15", "2024-04-12", freq="h", tz="UTC")
    df = pd.DataFrame(index=idx)
    for col in RAW_PUBLICATION:
        df[col] = rng.normal(100, 30, len(idx))
    df["time_local"] = idx.tz_convert(TZ)
    # Gas and carbon: one value per local day, as in the real dataset
    day = df["time_local"].dt.tz_localize(None).dt.normalize()
    for col in ("ttf_d2", "eua_d2"):
        daily = pd.Series(rng.normal(50, 10, day.nunique()), index=day.unique())
        df[col] = daily.reindex(day.to_numpy()).to_numpy()
    return df


# ------------------------------------------------------------------ 1. the register is complete and safe
def test_every_feature_is_registered():
    missing = [f for f in ALL_FEATURES if f not in FEATURE_REGISTRY]
    assert not missing, f"Features with no FEATURE_REGISTRY entry: {missing}"


def test_registered_sources_exist():
    unknown = {c for cols, _ in FEATURE_REGISTRY.values() for c in cols if c not in RAW_PUBLICATION}
    assert not unknown, f"Columns with no publication time in RAW_PUBLICATION: {unknown}"


@pytest.mark.parametrize("feature", ALL_FEATURES)
def test_declared_feature_is_known_at_gate_closure(feature):
    cols, lag_days = FEATURE_REGISTRY[feature]
    late = [c for c in cols if not is_known_at_gate_closure(c, lag_days)]
    assert not late, f"{feature} uses {late} (D-{lag_days}), published after 12:00 on D-1"


# ------------------------------------------------------------------ 2. the code complies with the register
DELIVERY_DAYS = ["2024-03-29",   # Good Friday
                 "2024-03-31",   # spring DST change (23 h)
                 "2024-04-01",   # Easter Monday, the day after the DST change
                 "2024-04-08"]   # D-7 falls on the DST change


@pytest.mark.parametrize("delivery_day", DELIVERY_DAYS)
def test_features_unchanged_when_future_is_erased(synthetic, delivery_day):
    """Erasing everything published after gate closure must not change the features of day D."""
    full = add_features(synthetic)
    seen = add_features(mask_unpublished(synthetic, delivery_day))
    on_day = (full["time_local"].dt.strftime("%Y-%m-%d") == delivery_day).to_numpy()
    assert on_day.sum() in (23, 24, 25)
    pdt.assert_frame_equal(full.loc[on_day, ALL_FEATURES], seen.loc[on_day, ALL_FEATURES],
                           check_dtype=False)
    for step in STEPS:  # including the SRMC-normalised features
        pdt.assert_frame_equal(get_X(full, step)[on_day], get_X(seen, step)[on_day], check_dtype=False)


def test_masking_catches_a_leaky_feature(synthetic):
    """Check of the test itself: a feature reading the previous hour's price must be caught."""
    day = "2024-04-08"
    leaky_full = synthetic["price_da"].shift(1)
    leaky_seen = mask_unpublished(synthetic, day)["price_da"].shift(1)
    on_day = (synthetic["time_local"].dt.strftime("%Y-%m-%d") == day).to_numpy()
    assert not leaky_full[on_day].equals(leaky_seen[on_day])


# ------------------------------------------------------------------ 3. lags at a fixed local hour
def test_lags_follow_local_hour_across_dst(synthetic):
    f = add_features(synthetic).set_index("time_local")
    price = synthetic.set_index("time_local")["price_da"]
    ts = lambda s: pd.Timestamp(s, tz=TZ)
    # Monday 01/04 at 03:00 (summer time) -> Sunday 31/03 at 03:00 (summer time): 23 h apart, not 24
    assert f.loc[ts("2024-04-01 03:00"), "price_d1"] == price[ts("2024-03-31 03:00")]
    # 02:00 does not exist on 31/03: no D-1 lag at 02:00 on 01/04
    assert np.isnan(f.loc[ts("2024-04-01 02:00"), "price_d1"])
    # D-7 crosses the DST change
    assert f.loc[ts("2024-04-05 18:00"), "price_d7"] == price[ts("2024-03-29 18:00")]
    # Mean of D-1 = mean of the previous local day (23 h on 31/03)
    expected = price[(price.index >= ts("2024-03-31")) & (price.index < ts("2024-04-01"))].mean()
    assert f.loc[ts("2024-04-01 12:00"), "price_d1_mean"] == pytest.approx(expected)


def test_error_correction_uses_only_past_prices():
    """The correction for day D must not depend on the price of D or of any later day."""
    from src.correction import error_correction
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", "2024-06-30", freq="h", tz="UTC")
    t = pd.Series(idx.tz_convert(TZ), index=idx)
    price = pd.Series(rng.normal(80, 20, len(idx)), index=idx)
    base = (price + rng.normal(5, 10, len(idx))).rename("base")
    start, end, day = "2024-05-01", "2024-06-01", "2024-05-15"
    full = error_correction(base, price, t, start, end)
    after = (t >= pd.Timestamp(day, tz=TZ)).to_numpy()
    price2 = price.copy()
    price2[after] = rng.normal(500, 100, after.sum())     # prices of D and later days scrambled
    base2 = base.copy()
    later = (t >= pd.Timestamp(day, tz=TZ) + pd.Timedelta(days=1)).to_numpy()
    base2[later] = 0.0                                      # forecasts of later days too
    seen = error_correction(base2, price2, t, start, end)
    on_day = (full.index.tz_convert(TZ).strftime("%Y-%m-%d") == day)
    pdt.assert_series_equal(full[on_day], seen[on_day])
    assert not full.equals(seen)                            # the test does have teeth


def test_calendar_flags():
    idx = pd.date_range("2024-05-08", "2024-05-12", freq="h", tz=TZ)
    df = pd.DataFrame({c: 1.0 for c in RAW_PUBLICATION}, index=idx)
    df["time_local"] = idx
    f = add_features(df).groupby(df["time_local"].dt.strftime("%Y-%m-%d"))[
        ["is_holiday", "is_bridge"]].first()
    assert f.loc["2024-05-09", "is_holiday"] == 1   # Ascension Day (Thursday)
    assert f.loc["2024-05-10", "is_bridge"] == 1    # bridge Friday
    assert f.loc["2024-05-08", "is_holiday"] == 1   # 8 May
    assert f.loc["2024-05-10", "is_holiday"] == 0
