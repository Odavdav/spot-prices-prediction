"""Feature importance of the main model (B2): SHAP values, overall, by regime and by hour.

The model is trained once on every delivery before the test year (before 2025-11-01), then
explained on the test hours. SHAP values (TreeSHAP, provided by LightGBM) decompose each prediction
into additive feature contributions, in €/MWh. Since B2's target is price − SRMC, they explain the
spread to the cost of a CCGT; the SRMC level itself is added back afterwards and does not appear here.

Usage: python -m src.importance
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.backtest import LGB_PARAMS, regime
from src.config import PERIODS, RESULTS, TZ
from src.data import load_dataset
from src.features import add_features, get_X, get_y

STEP = "B2"

# Readable feature groups (shown in the README figures)
GROUPS = {
    "FR price D-1 (same hour)": ["price_d1"],
    "FR price D-1 (mean, min, max)": ["price_d1_mean", "price_d1_min", "price_d1_max"],
    "FR price D-2, D-7": ["price_d2", "price_d7"],
    "FR residual load": ["fr_residual_fc"],
    "FR load": ["fr_load_fc"],
    "FR solar": ["fr_solar_fc"],
    "FR wind": ["fr_wind_fc"],
    "FR renewables share": ["fr_vre_share"],
    "DE price D-1": ["de_price_d1", "de_price_d1_mean"],
    "DE residual load / load": ["de_residual_fc", "de_load_fc"],
    "DE wind + solar": ["de_vre_fc"],
    "CCGT SRMC (gas, CO2)": ["srmc_ccgt"],
    "Hour": ["hour"],
    "Calendar (day, month, holidays)": ["dayofweek", "month", "is_holiday", "is_bridge", "is_dst_change"],
}


def shap_values() -> tuple[pd.DataFrame, pd.DataFrame]:
    """SHAP values (test hours × feature groups) and the matching dataset rows."""
    f = add_features(load_dataset())
    start, end = (pd.Timestamp(d, tz=TZ) for d in PERIODS["test"])
    X, y = get_X(f, STEP), get_y(f, STEP)
    train = (f["time_local"] < start) & y.notna()
    test = (f["time_local"] >= start) & (f["time_local"] < end) & f["price_da"].notna()
    model = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[train], y[train])
    contrib = model.predict(X[test], pred_contrib=True)[:, :-1]   # last column: constant
    shap = pd.DataFrame(contrib, index=f.index[test], columns=X.columns)
    grouped = pd.DataFrame({g: shap[cols].sum(axis=1) for g, cols in GROUPS.items()})
    return grouped, f.loc[test]


def main():
    shap, rows = shap_values()
    mag = shap.abs()
    reg = regime(rows).map({"negative": "Negative hours", "evening_peak": "Evening peak (18-22h)",
                            "other": "Other hours"})
    hour = rows["time_local"].dt.hour

    overall = mag.mean().sort_values(ascending=False).rename("mean_abs_shap")
    by_regime = mag.groupby(reg.to_numpy()).mean().T
    by_hour = mag.groupby(hour.to_numpy()).mean().T

    RESULTS.mkdir(exist_ok=True)
    overall.round(3).to_csv(RESULTS / "importance_overall.csv")
    by_regime.round(3).to_csv(RESULTS / "importance_by_regime.csv")
    by_hour.round(3).to_csv(RESULTS / "importance_by_hour.csv")
    with pd.option_context("display.width", 140, "display.float_format", "{:.2f}".format):
        print("Mean |SHAP| importance (€/MWh), test:\n", overall)
        print("\nBy regime:\n", by_regime.loc[overall.index])


if __name__ == "__main__":
    main()
