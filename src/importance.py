"""Importance des features du modèle final (B2) : valeurs SHAP, globales, par régime et par heure.

Le modèle est entraîné une fois sur toutes les livraisons antérieures au test (avant le 01/11/2025),
puis expliqué sur les heures du test. Les valeurs SHAP (TreeSHAP, fournies par LightGBM) décomposent
chaque prédiction en contributions additives des features, en €/MWh. La cible de B2 étant
prix − SRMC, elles expliquent l'écart au coût d'un CCGT ; le niveau du SRMC lui-même est rajouté
après coup et n'apparaît donc pas ici.

Usage : python -m src.importance
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.backtest import LGB_PARAMS, regime
from src.config import PERIODS, RESULTS, TZ
from src.data import load_dataset
from src.features import add_features, get_X, get_y

STEP = "B2"

# Regroupement lisible des features (libellés en anglais : ils apparaissent dans les figures du README)
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
    """Valeurs SHAP (heures du test × groupes de features) et lignes correspondantes du jeu."""
    f = add_features(load_dataset())
    start, end = (pd.Timestamp(d, tz=TZ) for d in PERIODS["test"])
    X, y = get_X(f, STEP), get_y(f, STEP)
    train = (f["time_local"] < start) & y.notna()
    test = (f["time_local"] >= start) & (f["time_local"] < end) & f["price_da"].notna()
    model = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[train], y[train])
    contrib = model.predict(X[test], pred_contrib=True)[:, :-1]   # dernière colonne : constante
    shap = pd.DataFrame(contrib, index=f.index[test], columns=X.columns)
    grouped = pd.DataFrame({g: shap[cols].sum(axis=1) for g, cols in GROUPS.items()})
    return grouped, f.loc[test]


def main():
    shap, rows = shap_values()
    mag = shap.abs()
    reg = regime(rows).map({"negatif": "Negative hours", "pointe_soir": "Evening peak (18-22h)",
                            "reste": "Other hours"})
    hour = rows["time_local"].dt.hour

    overall = mag.mean().sort_values(ascending=False).rename("mean_abs_shap")
    by_regime = mag.groupby(reg.to_numpy()).mean().T
    by_hour = mag.groupby(hour.to_numpy()).mean().T

    RESULTS.mkdir(exist_ok=True)
    overall.round(3).to_csv(RESULTS / "importance_globale.csv")
    by_regime.round(3).to_csv(RESULTS / "importance_par_regime.csv")
    by_hour.round(3).to_csv(RESULTS / "importance_par_heure.csv")
    with pd.option_context("display.width", 140, "display.float_format", "{:.2f}".format):
        print("Importance moyenne |SHAP| (€/MWh), test :\n", overall)
        print("\nPar régime :\n", by_regime.loc[overall.index])


if __name__ == "__main__":
    main()
