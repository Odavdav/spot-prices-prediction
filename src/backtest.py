"""Backtest walk-forward mensuel et métriques.

Protocole : pour chaque mois M de la période évaluée, le modèle est entraîné sur toutes les
livraisons antérieures au 1er du mois (fenêtre expansive), puis prévoit chaque jour de M.
Ne pas réentraîner en cours de mois est prudent : le 15, on aurait déjà les prix jusqu'au 14.

Métriques : MAE et rMAE (MAE / MAE du naïf J-7 sur les mêmes heures), par régime.

Usage :
    python -m src.backtest                    # validation (nov. 2024 -> oct. 2025)
    python -m src.backtest --period test      # test : à ne lancer qu'à la fin du projet
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.benchmarks import BENCHMARKS
from src.config import PERIODS, PREDICTIONS_DIR, RESULTS, SEED, TZ
from src.data import load_dataset
from src.features import STEPS, add_features, get_X, get_y, to_price

# Hyperparamètres fixes, raisonnables pour ~50 000 lignes et ~30 features (pas de tuning à ce stade)
LGB_PARAMS = dict(
    objective="l1",          # cohérent avec la MAE ; robuste aux pics de prix
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
EVENING_PEAK = range(18, 22)  # 18h-21h59, heure locale


# --------------------------------------------------------------------- walk-forward
def month_starts(start: str, end: str) -> list:
    return list(pd.date_range(start, end, freq="MS", tz=TZ, inclusive="left"))


def walk_forward(f: pd.DataFrame, step: str, start: str, end: str) -> pd.Series:
    """Prévisions (en €/MWh) de la marche `step` sur [start, end), réentraînement mensuel."""
    y = get_y(f, step)
    X = get_X(f, step)
    preds = []
    for m in month_starts(start, end):
        m_end = m + pd.offsets.MonthBegin(1)
        train = (f["time_local"] < m) & y.notna()
        pred = (f["time_local"] >= m) & (f["time_local"] < m_end)
        model = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[train], y[train])
        p = to_price(model.predict(X[pred]), f[pred], step)
        preds.append(pd.Series(p, index=f.index[pred]))
        print(f"  {step} {m:%Y-%m}  train={train.sum():6d}  pred={pred.sum():4d}")
    return pd.concat(preds).rename(step)


# --------------------------------------------------------------------- métriques
def regime(f: pd.DataFrame) -> pd.Series:
    """Régime de chaque heure : prix négatif, pointe du soir, reste (sur le prix réalisé)."""
    r = pd.Series("reste", index=f.index)
    r[f["time_local"].dt.hour.isin(EVENING_PEAK)] = "pointe_soir"
    r[f["price_da"] < 0] = "negatif"   # prioritaire : une heure négative de pointe reste négative
    return r


def metrics(preds: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """MAE et rMAE (vs naïf J-7) par modèle, sur toutes les heures et par régime.

    Seules les heures où tous les modèles ont une prévision sont comptées, pour comparer à périmètre égal.
    """
    rows = f.loc[preds.index]
    ok = preds.notna().all(axis=1) & rows["price_da"].notna()
    err = preds[ok].sub(rows.loc[ok, "price_da"], axis=0).abs()
    reg = regime(rows[ok])
    out = {}
    for name, mask in [("toutes", slice(None))] + [(r, reg == r) for r in ("negatif", "pointe_soir", "reste")]:
        e = err[mask]
        out[(name, "MAE")] = e.mean()
        out[(name, "rMAE")] = e.mean() / e["naive_d7"].mean()
        out[(name, "heures")] = pd.Series(len(e), index=e.columns)
    table = pd.DataFrame(out)
    table.columns = pd.MultiIndex.from_tuples(table.columns)
    return table


# --------------------------------------------------------------------- main
def main():
    models = list(STEPS) + list(BENCHMARKS)
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=list(PERIODS), default="validation")
    ap.add_argument("--models", nargs="+", choices=models, default=models,
                    help="modèles à (re)calculer ; les autres sont relus dans les prédictions déjà sauvées")
    args = ap.parse_args()
    if args.period == "test":
        print("ATTENTION : période de test. À ne lancer qu'une fois les choix figés sur la validation.")

    start, end = PERIODS[args.period]
    f = add_features(load_dataset())
    in_period = (f["time_local"] >= pd.Timestamp(start, tz=TZ)) & (f["time_local"] < pd.Timestamp(end, tz=TZ))

    path = PREDICTIONS_DIR / f"predictions_{args.period}.parquet"
    preds = pd.read_parquet(path) if path.exists() else pd.DataFrame(index=f.index[in_period])
    for name, col in NAIVE.items():
        preds[name] = f.loc[preds.index, col]
    for name in args.models:
        if name in STEPS:
            preds[name] = walk_forward(f, name, start, end)
        else:
            preds[name] = BENCHMARKS[name](f, start, end)
    order = list(NAIVE) + [m for m in models if m in preds]
    preds = preds[order]

    table = metrics(preds, f)
    RESULTS.mkdir(exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(RESULTS / f"metrics_{args.period}.csv")
    preds.to_parquet(path)

    with pd.option_context("display.width", 140, "display.float_format", "{:.2f}".format):
        print(f"\n=== {args.period} : {start} -> {end} ===")
        print(table.drop(columns="heures", level=1))
        print("heures par régime :", table.xs("heures", axis=1, level=1).iloc[0].astype(int).to_dict())


if __name__ == "__main__":
    main()
