"""Correction d'un modèle par ses erreurs récentes (deuxième étage).

Idée (imiter l'adaptation de Chronos-2) : le modèle de base fait ses prévisions hors échantillon ;
un petit modèle apprend à prévoir l'erreur du jour J à partir des erreurs récentes :
    - erreur de J-1 à la même heure ;
    - erreur moyenne de J-1 ;
    - erreur moyenne des 7 derniers jours, à la même heure et toutes heures confondues.
Prévision corrigée = prévision de base − erreur prévue.

Ensemble d'information : l'erreur de J-1 = prévision de J-1 (faite à 12h en J-2) − prix de J-1
(publié vers 13h en J-2). Elle est donc connue à 12h en J-1, avant la fermeture pour J.
Le modèle de correction est une régression médiane (cohérente avec la MAE), réestimée chaque semaine
sur tout l'historique d'erreurs disponible (fenêtre expansive).

Usage :
    python -m src.correction --base B2_hebdo --period validation --history rodage
    python -m src.correction --base B2_quotidien --period test --history validation
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from src.backtest import PREDICTIONS_DIR, save_prediction
from src.config import PERIODS, TZ
from src.data import load_dataset

FEATURES = ["err_d1", "err_d1_mean", "err_7d_hour", "err_7d_mean"]
MIN_HISTORY_DAYS = 90   # pas de correction tant que l'historique d'erreurs est trop court


def _day_hour(time_local: pd.Series):
    t = time_local.dt.tz_convert(TZ)
    return t.dt.tz_localize(None).dt.normalize(), t.dt.hour


def error_features(error: pd.Series, time_local: pd.Series) -> pd.DataFrame:
    """Tableau (jour, heure) : erreur du jour (cible) et features d'erreurs passées (J-1 à J-7)."""
    day, hour = _day_hour(time_local)
    panel = error.groupby([day, hour]).mean().unstack()            # jours × heures locales
    panel = panel.reindex(pd.date_range(panel.index.min(), panel.index.max(), freq="D"))
    past = [panel.shift(k) for k in range(1, 8)]                   # erreurs de J-1 … J-7
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
    """Prévision corrigée sur [start, end). `base`, `price`, `time_local` couvrent l'historique et la période.

    Seuls les prix des jours < J servent à corriger le jour J (via les erreurs passées)."""
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
    # Retour sur l'index horaire
    day, hour = _day_hour(time_local)
    in_period = (day >= days[0]) & (day <= days[-1])
    key = pd.MultiIndex.from_arrays([day[in_period], hour[in_period]])
    est_error = corr.reindex(key).fillna(0.0).to_numpy()
    return (base[in_period] - est_error).rename(f"{base.name}_corr")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--period", choices=list(PERIODS), required=True)
    ap.add_argument("--history", nargs="+", default=[], help="périodes dont les prévisions servent d'historique")
    args = ap.parse_args()

    parts = [pd.read_parquet(PREDICTIONS_DIR / p / f"{args.base}.parquet").iloc[:, 0]
             for p in args.history + [args.period]]
    base = pd.concat(parts).sort_index()
    base = base[~base.index.duplicated(keep="last")].rename(args.base)
    df = load_dataset().loc[base.index]
    out = error_correction(base, df["price_da"], df["time_local"], *PERIODS[args.period])
    save_prediction(args.period, out)
    real = df.loc[out.index, "price_da"]
    print(f"{args.base} : MAE {(base.loc[out.index] - real).abs().mean():.2f} -> "
          f"{out.name} : {(out - real).abs().mean():.2f} €/MWh")


if __name__ == "__main__":
    main()
