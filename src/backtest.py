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
REFIT_FREQ = {"monthly": "MS", "weekly": "7D", "daily": "D"}
REFIT_SUFFIX = {"weekly": "_hebdo", "daily": "_quotidien"}
# Variantes LightGBM à réentraînement plus fréquent que mensuel : nom -> (marche, rythme)
REFIT_VARIANTS = {"B1_hebdo": ("B1", "weekly"), "B2_hebdo": ("B2", "weekly"),
                  "B2_quotidien": ("B2", "daily"), "B2_rl_hebdo": ("B2_rl", "weekly"),
                  "B2_brut_hebdo": ("B2_brut", "weekly")}


def refit_dates(start: str, end: str, refit: str = "monthly") -> list:
    """Dates de réentraînement : le 1er de chaque mois, tous les 7 jours, ou chaque jour depuis `start`."""
    return list(pd.date_range(start, end, freq=REFIT_FREQ[refit], tz=TZ, inclusive="left"))


def walk_forward(f: pd.DataFrame, step: str, start: str, end: str, refit: str = "monthly") -> pd.Series:
    """Prévisions (en €/MWh) de la marche `step` sur [start, end), réentraînement au rythme `refit`."""
    y = get_y(f, step)
    X = get_X(f, step)
    dates = refit_dates(start, end, refit) + [pd.Timestamp(end, tz=TZ)]
    preds = []
    for m, m_end in zip(dates[:-1], dates[1:]):
        train = (f["time_local"] < m) & y.notna()
        pred = (f["time_local"] >= m) & (f["time_local"] < m_end)
        model = lgb.LGBMRegressor(**LGB_PARAMS).fit(X[train], y[train])
        p = to_price(model.predict(X[pred]), f[pred], step)
        preds.append(pd.Series(p, index=f.index[pred]))
        if refit != "daily" or m.day == 1:
            print(f"  {step} ({refit}) {m:%Y-%m-%d}  train={train.sum():6d}  pred={pred.sum():4d}")
    return pd.concat(preds).rename(step + REFIT_SUFFIX.get(refit, ""))


# --------------------------------------------------------------------- test de Diebold-Mariano
# Paires comparées (A, B) : choisies à l'avance pour limiter les tests multiples.
DM_PAIRS = [
    ("B2", "naive_d7"),             # le modèle complet bat-il la référence ?
    ("B2", "B2_brut"),              # la normalisation par le SRMC change-t-elle quelque chose ?
    ("B2", "LEAR_quotidien"),       # meilleur LightGBM vs référence de la littérature
    ("B1", "LEAR_quotidien"),       # à information égale
    ("B1_hebdo", "LEAR_hebdo"),     # à information égale et même rythme de réentraînement
    ("B2_hebdo", "B2"),             # réentraîner chaque semaine plutôt que chaque mois
    ("B2_hebdo", "LEAR_quotidien"), # meilleur LightGBM vs meilleur LEAR (ajoutée après les runs hebdo)
    ("B2_quotidien", "B2_hebdo"),   # le réentraînement quotidien apporte-t-il encore ? (fixée avant le run)
    ("B2_quotidien", "LEAR_quotidien"),  # même rythme que le LEAR de l'article (fixée avant le run)
    ("B2_rl_hebdo", "B2_hebdo"),    # normalisation pondérée par la charge résiduelle vs pleine
    ("B2_rl_hebdo", "B2_brut_hebdo"),  # ... vs aucune normalisation
    ("B2_rl", "B2"),                # idem au rythme mensuel (période de crise)
    ("B2_rl", "B2_brut"),
    ("Chronos2_fond", "Chronos2_prix"),  # les fondamentaux aident-ils un modèle de fondation ?
    ("B2_hebdo", "Chronos2_fond"),       # LightGBM vs modèle de fondation, mêmes informations
    ("LEAR_quotidien", "LEAR_mensuel"),
    ("B2", "MSTL"),
]


def diebold_mariano(err_a: pd.Series, err_b: pd.Series, day: pd.Series) -> dict:
    """Test de Diebold-Mariano sur la perte absolue, à variance HAC (Newey-West).

    Comme Lago et al. (2021), on moyenne d'abord le différentiel de perte par jour (les 24 heures d'un
    jour sont fortement corrélées), puis on corrige l'autocorrélation entre jours par Newey-West.
    t < 0 : A a une erreur plus faible que B.
    """
    d = (err_a.abs() - err_b.abs()).groupby(day.to_numpy()).mean().dropna().to_numpy()
    T = len(d)
    lags = int(np.floor(4 * (T / 100) ** (2 / 9)))      # règle usuelle de Newey-West
    u = d - d.mean()
    lrv = u @ u / T
    for k in range(1, lags + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / T
    if lrv <= 0:   # différentiel constant (ex. deux prévisions identiques)
        return {"ecart_MAE": d.mean(), "t_HAC": np.nan, "p_value": np.nan, "jours": T, "lags_NW": lags}
    t = d.mean() / np.sqrt(lrv / T)
    p = 2 * (1 - _normal_cdf(abs(t)))
    return {"ecart_MAE": d.mean(), "t_HAC": t, "p_value": p, "jours": T, "lags_NW": lags}


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
def load_predictions(period: str) -> pd.DataFrame:
    """Relit les prédictions déjà calculées : un fichier par modèle, pour que plusieurs runs
    en parallèle n'écrasent pas les résultats les uns des autres."""
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
                    help="modèles à (re)calculer ; les autres sont relus dans les prédictions déjà sauvées "
                         "(--models sans argument : recalcule seulement les tableaux)")
    args = ap.parse_args()
    if args.period == "test":
        print("ATTENTION : période de test. À ne lancer qu'une fois les choix figés sur la validation.")

    start, end = PERIODS[args.period]
    f = add_features(load_dataset())
    in_period = (f["time_local"] >= pd.Timestamp(start, tz=TZ)) & (f["time_local"] < pd.Timestamp(end, tz=TZ))

    for name in args.models:
        if name in BENCHMARKS:
            pred = BENCHMARKS[name](f, start, end)
        elif name in STEPS:
            pred = walk_forward(f, name, start, end)
        else:
            step, refit = REFIT_VARIANTS[name]
            pred = walk_forward(f, step, start, end, refit=refit)
        save_prediction(args.period, pred.rename(name))   # sauvé tout de suite : un plantage ne perd rien

    # Tableaux sur tout ce qui est disponible (y compris les modèles d'autres runs)
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
        print(f"\n=== {args.period} : {start} -> {end} ===")
        print(table.drop(columns="heures", level=1))
        print("heures par régime :", table.xs("heures", axis=1, level=1).iloc[0].astype(int).to_dict())
        print("\nDiebold-Mariano (perte absolue, moyenne journalière, HAC Newey-West) :")
        print(dm)


if __name__ == "__main__":
    main()
