"""Benchmarks : MSTL (séries temporelles classiques) et LEAR (référence linéaire de la littérature).

Les naïfs J-1 et J-7 sont directement les features price_d1 et price_d7 (voir backtest.py).

MSTL — prix seul. Décompose les 8 dernières semaines de prix horaires en tendance + saisonnalité
journalière (24 h) + hebdomadaire (168 h) + reste, puis prolonge chaque composante. Réestimé chaque
jour : sans entrée exogène, un modèle figé un mois ne verrait pas les prix récents.

LEAR — Lago, Marcjasz, De Schutter et Weron (2021), Applied Energy 293:116983. Régression linéaire
pénalisée (LASSO, pénalité choisie par AIC), un modèle par heure de livraison. Entrées pour le jour J :
prix des 24 heures de J-1, J-2, J-3 et J-7, prévisions de conso et d'ENR françaises pour J, J-1 et
J-7, jour de la semaine. Prix et entrées transformés par asinh après centrage-réduction robuste
(médiane, MAD). Prévision = moyenne des modèles calibrés sur 56, 84, 1092 et 1456 jours.
Mêmes informations que la marche B1 (calendrier, lags de prix, fondamentaux FR), sans gaz ni DE.

Écarts à l'original, à documenter :
- Fenêtres courtes (56 et 84 j < 247 entrées) : scikit-learn récent exige alors une estimation de la
  variance du bruit pour l'AIC ; on reprend celle du modèle 1456 j à la même heure.
- Version mensuelle en plus de la version quotidienne de l'article, pour comparer à LightGBM avec le
  même rythme de réestimation.
"""
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsIC
from statsforecast.models import MSTL

from src.config import TZ

LEAR_WINDOWS = (1456, 1092, 84, 56)   # la plus longue d'abord : elle fournit la variance du bruit
MSTL_WEEKS = 8


# --------------------------------------------------------------------- outils
def _local_day_hour(f: pd.DataFrame):
    t = f["time_local"].dt.tz_convert(TZ)
    return t.dt.tz_localize(None).dt.normalize(), t.dt.hour


def _daily_panel(values: pd.Series, day: pd.Series, hour: pd.Series, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Tableau jours × 24 heures locales. L'heure manquante (passage à l'heure d'été) est interpolée,
    l'heure doublée (octobre) moyennée."""
    panel = values.groupby([day, hour]).mean().unstack().reindex(index=days, columns=range(24))
    return panel.interpolate(axis=1, limit_direction="both")


def _to_hourly(panel: pd.DataFrame, day: pd.Series, hour: pd.Series) -> np.ndarray:
    """Repasse un tableau jours × heures sur l'index horaire."""
    stacked = panel.stack()
    return stacked.reindex(pd.MultiIndex.from_arrays([day, hour])).to_numpy()


def _period_days(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="D", inclusive="left")


# --------------------------------------------------------------------- MSTL
def mstl_forecasts(f: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Prévision MSTL de chaque jour de [start, end), à partir des prix connus jusqu'à la fin de J-1."""
    price = f["price_da"]
    local = f["time_local"].dt.tz_convert(TZ)
    out = pd.Series(np.nan, index=f.index)
    t0 = time.time()
    for d in _period_days(start, end):
        day_start = pd.Timestamp(d, tz=TZ)
        day_end = day_start + pd.Timedelta(days=1)
        history = price[(local < day_start).to_numpy()].dropna().iloc[-MSTL_WEEKS * 168:]
        target = f.index[((local >= day_start) & (local < day_end)).to_numpy()]
        fc = MSTL(season_length=[24, 168]).fit(history.to_numpy()).predict(h=len(target))["mean"]
        out[target] = fc
    print(f"  MSTL : {len(_period_days(start, end))} jours en {time.time() - t0:.0f} s")
    return out.loc[out.notna()].rename("MSTL")


# --------------------------------------------------------------------- LEAR
def _lear_design(f: pd.DataFrame):
    """Matrice d'entrées LEAR (une ligne par jour de livraison) et cible jours × 24 heures."""
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
    X = pd.concat(blocks, axis=1)
    X.columns = [f"{b}_h{h}" for b, h in X.columns]
    dummies = pd.get_dummies(days.dayofweek, prefix="dow").astype(float)
    dummies.index = days
    X = X.ffill()   # trous de prévision : on reprend la veille à la même heure
    return X, dummies, price, day, hour


def _robust_asinh(train: np.ndarray, apply: np.ndarray):
    """Centrage-réduction par médiane et MAD (calculés sur `train`), puis asinh."""
    med = np.nanmedian(train, axis=0)
    mad = np.nanmedian(np.abs(train - med), axis=0) / 0.6745
    mad = np.where(mad > 0, mad, 1.0)
    return np.arcsinh((train - med) / mad), np.arcsinh((apply - med) / mad), med, mad


def _lear_fit_predict(X, dummies, price, fit_day, predict_days):
    """Calibre les 4 fenêtres × 24 heures sur les jours < fit_day, prévoit `predict_days` (jours × 24)."""
    valid = X.notna().all(axis=1) & price.notna().all(axis=1)
    hist = valid.index[(valid.index < fit_day) & valid.to_numpy()]
    Xp_raw = X.loc[predict_days].to_numpy()
    Dp = dummies.loc[predict_days].to_numpy()
    preds = []
    noise = {}
    for w in LEAR_WINDOWS:
        tr = hist[-w:]
        Xt, Xp, _, _ = _robust_asinh(X.loc[tr].to_numpy(), Xp_raw)
        Xt = np.hstack([Xt, dummies.loc[tr].to_numpy()])
        Xp = np.hstack([Xp, Dp])
        p = np.empty((len(predict_days), 24))
        for h in range(24):
            yt, _, med, mad = _robust_asinh(price.loc[tr, h].to_numpy(), np.zeros(1))
            kw = {"noise_variance": noise[h]} if len(tr) <= Xt.shape[1] else {}
            m = LassoLarsIC(criterion="aic", **kw).fit(Xt, yt)
            if not kw:
                noise.setdefault(h, m.noise_variance_)
            p[:, h] = np.sinh(m.predict(Xp)) * mad + med
        preds.append(p)
    return pd.DataFrame(np.mean(preds, axis=0), index=predict_days, columns=range(24))


LEAR_NAMES = {"daily": "LEAR_quotidien", "weekly": "LEAR_hebdo", "monthly": "LEAR_mensuel"}


def lear_forecasts(f: pd.DataFrame, start: str, end: str, refit: str = "daily") -> pd.Series:
    """Prévisions LEAR sur [start, end), réestimées chaque jour, semaine (7 j) ou mois."""
    X, dummies, price, day, hour = _lear_design(f)
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
            print(f"  LEAR quotidien : {i}/{len(groups)} jours ({time.time() - t0:.0f} s)")
    name = LEAR_NAMES[refit]
    print(f"  {name} : {len(groups)} réestimations en {time.time() - t0:.0f} s")
    hourly = pd.Series(_to_hourly(pd.concat(panels), day, hour), index=f.index)
    return hourly.loc[hourly.notna()].rename(name)


# --------------------------------------------------------------------- Chronos-2
# Modèle de fondation zero-shot (Amazon, rapport technique arXiv 2510.15821 du 17/10/2025).
# Sorti avant le début du test (01/11/2025) : il ne peut pas avoir vu le test. Il a pu voir la
# validation (nov. 2024 -> oct. 2025) : ses résultats de validation sont indicatifs seulement.
CHRONOS_MODEL = "amazon/chronos-2"
CHRONOS_CONTEXT = 2048     # heures d'historique (~12 semaines)
# Variables connues à l'avance pour J (passées et futures), comme dans la marche B2
CHRONOS_KNOWN = ["fr_load_fc", "fr_wind_fc", "fr_solar_fc", "de_load_fc", "de_wind_fc",
                 "de_solar_fc", "srmc_ccgt", "is_holiday"]
CHRONOS_PAST_ONLY = ["de_price_da"]   # connu jusqu'à J-1 seulement

_chronos_pipeline = None


def _chronos():
    global _chronos_pipeline
    if _chronos_pipeline is None:
        from chronos import Chronos2Pipeline   # import tardif : PyTorch seulement si on s'en sert
        _chronos_pipeline = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL, device_map="cpu")
    return _chronos_pipeline


def chronos_forecasts(f: pd.DataFrame, start: str, end: str, covariates: bool) -> pd.Series:
    """Médiane prédite par Chronos-2 pour chaque jour de [start, end), à partir des
    CHRONOS_CONTEXT dernières heures connues à la fin de J-1 (et des variables connues pour J)."""
    name = "Chronos2_fond" if covariates else "Chronos2_prix"
    local = f["time_local"].dt.tz_convert(TZ)
    cols = CHRONOS_KNOWN + CHRONOS_PAST_ONLY
    cov = f[cols].interpolate(limit_direction="both").astype(float) if covariates else None
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
    for length in sorted({len(h) for h in targets}):   # 23, 24 ou 25 h (changements d'heure)
        idx = [i for i, h in enumerate(targets) if len(h) == length]
        quantiles, _ = _chronos().predict_quantiles([tasks[i] for i in idx], prediction_length=length,
                                                    quantile_levels=[0.5], batch_size=32)
        for i, q in zip(idx, quantiles):
            out.iloc[targets[i]] = q[0, :, 0].numpy()
    print(f"  {name} : {len(tasks)} jours en {time.time() - t0:.0f} s")
    return out.loc[out.notna()].rename(name)


BENCHMARKS = {
    "Chronos2_prix": lambda f, s, e: chronos_forecasts(f, s, e, covariates=False),
    "Chronos2_fond": lambda f, s, e: chronos_forecasts(f, s, e, covariates=True),
    "MSTL": mstl_forecasts,
    "LEAR_mensuel": lambda f, s, e: lear_forecasts(f, s, e, refit="monthly"),
    "LEAR_hebdo": lambda f, s, e: lear_forecasts(f, s, e, refit="weekly"),
    "LEAR_quotidien": lambda f, s, e: lear_forecasts(f, s, e, refit="daily"),
}
