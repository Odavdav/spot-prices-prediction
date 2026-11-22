"""Construction des features et registre de publication (anti-fuite).

Règle d'or : pour la livraison du jour J, une feature doit être connue avant la fermeture de
l'enchère SDAC, à 12h00 (heure de Paris) en J-1.

Deux garde-fous, vérifiés par tests/test_no_leakage.py :
1. Le registre déclare, pour chaque colonne brute, quand une valeur est publiée par rapport à son
   jour de livraison (RAW_PUBLICATION), et pour chaque feature, quelles colonnes et quel décalage
   en jours elle utilise (FEATURE_REGISTRY). Le test vérifie que chaque déclaration respecte
   la règle.
2. Le code est confronté au registre : on efface toutes les valeurs publiées après la fermeture
   (mask_unpublished) et on vérifie que les features du jour J ne changent pas.

Toutes les transformations sont sans état (rien n'est appris sur les données) : on peut les
appliquer à tout le jeu avant le découpage en périodes, sans fuite.
"""
import holidays
import numpy as np
import pandas as pd

TZ = "Europe/Paris"
GATE_CLOSURE_HOUR = 12  # fermeture SDAC : 12h00 heure de Paris, en J-1

# --------------------------------------------------------------------- registre des sources
# Colonne brute -> (jours avant la livraison, heure locale) à laquelle est publiée une valeur
# livrée le jour d. Exemple : un prix DA du jour d est publié vers 12h45 en d-1.
RAW_PUBLICATION = {
    "price_da": (1, 13),     # résultats SDAC publiés vers 12h45-13h00 en d-1
    "de_price_da": (1, 13),
    # Prévision de conso J-1 : publiée au plus tard 2 h avant la fermeture (règl. UE 543/2013, art. 6.1).
    "fr_load_fc": (1, 10),
    "de_load_fc": (1, 10),
    # Prévisions éolien/solaire J-1 : HYPOTHÈSE. L'échéance réglementaire est 18h en J-1 (art. 14.1),
    # mais les GRT (RTE, 50Hertz…) les publient en pratique avant la fermeture, et ENTSO-E ne garde
    # que la dernière version. Pratique standard en prévision de prix (cf. limites du README).
    "fr_wind_fc": (1, 12),
    "fr_solar_fc": (1, 12),
    "de_wind_fc": (1, 12),
    "de_solar_fc": (1, 12),
    # Gaz et carbone : déjà alignés dans get_data sur la dernière clôture datée au plus tard J-2.
    "ttf_d2": (2, 18),
    "eua_d2": (2, 18),
}

# Feature -> (colonnes brutes utilisées, décalage en jours de la livraison utilisée)
# Décalage 0 : valeur attachée au jour J lui-même (prévision J-1 pour J, TTF déjà aligné…).
FEATURE_REGISTRY = {
    # Calendrier : connu à l'avance, aucune source
    "hour": ([], 0),
    "dayofweek": ([], 0),
    "month": ([], 0),
    "is_holiday": ([], 0),
    "is_bridge": ([], 0),
    "is_dst_change": ([], 0),
    # Lags de prix France
    "price_d1": (["price_da"], 1),
    "price_d2": (["price_da"], 2),
    "price_d7": (["price_da"], 7),
    "price_d1_mean": (["price_da"], 1),
    "price_d1_min": (["price_da"], 1),
    "price_d1_max": (["price_da"], 1),
    # Fondamentaux France (prévisions J-1)
    "fr_load_fc": (["fr_load_fc"], 0),
    "fr_wind_fc": (["fr_wind_fc"], 0),
    "fr_solar_fc": (["fr_solar_fc"], 0),
    "fr_residual_fc": (["fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 0),
    "fr_vre_share": (["fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 0),
    # Combustibles et voisins
    "srmc_ccgt": (["ttf_d2", "eua_d2"], 0),
    "de_price_d1": (["de_price_da"], 1),
    "de_price_d1_mean": (["de_price_da"], 1),
    "de_load_fc": (["de_load_fc"], 0),
    "de_vre_fc": (["de_wind_fc", "de_solar_fc"], 0),
    "de_residual_fc": (["de_load_fc", "de_wind_fc", "de_solar_fc"], 0),
}


def is_known_at_gate_closure(column: str, lag_days: int) -> bool:
    """Une valeur de `column`, livrée `lag_days` jours avant J, est-elle publiée avant 12h en J-1 ?"""
    days_before, hour = RAW_PUBLICATION[column]
    # Publication en (J - lag_days - days_before) à `hour` ; fermeture en (J - 1) à 12h.
    offset = lag_days + days_before
    return offset > 1 or (offset == 1 and hour <= GATE_CLOSURE_HOUR)


# --------------------------------------------------------------------- marches de l'ablation
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
ALL_FEATURES = STEPS["B2"]

# Marches où la cible est prix − SRMC : les features en €/MWh sont alors exprimées en écart au SRMC
NORMALIZED_STEPS = {"B2"}
PRICE_LEVEL_FEATURES = ["price_d1", "price_d2", "price_d7", "price_d1_mean", "price_d1_min",
                        "price_d1_max", "de_price_d1", "de_price_d1_mean"]

# SRMC d'un CCGT, en €/MWh électrique
CCGT_EFFICIENCY = 0.55   # rendement électrique
CCGT_EMISSIONS = 0.37    # tCO2 par MWh électrique
CCGT_VOM = 2.0           # coûts variables hors combustible, €/MWh


# --------------------------------------------------------------------- calendrier
def _calendar_by_day(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Features calendaires journalières (jours fériés, ponts, changement d'heure).

    `days` : dates locales naïves (minuit), uniques.
    """
    years = range(days.year.min() - 1, days.year.max() + 2)  # marge pour les ponts en limite d'année
    fr_holidays = holidays.France(years=years)

    def is_off(d: pd.Timestamp) -> bool:
        return d.dayofweek >= 5 or d.date() in fr_holidays

    one_day = pd.Timedelta(days=1)
    is_holiday = [d.date() in fr_holidays for d in days]
    # Pont : jour ouvré coincé entre deux jours chômés (ex. vendredi après l'Ascension)
    is_bridge = [not is_off(d) and is_off(d - one_day) and is_off(d + one_day) for d in days]
    # Changement d'heure : la journée locale ne dure pas 24 h
    start = days.tz_localize(TZ)
    end = (days + one_day).tz_localize(TZ)
    is_dst_change = (end - start) != pd.Timedelta(hours=24)

    return pd.DataFrame({"is_holiday": is_holiday, "is_bridge": is_bridge,
                         "is_dst_change": is_dst_change}, index=days).astype("int8")


# --------------------------------------------------------------------- lags
def _same_hour_lag(values: pd.Series, day: pd.Series, hour: pd.Series, k: int) -> np.ndarray:
    """Valeur à la même heure locale, k jours plus tôt.

    On indexe par (jour local, heure locale) plutôt que de décaler de 24 h : un décalage de 24 h
    se tromperait d'une heure autour des changements d'heure. L'heure qui n'existe pas (2h le
    jour du passage à l'heure d'été) donne NaN ; l'heure doublée (octobre) est moyennée.
    """
    by_day_hour = values.groupby([day, hour]).mean()
    target = pd.MultiIndex.from_arrays([day - pd.Timedelta(days=k), hour])
    return by_day_hour.reindex(target).to_numpy()


def _daily_stats_lag(values: pd.Series, day: pd.Series, k: int, stats=("mean", "min", "max")):
    """Statistiques de la journée locale J-k (moyenne, min, max), recopiées sur chaque heure de J."""
    daily = values.groupby(day).agg(list(stats))
    daily.index = daily.index + pd.Timedelta(days=k)
    return daily.reindex(day.to_numpy())


# --------------------------------------------------------------------- construction
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute toutes les features (ALL_FEATURES) à une copie du DataFrame.

    Attend un index horaire et les colonnes de RAW_PUBLICATION, plus `time_local`.
    Les lags supposent un historique horaire continu : appliquer sur le jeu complet, pas par morceaux.
    """
    missing = [c for c in list(RAW_PUBLICATION) + ["time_local"] if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes manquantes : {missing}")

    out = df.copy()
    t = out["time_local"].dt.tz_convert(TZ)
    day = t.dt.tz_localize(None).dt.normalize()   # jour local, naïf
    hour = t.dt.hour

    # --- Calendrier
    out["hour"] = hour.astype("int8")
    out["dayofweek"] = t.dt.dayofweek.astype("int8")
    out["month"] = t.dt.month.astype("int8")
    cal = _calendar_by_day(pd.DatetimeIndex(day.unique()))
    for c in cal.columns:
        out[c] = cal[c].reindex(day.to_numpy()).to_numpy()

    # --- Lags de prix France
    for k in (1, 2, 7):
        out[f"price_d{k}"] = _same_hour_lag(out["price_da"], day, hour, k)
    stats = _daily_stats_lag(out["price_da"], day, 1)
    for s in ("mean", "min", "max"):
        out[f"price_d1_{s}"] = stats[s].to_numpy()

    # --- Fondamentaux France : charge résiduelle = ce que le non-pilotable laisse à couvrir
    fr_vre = out["fr_wind_fc"] + out["fr_solar_fc"]
    out["fr_residual_fc"] = out["fr_load_fc"] - fr_vre
    out["fr_vre_share"] = fr_vre / out["fr_load_fc"].where(out["fr_load_fc"] > 0)

    # --- Combustibles : coût marginal d'un CCGT
    out["srmc_ccgt"] = (out["ttf_d2"] / CCGT_EFFICIENCY
                        + out["eua_d2"] * CCGT_EMISSIONS + CCGT_VOM)

    # --- Allemagne
    out["de_price_d1"] = _same_hour_lag(out["de_price_da"], day, hour, 1)
    out["de_price_d1_mean"] = _daily_stats_lag(out["de_price_da"], day, 1, ("mean",))["mean"].to_numpy()
    out["de_vre_fc"] = out["de_wind_fc"] + out["de_solar_fc"]
    out["de_residual_fc"] = out["de_load_fc"] - out["de_vre_fc"]

    return out


def get_X(df: pd.DataFrame, step: str) -> pd.DataFrame:
    """Matrice de features d'une marche de l'ablation (DataFrame déjà passé par add_features).

    Sur les marches normalisées, les features en €/MWh sont exprimées en écart au SRMC du jour,
    comme la cible : sinon les entrées garderaient le problème de niveau que la cible n'a plus.
    """
    X = df[STEPS[step]].copy()
    if step in NORMALIZED_STEPS:
        for c in PRICE_LEVEL_FEATURES:
            if c in X:
                X[c] = X[c] - df["srmc_ccgt"]
    return X


def get_y(df: pd.DataFrame, step: str) -> pd.Series:
    """Cible : prix DA, ou prix − SRMC sur les marches normalisées."""
    if step in NORMALIZED_STEPS:
        return df["price_da"] - df["srmc_ccgt"]
    return df["price_da"]


def to_price(pred: np.ndarray, df: pd.DataFrame, step: str) -> np.ndarray:
    """Repasse une prédiction de la cible en prix (€/MWh)."""
    return pred + df["srmc_ccgt"].to_numpy() if step in NORMALIZED_STEPS else pred


# --------------------------------------------------------------------- contrôle de fuite
def gate_closure(delivery_day) -> pd.Timestamp:
    """Fermeture de l'enchère pour la livraison du jour `delivery_day` : 12h00 en J-1, heure de Paris."""
    d = pd.Timestamp(delivery_day).normalize()
    if d.tzinfo is not None:
        d = d.tz_localize(None)
    return (d - pd.Timedelta(days=1) + pd.Timedelta(hours=GATE_CLOSURE_HOUR)).tz_localize(TZ)


def mask_unpublished(df: pd.DataFrame, delivery_day) -> pd.DataFrame:
    """Copie où toute valeur publiée après la fermeture de l'enchère de `delivery_day` vaut NaN.

    Reconstitue ce qu'un prévisionniste voyait à 12h en J-1, d'après RAW_PUBLICATION.
    """
    cutoff = gate_closure(delivery_day)
    out = df.copy()
    day = out["time_local"].dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    for col, (days_before, hour) in RAW_PUBLICATION.items():
        published = (day - pd.Timedelta(days=days_before) + pd.Timedelta(hours=hour)).dt.tz_localize(TZ)
        out.loc[(published > cutoff).to_numpy(), col] = np.nan
    return out
