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
    # Forme de la journée J (calculée sur les 24 h de prévisions pour J, connues avant la fermeture)
    "fr_rl_day_min": (["fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 0),
    "fr_rl_day_max": (["fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 0),
    "fr_rl_day_mean": (["fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 0),
    "fr_rl_rel": (["fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 0),
    "fr_rl_rank": (["fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 0),
    "fr_rl_min_hour": (["fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 0),
    "fr_solar_day_max": (["fr_solar_fc"], 0),
    "de_rl_day_mean": (["de_load_fc", "de_wind_fc", "de_solar_fc"], 0),
    "price_d1_rel": (["price_da"], 1),
    # Régime récent (fenêtres glissantes qui s'arrêtent à J-1)
    "spread_14d": (["price_da", "ttf_d2", "eua_d2"], 1),
    "vol_14d": (["price_da"], 1),
    "neg_share_14d": (["price_da"], 1),
    "rl_slope_28d": (["price_da", "fr_load_fc", "fr_wind_fc", "fr_solar_fc"], 1),
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
# Extension post-test (inspirée de Chronos-2) : forme de la journée et régime récent
DAY_SHAPE = ["fr_rl_day_min", "fr_rl_day_max", "fr_rl_day_mean", "fr_rl_rel", "fr_rl_rank",
             "fr_rl_min_hour", "fr_solar_day_max", "de_rl_day_mean", "price_d1_rel"]
REGIME = ["spread_14d", "vol_14d", "neg_share_14d", "rl_slope_28d"]
STEPS["B3"] = STEPS["B2"] + DAY_SHAPE + REGIME
ALL_FEATURES = STEPS["B3"]
# Variante de contrôle : mêmes features que B2, cible prix brut (mesure l'apport de la normalisation)
STEPS["B2_brut"] = STEPS["B2"]
# Variante : normalisation pondérée par la charge résiduelle (cible prix − w(RL) × SRMC)
STEPS["B2_rl"] = STEPS["B2"]

# Normalisation de la cible et des features en €/MWh, par marche :
#   "full" : prix − SRMC ;  "rl" : prix − w(RL) × SRMC ;  absente : prix brut.
NORMALIZATION = {"B2": "full", "B2_rl": "rl", "B3": "full"}
NORMALIZED_STEPS = set(NORMALIZATION)
PRICE_LEVEL_FEATURES = ["price_d1", "price_d2", "price_d7", "price_d1_mean", "price_d1_min",
                        "price_d1_max", "de_price_d1", "de_price_d1_mean"]

# Transmission du coût du gaz au prix français selon la charge résiduelle prévue (FR, MW) :
# pente du prix sur le SRMC par décile de charge résiduelle, avec effets fixes heure et mois,
# estimée sur les données d'avant la validation (oct. 2018 -> oct. 2024), donc sans fuite.
# Lecture : sous ~35 GW, le gaz fixe rarement le prix mais pèse encore (0,7 : couplage avec
# les voisins, valeur de l'eau) ; au-delà de ~40 GW, transmission d'environ 1,1-1.2.
PASS_THROUGH_RL = [28261, 33174, 36055, 38614, 41098, 43760, 47089, 51881, 58135, 68107]
PASS_THROUGH = [0.71, 0.86, 0.95, 1.04, 1.13, 1.20, 1.18, 1.05, 1.18, 1.47]

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

    # --- Forme de la journée J : chaque heure « voit » le profil des prévisions de toute la journée
    keys = day.to_numpy()
    rl = out["fr_residual_fc"]
    g = rl.groupby(keys)
    out["fr_rl_day_min"] = g.transform("min").to_numpy()
    out["fr_rl_day_max"] = g.transform("max").to_numpy()
    out["fr_rl_day_mean"] = g.transform("mean").to_numpy()
    out["fr_rl_rel"] = rl - out["fr_rl_day_mean"]                 # position de l'heure dans la journée
    out["fr_rl_rank"] = g.rank(pct=True).to_numpy()
    tmp = pd.DataFrame({"rl": rl.to_numpy(), "hour": hour.to_numpy(), "day": keys}).dropna()
    min_hour = tmp.loc[tmp.groupby("day")["rl"].idxmin()].set_index("day")["hour"]
    out["fr_rl_min_hour"] = min_hour.reindex(keys).to_numpy()    # heure du creux (souvent midi solaire)
    out["fr_solar_day_max"] = out["fr_solar_fc"].groupby(keys).transform("max").to_numpy()
    out["de_rl_day_mean"] = out["de_residual_fc"].groupby(keys).transform("mean").to_numpy()
    out["price_d1_rel"] = out["price_d1"] - out["price_d1_mean"]  # forme de la courbe de la veille

    # --- Régime récent : statistiques des jours J-1 à J-k (décalage d'un jour : rien de J lui-même)
    for c, v in _recent_regime(out, day).items():
        out[c] = v.reindex(keys).to_numpy()

    return out


def _recent_regime(out: pd.DataFrame, day: pd.Series) -> dict:
    """Indicateurs journaliers du régime récent, calculés sur les jours strictement antérieurs."""
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
        "spread_14d": r14["spread"].mean().shift(1),     # écart moyen prix − SRMC : le gaz fixe-t-il le prix ?
        "vol_14d": r14["price"].std().shift(1),          # volatilité des prix journaliers moyens
        "neg_share_14d": (r14["neg"].sum() / r14["hours"].sum()).shift(1),
        "rl_slope_28d": slope,                           # €/MWh par GW de charge résiduelle, 28 derniers jours
    }


def gas_weight(df: pd.DataFrame) -> pd.Series:
    """Poids w(RL) du SRMC : transmission estimée du coût du gaz, interpolée selon la charge
    résiduelle prévue (constante au-delà des déciles extrêmes ; 1 si la prévision manque)."""
    rl = df["fr_residual_fc"]
    w = np.interp(rl.fillna(np.nanmedian(PASS_THROUGH_RL)), PASS_THROUGH_RL, PASS_THROUGH)
    return pd.Series(np.where(rl.notna(), w, 1.0), index=df.index)


def srmc_offset(df: pd.DataFrame, step: str) -> pd.Series:
    """Montant retranché au prix (cible et features en €/MWh) selon la normalisation de la marche."""
    mode = NORMALIZATION.get(step)
    if mode == "full":
        return df["srmc_ccgt"]
    if mode == "rl":
        return gas_weight(df) * df["srmc_ccgt"]
    return pd.Series(0.0, index=df.index)


def get_X(df: pd.DataFrame, step: str) -> pd.DataFrame:
    """Matrice de features d'une marche de l'ablation (DataFrame déjà passé par add_features).

    Sur les marches normalisées, les features en €/MWh sont exprimées en écart au même montant que
    la cible : sinon les entrées garderaient le problème de niveau que la cible n'a plus.
    """
    X = df[STEPS[step]].copy()
    if step in NORMALIZED_STEPS:
        offset = srmc_offset(df, step)
        for c in PRICE_LEVEL_FEATURES:
            if c in X:
                X[c] = X[c] - offset
    return X


def get_y(df: pd.DataFrame, step: str) -> pd.Series:
    """Cible : prix DA, moins le SRMC (éventuellement pondéré) sur les marches normalisées."""
    return df["price_da"] - srmc_offset(df, step)


def to_price(pred: np.ndarray, df: pd.DataFrame, step: str) -> np.ndarray:
    """Repasse une prédiction de la cible en prix (€/MWh)."""
    return pred + srmc_offset(df, step).to_numpy()


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
