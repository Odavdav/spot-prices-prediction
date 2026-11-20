"""Feature engineering.

Règle anti-fuite : toutes les features dérivent de prévisions J-1 ou du calendrier, donc elles sont
connues à J-1 12 h. Les transformations sont sans état (rien n'est appris sur les données) : on peut
les appliquer à tout le jeu avant le split sans fuite. L'imputation et la standardisation, elles,
apprennent des statistiques et restent dans le Pipeline sklearn.
"""
import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from src.config import TARGET

# Colonnes brutes attendues dans le Parquet (prévisions J-1 ENTSO-E, en MW)
RAW_COLUMNS = [
    "fr_load_fc",
    "fr_solar_fc",
    "fr_wind_fc",        # terrestre + en mer
    "de_load_fc",
    "de_solar_fc",
    "de_wind_fc",
    "fr_solar_capacity",  # sert uniquement à normaliser, pas une feature (proxy de l'année)
]

# Les 16 features du modèle
FEATURES = [
    # France
    "fr_load_fc",
    "fr_solar_fc",
    "fr_wind_fc",
    "fr_residual_fc",
    "fr_vre_share",
    "fr_solar_cf",
    # Allemagne-Luxembourg
    "de_load_fc",
    "de_vre_fc",
    "de_residual_fc",
    "de_vre_share",
    # Couplage
    "fr_de_residual_sum",
    # Calendrier
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "is_offday",
]

ENGINEERED = [f for f in FEATURES if f not in RAW_COLUMNS]

# Tout ce qui ne doit jamais entrer dans X (réalisé, conséquences du prix, tendance)
FORBIDDEN = ["price_da", TARGET, "year", "fr_solar_capacity"]
FORBIDDEN_SUBSTRINGS = ["actual", "realised", "flow", "exchange", "pump", "co2", "price"]


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    """Division qui renvoie NaN si le dénominateur est nul ou négatif."""
    return num / den.where(den > 0)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features construites à une copie du DataFrame.

    Attend les colonnes de RAW_COLUMNS et une colonne `time_local` (datetime tz Europe/Paris).
    """
    missing = [c for c in RAW_COLUMNS + ["time_local"] if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes manquantes : {missing}")

    out = df.copy()
    t = out["time_local"]

    # --- France : charge résiduelle = ce que le non-pilotable laisse à couvrir
    fr_vre = out["fr_solar_fc"] + out["fr_wind_fc"]
    out["fr_residual_fc"] = out["fr_load_fc"] - fr_vre
    out["fr_vre_share"] = _safe_div(fr_vre, out["fr_load_fc"])
    # Facteur de charge solaire : neutralise la croissance du parc installé
    out["fr_solar_cf"] = _safe_div(out["fr_solar_fc"], out["fr_solar_capacity"])

    # --- Allemagne-Luxembourg
    out["de_vre_fc"] = out["de_solar_fc"] + out["de_wind_fc"]
    out["de_residual_fc"] = out["de_load_fc"] - out["de_vre_fc"]
    out["de_vre_share"] = _safe_div(out["de_vre_fc"], out["de_load_fc"])

    # --- Couplage : surplus cumulé au cœur de la zone couplée
    out["fr_de_residual_sum"] = out["fr_residual_fc"] + out["de_residual_fc"]

    # --- Calendrier (heure locale). Encodage cyclique : 23 h et 0 h deviennent voisines.
    hour = t.dt.hour
    month = t.dt.month
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    # Jour off : samedi, dimanche ou jour férié national
    years = range(int(t.dt.year.min()), int(t.dt.year.max()) + 1)
    fr_holidays = holidays.country_holidays("FR", years=years)
    is_holiday = t.dt.date.map(lambda d: d in fr_holidays).astype(bool)
    out["is_offday"] = ((t.dt.dayofweek >= 5) | is_holiday).astype("int8")

    return out


def get_X_y(df: pd.DataFrame, features=FEATURES):
    """Renvoie X (features seulement) et y (0/1) à partir d'un DataFrame déjà passé par add_features."""
    check_no_leakage(features)
    return df[features], df[TARGET].astype(int)


def check_no_leakage(features) -> None:
    """Lève une erreur si une colonne interdite se glisse dans la liste des features."""
    bad = [
        f for f in features
        if f in FORBIDDEN or any(s in f.lower() for s in FORBIDDEN_SUBSTRINGS)
    ]
    if bad:
        raise ValueError(f"Features interdites (fuite) : {bad}")


def _build_X(df: pd.DataFrame) -> pd.DataFrame:
    return add_features(df)[FEATURES]


def make_feature_transformer() -> FunctionTransformer:
    """Étape de Pipeline qui part des colonnes brutes + time_local.

    Utile pour un pipeline complet « données brutes → prédiction ». Pour permutation_importance
    sur les 16 features, préférer add_features() en amont puis get_X_y().
    """
    return FunctionTransformer(_build_X, validate=False)
