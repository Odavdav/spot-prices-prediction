"""Chargement du jeu horaire construit par get_data.py."""
import pandas as pd

from src.config import DATASET_PATH, TZ


def load_dataset(path=DATASET_PATH) -> pd.DataFrame:
    """Lit le Parquet horaire (index UTC) et garantit une colonne `time_local` en heure de Paris.

    Les heures sans prix sont conservées : elles servent au calcul des lags et sont écartées
    seulement au moment d'entraîner ou d'évaluer.
    """
    df = pd.read_parquet(path).sort_index()
    if not df.index.is_unique:
        raise ValueError("Index horaire dupliqué : reconstruire le jeu avec get_data.py.")
    df["time_local"] = df.index.tz_convert(TZ)
    return df
