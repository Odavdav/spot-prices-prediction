"""Chargement du jeu figé. Les contrôles (fuseaux, trous, doublons) pourront être ajoutés ici."""
import pandas as pd

from src.config import DATASET_PATH, GROUP, TARGET, TZ


def load_dataset(path=DATASET_PATH) -> pd.DataFrame:
    """Lit le Parquet, retire les heures sans prix ni cible, recalcule l'heure et le jour locaux.

    Les NaN des prévisions (solaire, éolien FR) sont conservés : ils sont imputés dans le Pipeline.
    """
    df = pd.read_parquet(path)
    df = df.dropna(subset=["price_da", TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    if "time_local" not in df.columns:
        df["time_local"] = df.index.tz_convert(TZ)
    df["time_local"] = df["time_local"].dt.tz_convert(TZ)
    # Jour local en texte (AAAA-MM-JJ) : clé de groupe, lisible dans splits.json
    df[GROUP] = df["time_local"].dt.strftime("%Y-%m-%d")
    return df.sort_values("time_local")
