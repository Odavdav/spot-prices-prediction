"""Constantes partagées par tout le projet : chemins, dates de coupure, graine."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
DATASET_PATH = DATA_PROCESSED / "dataset_hourly.parquet"
SPLITS_PATH = DATA_PROCESSED / "splits.json"
FIGURES = ROOT / "reports" / "figures"

TZ = "Europe/Paris"
# Passage du day-ahead au quart d'heure : tout ce qui suit sert de test hors échantillon.
OOS_START = "2025-10-01"

SEED = 42
N_SPLITS = 5

TARGET = "target"
GROUP = "date_local"  # un groupe = un jour local, pour ne jamais couper une journée
