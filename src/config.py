"""Constantes partagées par tout le projet : chemins, périodes d'évaluation, graine."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
DATASET_PATH = DATA_PROCESSED / "dataset_hourly.parquet"
PREDICTIONS_DIR = DATA_PROCESSED / "predictions"   # non versionné
RESULTS = ROOT / "results"                          # métriques CSV et figures, versionnés

TZ = "Europe/Paris"

# Périodes (dates de livraison, heure de Paris, borne de fin exclue).
# Le test est entièrement après le passage du DA au pas de 15 min (01/10/2025) et n'est touché
# qu'à la fin ; la validation sert à régler les choix de modélisation.
PERIODS = {
    "validation": ("2024-11-01", "2025-11-01"),
    "test": ("2025-11-01", "2026-11-01"),
}

SEED = 42
