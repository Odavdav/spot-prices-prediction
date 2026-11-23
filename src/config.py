"""Constants shared by the whole project: paths, evaluation periods, seed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
DATASET_PATH = DATA_PROCESSED / "dataset_hourly.parquet"
PREDICTIONS_DIR = DATA_PROCESSED / "predictions"   # not versioned
RESULTS = ROOT / "results"                          # CSV metrics and figures, versioned

TZ = "Europe/Paris"

# Periods (delivery days, Paris time, end excluded).
# The test year lies entirely after the switch of the day-ahead market to 15-min products (2025-10-01)
# and is only used at the end; the validation year is used for modelling choices.
PERIODS = {
    "validation": ("2024-11-01", "2025-11-01"),
    "test": ("2025-11-01", "2026-11-01"),
    # Robustness check: the gas crisis (TTF from 47 to over 300 €/MWh), to measure what the SRMC
    # normalisation brings when the price level shifts abruptly.
    "crisis": ("2021-07-01", "2023-01-01"),
    # Burn-in: out-of-sample forecasts for the year before validation, used as the error history
    # of the correction model (src/correction.py).
    "burn_in": ("2023-11-01", "2024-11-01"),
}

SEED = 42
