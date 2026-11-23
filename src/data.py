"""Loading of the hourly dataset built by get_data.py."""
import pandas as pd

from src.config import DATASET_PATH, TZ


def load_dataset(path=DATASET_PATH) -> pd.DataFrame:
    """Read the hourly Parquet file (UTC index) and add a `time_local` column in Paris time.

    Hours without a price are kept: they are needed to compute lags and are only dropped
    when training or evaluating.
    """
    df = pd.read_parquet(path).sort_index()
    if not df.index.is_unique:
        raise ValueError("Duplicated hourly index: rebuild the dataset with get_data.py.")
    df["time_local"] = df.index.tz_convert(TZ)
    return df
