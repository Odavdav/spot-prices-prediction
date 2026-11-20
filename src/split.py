"""Découpage train / test interne / test hors échantillon, figé dans splits.json.

- Test hors échantillon (OOS) : tout ce qui suit OOS_START (marché au quart d'heure).
- Test interne : 20 % des jours d'avant OOS_START, tirés par StratifiedGroupKFold (un fold mis de côté).
- CV : les 80 % restants, répartis en N_SPLITS folds, toujours par jour.

Usage : python -m src.split   (à lancer une seule fois, puis commiter splits.json)
"""
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from src.config import GROUP, N_SPLITS, OOS_START, SEED, SPLITS_PATH, TARGET, TZ


def _oos_mask(df: pd.DataFrame, oos_start: str) -> pd.Series:
    return df["time_local"] >= pd.Timestamp(oos_start, tz=TZ)


def make_splits(df: pd.DataFrame, oos_start=OOS_START, n_splits=N_SPLITS, seed=SEED) -> dict:
    """Calcule les jours de test et l'affectation de chaque jour d'entraînement à un fold de CV."""
    dev = df.loc[~_oos_mask(df, oos_start)]
    days = dev[GROUP].astype(str).to_numpy()
    y = dev[TARGET].astype(int).to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # 1) Test interne : premier fold (≈ 1/n_splits des jours)
    _, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups=days))
    test_days = sorted(set(days[test_idx]))

    # 2) Folds de CV sur les jours restants
    is_train = ~np.isin(days, test_days)
    days_tr, y_tr = days[is_train], y[is_train]
    cv_folds = {}
    for k, (_, val_idx) in enumerate(sgkf.split(np.zeros(len(y_tr)), y_tr, groups=days_tr)):
        for d in set(days_tr[val_idx]):
            cv_folds[d] = k

    oos = df.loc[_oos_mask(df, oos_start)]
    summary = {
        name: {"n_rows": int(len(part)), "n_days": int(part[GROUP].nunique()),
               "positive_rate": round(float(part[TARGET].mean()), 5)}
        for name, part in [("train", dev.loc[is_train]), ("test", dev.loc[~is_train]), ("oos", oos)]
    }

    return {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "seed": seed,
            "n_splits": n_splits,
            "n_rows_dataset": int(len(df)),
            "summary": summary,
        },
        "oos_start": oos_start,
        "test_days": test_days,
        "cv_folds": cv_folds,
    }


def save_splits(splits: dict, path=SPLITS_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=1)


def load_splits(path=SPLITS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def split_frames(df: pd.DataFrame, splits: dict) -> dict:
    """Renvoie {"train", "test", "oos"} à partir du DataFrame et du découpage figé."""
    if len(df) != splits["meta"]["n_rows_dataset"]:
        raise ValueError(
            f"Le jeu a {len(df)} lignes, splits.json a été fait sur "
            f"{splits['meta']['n_rows_dataset']} : le dataset a changé, régénérer le split."
        )
    oos = _oos_mask(df, splits["oos_start"])
    is_test = df[GROUP].isin(set(splits["test_days"]))
    return {
        "train": df.loc[~oos & ~is_test],
        "test": df.loc[~oos & is_test],
        "oos": df.loc[oos],
    }


def cv_indices(train: pd.DataFrame, splits: dict) -> list:
    """Liste de (indices_train, indices_val) en positions, à passer en cv= de cross_validate,
    GridSearchCV, RandomizedSearchCV ou cross_val_predict. Tous les modèles voient ainsi les
    mêmes folds, et il n'y a plus besoin de passer groups= à .fit().
    """
    folds = train[GROUP].map(splits["cv_folds"])
    if folds.isna().any():
        raise ValueError("Des jours du train n'ont pas de fold : régénérer splits.json.")
    folds = folds.to_numpy().astype(int)
    return [
        (np.flatnonzero(folds != k), np.flatnonzero(folds == k))
        for k in range(splits["meta"]["n_splits"])
    ]


if __name__ == "__main__":
    from src.data import load_dataset

    df = load_dataset()
    splits = make_splits(df)
    save_splits(splits)
    print(f"splits.json écrit dans {SPLITS_PATH}")
    for name, s in splits["meta"]["summary"].items():
        print(f"  {name:5s} : {s['n_rows']:6d} heures, {s['n_days']:5d} jours, "
              f"{100 * s['positive_rate']:.2f} % d'heures négatives")
