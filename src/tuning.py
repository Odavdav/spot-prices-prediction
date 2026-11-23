"""Hyperparameter tuning of LightGBM, on the validation year only.

Protocol (fixed before the search):
1. Random search over SPACE for step B2 with weekly retraining, on the validation year.
   Objective: MAE on all hours. Trial 0 is the current setting (LGB_PARAMS).
2. The best configuration is applied unchanged to the daily models on validation and test:
   B2_daily_tuned, and B3_daily_hl90_tuned (then its error correction). Their test figures
   are reported whatever they are.

Usage:
    python -m src.tuning search --worker 0 --workers 3 --until 03:30   # one worker of the search
    python -m src.tuning summary                                       # results/tuning_validation.csv
    python -m src.tuning final --period test --model B2_daily_tuned
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

from src.backtest import LGB_PARAMS, save_prediction, walk_forward
from src.config import DATA_PROCESSED, PERIODS, RESULTS, SEED, TZ
from src.data import load_dataset
from src.features import add_features

TRIALS_DIR = DATA_PROCESSED / "tuning"   # one JSON per trial, so that workers never collide
N_JOBS = 2                               # faster than all cores on this machine (efficiency cores)
FINAL_MODELS = {"B2_daily_tuned": ("B2", None), "B3_daily_hl90_tuned": ("B3", 90)}


def sample(trial: int) -> dict:
    """Configuration of a trial (reproducible: one random generator per trial)."""
    if trial == 0:
        return {**LGB_PARAMS, "n_jobs": N_JOBS}
    rng = np.random.default_rng(SEED + trial)
    log_uniform = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    lr = log_uniform(0.01, 0.1)
    return {**LGB_PARAMS, "n_jobs": N_JOBS,
            "num_leaves": int(round(log_uniform(15, 255))),
            "min_child_samples": int(round(log_uniform(10, 500))),
            "learning_rate": lr,
            "n_estimators": int(round(min(3000, log_uniform(15, 60) / lr), -1)),  # lr × trees in [15, 60]
            "subsample": float(rng.uniform(0.5, 1.0)),
            "colsample_bytree": float(rng.uniform(0.4, 1.0)),
            "reg_lambda": float(rng.choice([0.0, log_uniform(0.01, 10)]))}


def run_trial(f: pd.DataFrame, trial: int) -> dict:
    params = sample(trial)
    start, end = PERIODS["validation"]
    t0 = time.time()
    pred = walk_forward(f, "B2", start, end, refit="weekly", params=params, verbose=False)
    err = (pred - f.loc[pred.index, "price_da"]).abs()
    negative = f.loc[pred.index, "price_da"] < 0
    return {"trial": trial, "mae": err.mean(), "mae_negative": err[negative].mean(),
            "minutes": (time.time() - t0) / 60, "params": params}


def search(worker: int, workers: int, until: str):
    """Run trials worker, worker + workers, … until the `until` clock time (HH:MM, next occurrence)."""
    now = pd.Timestamp.now(tz=TZ)
    deadline = now.normalize() + pd.Timedelta(until + ":00")
    if deadline <= now:
        deadline += pd.Timedelta(days=1)
    TRIALS_DIR.mkdir(parents=True, exist_ok=True)
    f = add_features(load_dataset())
    trial = worker
    while pd.Timestamp.now(tz=TZ) < deadline:
        path = TRIALS_DIR / f"trial_{trial:04d}.json"
        if not path.exists():
            r = run_trial(f, trial)
            path.write_text(json.dumps(r))
            print(f"  trial {trial:4d}: MAE {r['mae']:.3f} (negative {r['mae_negative']:.2f}) "
                  f"in {r['minutes']:.1f} min", flush=True)
        trial += workers


def load_trials() -> pd.DataFrame:
    rows = [json.loads(p.read_text()) for p in sorted(TRIALS_DIR.glob("trial_*.json"))]
    out = pd.DataFrame([{**{k: v for k, v in r.items() if k != "params"}, **r["params"]} for r in rows])
    return out.sort_values("mae").reset_index(drop=True)


def best_params() -> dict:
    rows = [json.loads(p.read_text()) for p in TRIALS_DIR.glob("trial_*.json")]
    return min(rows, key=lambda r: r["mae"])["params"]


def summary():
    t = load_trials()
    keep = ["trial", "mae", "mae_negative", "minutes", "num_leaves", "min_child_samples", "learning_rate",
            "n_estimators", "subsample", "colsample_bytree", "reg_lambda"]
    RESULTS.mkdir(exist_ok=True)
    t[keep].round(4).to_csv(RESULTS / "tuning_validation.csv", index=False)
    base = t.loc[t["trial"] == 0, "mae"]
    print(f"{len(t)} trials; best {t['mae'].iloc[0]:.3f} (trial {t['trial'].iloc[0]}), "
          f"current setting {base.iloc[0]:.3f}" if len(base) else f"{len(t)} trials")
    print(t[keep].head(10).round(3).to_string(index=False))


def final(period: str, model: str):
    step, half_life = FINAL_MODELS[model]
    f = add_features(load_dataset())
    pred = walk_forward(f, step, *PERIODS[period], refit="daily", half_life=half_life, params=best_params())
    save_prediction(period, pred.rename(model))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--worker", type=int, default=0)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--until", default="03:30")
    sub.add_parser("summary")
    fi = sub.add_parser("final")
    fi.add_argument("--period", choices=list(PERIODS), required=True)
    fi.add_argument("--model", choices=list(FINAL_MODELS), required=True)
    args = ap.parse_args()
    if args.cmd == "search":
        search(args.worker, args.workers, args.until)
    elif args.cmd == "summary":
        summary()
    else:
        final(args.period, args.model)


if __name__ == "__main__":
    main()
