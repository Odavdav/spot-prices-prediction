"""
Usage :
    cp .env.example .env  puis y renseigner ENTSOE_API_KEY (lu automatiquement)
    python get_data.py                       # 2015-01-01 -> aujourd'hui
    python get_data.py --start 2019-01-01 --end 2026-01-01

Sorties :
    data/raw/<jeu>_<zone>_<année>.parquet    cache brut (un fichier par appel annuel)
    data/processed/dataset_hourly.parquet    jeu horaire fusionné, index UTC
"""
import argparse
import os
import time
from pathlib import Path
from src.config import DATA_PROCESSED, DATASET_PATH, ROOT

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

RAW = ROOT / "data" / "raw"
OUT = DATA_PROCESSED
TZ = "Europe/Paris"

FR = "FR"
# 'DE' = Allemagne (zone de contrôle nationale), disponible sur toute la période.
# La zone de prix DE_LU n'existe que depuis le 01/10/2018 : on ne l'utilise pas pour les fondamentaux.
DE = "DE"


# ---------------------------------------------------------------- téléchargement
def yearly_chunks(start: pd.Timestamp, end: pd.Timestamp):
    """Découpe [start, end) en tranches annuelles (une requête et un fichier cache par tranche)."""
    cur = start
    while cur < end:
        nxt = min(pd.Timestamp(year=cur.year + 1, month=1, day=1, tz=TZ), end)
        yield cur, nxt
        cur = nxt


def fetch(client, name, zone, start, end, retries=3):
    """Télécharge un jeu par tranches annuelles, avec cache local et reprises en cas d'erreur réseau."""
    parts = []
    for s, e in yearly_chunks(start, end):
        cache = RAW / f"{name}_{zone}_{s.year}.parquet"
        is_current_year = e >= pd.Timestamp.now(tz=TZ).normalize()
        if cache.exists() and not is_current_year:  # l'année en cours est toujours re-téléchargée
            parts.append(pd.read_parquet(cache))
            continue
        for attempt in range(1, retries + 1):
            try:
                df = QUERIES[name](client, zone, s, e)
                if isinstance(df, pd.Series):
                    df = df.to_frame(name)
                df.to_parquet(cache)
                parts.append(df)
                print(f"  ok   {name:<16} {zone:<3} {s.year}  ({len(df)} lignes)")
                break
            except NoMatchingDataError:
                print(f"  vide {name:<16} {zone:<3} {s.year}")
                break
            except Exception as exc:  # erreurs HTTP / timeouts : on réessaie
                print(f"  err  {name:<16} {zone:<3} {s.year}  essai {attempt}/{retries} : {exc}")
                time.sleep(5 * attempt)
        time.sleep(0.5)  # reste très en dessous de la limite de 400 requêtes/min
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts)
    return df[~df.index.duplicated(keep="last")].sort_index()


QUERIES = {
    "price_da": lambda c, z, s, e: c.query_day_ahead_prices(z, start=s, end=e),
    "load_fc": lambda c, z, s, e: c.query_load_forecast(z, start=s, end=e, process_type="A01"),  # A01 = day-ahead
    "wind_solar_fc": lambda c, z, s, e: c.query_wind_and_solar_forecast(z, start=s, end=e, process_type="A01"),
    "capacity": lambda c, z, s, e: c.query_installed_generation_capacity(z, start=s, end=e),
}


# ---------------------------------------------------------------- mise en forme
def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Passe en UTC puis moyenne horaire (gère les pas de 15 min, notamment les prix depuis le 01/10/2025)."""
    if df.empty:
        return df
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df.resample("1h").mean()


def pick(df: pd.DataFrame, *keywords) -> pd.Series:
    """Somme des colonnes dont le nom contient l'un des mots-clés (les noms varient selon la zone)."""
    cols = [c for c in df.columns if any(k.lower() in str(c).lower() for k in keywords)]
    if not cols:
        return pd.Series(index=df.index, dtype=float)
    return df[cols].sum(axis=1, min_count=1)


def build_dataset(raw: dict) -> pd.DataFrame:
    price = to_hourly(raw["price_da"])
    out = pd.DataFrame(index=price.index)
    out["price_da"] = price.iloc[:, 0]

    for zone, tag in [(FR, "fr"), (DE, "de")]:
        load = to_hourly(raw[f"load_fc_{zone}"])
        ws = to_hourly(raw[f"wind_solar_fc_{zone}"])
        out[f"{tag}_load_fc"] = pick(load, "forecast", "load") if not load.empty else float("nan")
        out[f"{tag}_solar_fc"] = pick(ws, "solar") if not ws.empty else float("nan")
        out[f"{tag}_wind_fc"] = pick(ws, "wind") if not ws.empty else float("nan")

    cap = raw["capacity_FR"]
    if not cap.empty:
        cap = cap.copy()
        cap.index = pd.DatetimeIndex(cap.index).tz_convert("UTC")
        solar_cap = pick(cap, "solar").rename("fr_solar_capacity")
        out = out.join(solar_cap.reindex(out.index, method="ffill"))

    out["target"] = (out["price_da"] < 0).astype("Int8").where(out["price_da"].notna())  # NA si prix absent
    out["time_local"] = out.index.tz_convert(TZ)  # pour les features calendaires et le groupe « jour »
    out["date_local"] = out["time_local"].dt.date
    return out


def report(df: pd.DataFrame):
    print("\n=== Résumé ===")
    print(f"Période : {df.index.min()} -> {df.index.max()}  ({len(df)} heures)")
    print(f"Heures à prix négatif : {int(df['target'].sum())} ({df['target'].mean():.2%})")
    print("\nTaux de valeurs manquantes par colonne :")
    print(df.drop(columns=["time_local", "date_local"]).isna().mean().map("{:.1%}".format).to_string())
    print("\nHeures négatives par an :")
    print(df.groupby(df["time_local"].dt.year)["target"].sum().to_string())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None, help="exclu ; par défaut demain 00:00")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise SystemExit("ENTSOE_API_KEY absente : la renseigner dans .env (voir .env.example).")

    start = pd.Timestamp(args.start, tz=TZ)
    end = pd.Timestamp(args.end, tz=TZ) if args.end else pd.Timestamp.now(tz=TZ).normalize() + pd.Timedelta(days=1)
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    client = EntsoePandasClient(api_key=key)
    raw = {"price_da": fetch(client, "price_da", FR, start, end)}
    for zone in (FR, DE):
        raw[f"load_fc_{zone}"] = fetch(client, "load_fc", zone, start, end)
        raw[f"wind_solar_fc_{zone}"] = fetch(client, "wind_solar_fc", zone, start, end)
    raw["capacity_FR"] = fetch(client, "capacity", FR, start, end)

    df = build_dataset(raw)
    df.to_parquet(DATASET_PATH)
    report(df)
    print(f"\nÉcrit : {OUT / 'dataset_hourly.parquet'}")


if __name__ == "__main__":
    main()
