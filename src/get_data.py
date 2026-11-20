"""
Usage :
    cp .env.example .env  puis y renseigner ENTSOE_API_KEY (lu automatiquement)
    python -m src.get_data                   # 2018-10-01 -> aujourd'hui
    python -m src.get_data --start 2019-01-01 --end 2026-01-01

Sorties :
    data/raw/<jeu>_<zone>_<année>.parquet    cache brut ENTSO-E (un fichier par appel annuel)
    data/raw/ttf_yahoo.parquet               clôtures journalières TTF front-month (Yahoo TTF=F)
    data/processed/dataset_hourly.parquet    jeu horaire fusionné, index UTC

Ensemble d'information : les prix (FR, DE) sont stockés bruts, à l'heure de livraison ; les lags sont
construits dans features.py. Le TTF est stocké déjà aligné (`ttf_d2` = dernière clôture datée au plus
tard J-2 pour la livraison J), car la clôture de J-1 n'est pas connue à la fermeture SDAC (12h J-1).
"""
import argparse
import os
import time
from pathlib import Path
from src.config import DATA_PROCESSED, DATASET_PATH, ROOT

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

RAW = ROOT / "data" / "raw"
OUT = DATA_PROCESSED
TZ = "Europe/Paris"

FR = "FR"
# 'DE' = Allemagne (zone de contrôle nationale) pour les prévisions de conso et d'ENR.
DE = "DE"
# Zone de prix Allemagne-Luxembourg, créée le 01/10/2018 (scission de DE-AT-LU) : début du dataset.
DE_LU = "DE_LU"
DEFAULT_START = "2018-10-01"
TTF_TICKER = "TTF=F"


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
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if name == "price_da":
        df = patch_missing_days(client, zone, df)
    return df


def patch_missing_days(client, zone, df: pd.DataFrame) -> pd.DataFrame:
    """Re-télécharge, par requêtes d'une journée, les jours où des heures de prix manquent.

    Les requêtes annuelles d'entsoe-py perdent l'heure du 31/12 à 00h (vérifié contre des requêtes
    courtes : le reste de l'année est identique). Une requête courte la restitue.
    """
    s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    hourly = s.tz_convert("UTC").resample("1h").mean()
    missing = hourly.index[hourly.isna()]
    days = sorted({t.tz_convert(TZ).normalize() for t in missing})
    for d in days:
        try:
            fix = QUERIES["price_da"](client, zone, d, d + pd.Timedelta(days=1))
        except NoMatchingDataError:
            print(f"  trou price_da {zone} {d.date()} : pas de données ENTSO-E")
            continue
        fix = fix.to_frame(df.columns[0]) if isinstance(fix, pd.Series) else fix
        df = df.combine_first(fix.tz_convert(df.index.tz))
        print(f"  fix  price_da {zone} {d.date()}")
        time.sleep(0.5)
    return df


QUERIES = {
    "price_da": lambda c, z, s, e: c.query_day_ahead_prices(z, start=s, end=e),
    "load_fc": lambda c, z, s, e: c.query_load_forecast(z, start=s, end=e, process_type="A01"),  # A01 = day-ahead
    "wind_solar_fc": lambda c, z, s, e: c.query_wind_and_solar_forecast(z, start=s, end=e, process_type="A01"),
    "capacity": lambda c, z, s, e: c.query_installed_generation_capacity(z, start=s, end=e),
}


def fetch_ttf(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Clôtures journalières TTF front-month (Yahoo), indexées par date de cotation.

    On remonte 30 jours avant `start` pour pouvoir aligner (forward-fill) les premiers jours.
    """
    cache = RAW / "ttf_yahoo.parquet"
    df = yf.download(TTF_TICKER, start=(start - pd.Timedelta(days=30)).date(), end=end.date(),
                     progress=False, auto_adjust=False)
    if df.empty:
        if cache.exists():  # Yahoo indisponible : on retombe sur le dernier téléchargement
            print("  err  TTF Yahoo vide, utilisation du cache")
            return pd.read_parquet(cache)["ttf"]
        raise SystemExit("Téléchargement TTF impossible (Yahoo) et pas de cache.")
    close = df["Close"].squeeze().rename("ttf").dropna()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close.to_frame().to_parquet(cache)
    print(f"  ok   TTF {TTF_TICKER:<12} {close.index.min().date()} -> {close.index.max().date()}"
          f"  ({len(close)} clôtures)")
    return close


def read_eua(path=None) -> pd.Series:
    """Clôtures journalières EUA (contrat de décembre, Investing.com), en €/t, indexées par date.

    CSV téléchargé à la main (redistribution interdite) : data/raw/eua_investing*.csv. Plusieurs
    fichiers qui se chevauchent sont acceptés et fusionnés.
    """
    files = [Path(path)] if path else sorted(RAW.glob("eua_investing*.csv"))
    if not files:
        raise SystemExit("CSV EUA absent : le placer dans data/raw/eua_investing.csv (voir README).")
    df = pd.concat(pd.read_csv(f, encoding="utf-8-sig") for f in files)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",", ""), errors="coerce")
    eua = df.dropna(subset=["Price"]).drop_duplicates("Date").set_index("Date")["Price"].sort_index()
    print(f"  ok   EUA Investing     {eua.index.min().date()} -> {eua.index.max().date()}"
          f"  ({len(eua)} clôtures)")
    return eua.rename("eua")


def close_for_delivery(close: pd.Series, delivery_dates: pd.Series) -> pd.Series:
    """Pour chaque date de livraison J, dernière clôture (TTF, EUA) datée au plus tard J-2.

    Forward-fill sur le calendrier : week-ends et jours fériés reprennent la dernière cotation.
    """
    days = pd.date_range(close.index.min(), pd.Timestamp(max(delivery_dates)), freq="D")
    known = close.reindex(days).ffill()          # valeur au jour d = dernière clôture <= d
    at_delivery = known.shift(2, freq="D")       # pour livrer J : connu en J-2
    return pd.to_datetime(delivery_dates).map(at_delivery).astype(float)


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


def build_dataset(raw: dict, ttf: pd.Series, eua: pd.Series) -> pd.DataFrame:
    price = to_hourly(raw["price_da"])
    out = pd.DataFrame(index=price.index)
    out["price_da"] = price.iloc[:, 0]
    de_price = to_hourly(raw["price_da_DE_LU"])
    out["de_price_da"] = de_price.iloc[:, 0].reindex(out.index) if not de_price.empty else float("nan")

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
    out["ttf_d2"] = close_for_delivery(ttf, out["date_local"]).to_numpy()
    out["eua_d2"] = close_for_delivery(eua, out["date_local"]).to_numpy()
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
    ap.add_argument("--start", default=DEFAULT_START)
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
    raw = {"price_da": fetch(client, "price_da", FR, start, end),
           "price_da_DE_LU": fetch(client, "price_da", DE_LU, start, end)}
    for zone in (FR, DE):
        raw[f"load_fc_{zone}"] = fetch(client, "load_fc", zone, start, end)
        raw[f"wind_solar_fc_{zone}"] = fetch(client, "wind_solar_fc", zone, start, end)
    raw["capacity_FR"] = fetch(client, "capacity", FR, start, end)
    ttf = fetch_ttf(start, end)
    eua = read_eua()

    df = build_dataset(raw, ttf, eua)
    df.to_parquet(DATASET_PATH)
    report(df)
    print(f"\nÉcrit : {OUT / 'dataset_hourly.parquet'}")


if __name__ == "__main__":
    main()
