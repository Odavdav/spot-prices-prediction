"""
Usage:
    cp .env.example .env  then set ENTSOE_API_KEY in it (read automatically)
    python -m src.get_data                   # 2018-10-01 -> today
    python -m src.get_data --start 2019-01-01 --end 2026-01-01

Outputs:
    data/raw/<dataset>_<zone>_<year>.parquet raw ENTSO-E cache (one file per yearly query)
    data/raw/ttf_yahoo.parquet               daily TTF front-month closes (Yahoo TTF=F)
    data/processed/dataset_hourly.parquet    merged hourly dataset, UTC index

Information set: prices (FR, DE) are stored raw, at their delivery hour; lags are built in
features.py. TTF and EUA are stored already aligned (`ttf_d2` = last close dated D-2 or earlier for
delivery day D), because the D-1 close is not known at SDAC gate closure (12:00 on D-1).
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
# 'DE' = Germany (national control area), for load and renewables forecasts.
DE = "DE"
# Germany-Luxembourg bidding zone, created on 2018-10-01 (split of DE-AT-LU): start of the dataset.
DE_LU = "DE_LU"
DEFAULT_START = "2018-10-01"
TTF_TICKER = "TTF=F"


# ---------------------------------------------------------------- download
def yearly_chunks(start: pd.Timestamp, end: pd.Timestamp):
    """Split [start, end) into yearly chunks (one query and one cache file per chunk)."""
    cur = start
    while cur < end:
        nxt = min(pd.Timestamp(year=cur.year + 1, month=1, day=1, tz=TZ), end)
        yield cur, nxt
        cur = nxt


def fetch(client, name, zone, start, end, retries=3):
    """Download a dataset in yearly chunks, with a local cache and retries on network errors."""
    parts = []
    for s, e in yearly_chunks(start, end):
        cache = RAW / f"{name}_{zone}_{s.year}.parquet"
        is_current_year = e >= pd.Timestamp.now(tz=TZ).normalize()
        if cache.exists() and not is_current_year:  # the current year is always downloaded again
            parts.append(pd.read_parquet(cache))
            continue
        for attempt in range(1, retries + 1):
            try:
                df = QUERIES[name](client, zone, s, e)
                if isinstance(df, pd.Series):
                    df = df.to_frame(name)
                df.to_parquet(cache)
                parts.append(df)
                print(f"  ok    {name:<16} {zone:<3} {s.year}  ({len(df)} rows)")
                break
            except NoMatchingDataError:
                print(f"  empty {name:<16} {zone:<3} {s.year}")
                break
            except Exception as exc:  # HTTP errors / timeouts: retry
                print(f"  err   {name:<16} {zone:<3} {s.year}  attempt {attempt}/{retries}: {exc}")
                time.sleep(5 * attempt)
        time.sleep(0.5)  # stays far below the limit of 400 requests/min
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if name == "price_da":
        df = patch_missing_days(client, zone, df)
    return df


def patch_missing_days(client, zone, df: pd.DataFrame) -> pd.DataFrame:
    """Download again, one day at a time, the days with missing price hours.

    entsoe-py yearly queries lose the 00:00 hour of 31 December (checked against short queries:
    the rest of the year is identical). A short query returns it.
    """
    s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
    hourly = s.tz_convert("UTC").resample("1h").mean()
    missing = hourly.index[hourly.isna()]
    days = sorted({t.tz_convert(TZ).normalize() for t in missing})
    for d in days:
        try:
            fix = QUERIES["price_da"](client, zone, d, d + pd.Timedelta(days=1))
        except NoMatchingDataError:
            print(f"  gap   price_da {zone} {d.date()}: no ENTSO-E data")
            continue
        fix = fix.to_frame(df.columns[0]) if isinstance(fix, pd.Series) else fix
        df = df.combine_first(fix.tz_convert(df.index.tz))
        print(f"  fix   price_da {zone} {d.date()}")
        time.sleep(0.5)
    return df


QUERIES = {
    "price_da": lambda c, z, s, e: c.query_day_ahead_prices(z, start=s, end=e),
    "load_fc": lambda c, z, s, e: c.query_load_forecast(z, start=s, end=e, process_type="A01"),  # A01 = day-ahead
    "wind_solar_fc": lambda c, z, s, e: c.query_wind_and_solar_forecast(z, start=s, end=e, process_type="A01"),
}


def fetch_ttf(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Daily TTF front-month closes (Yahoo), indexed by trading date.

    Goes back 30 days before `start` so that the first days can be aligned (forward-fill).
    """
    cache = RAW / "ttf_yahoo.parquet"
    df = yf.download(TTF_TICKER, start=(start - pd.Timedelta(days=30)).date(), end=end.date(),
                     progress=False, auto_adjust=False)
    if df.empty:
        if cache.exists():  # Yahoo unavailable: fall back on the last download
            print("  err   empty TTF from Yahoo, using the cache")
            return pd.read_parquet(cache)["ttf"]
        raise SystemExit("TTF download failed (Yahoo) and no cache.")
    close = df["Close"].squeeze().rename("ttf").dropna()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close.to_frame().to_parquet(cache)
    print(f"  ok    TTF {TTF_TICKER:<12} {close.index.min().date()} -> {close.index.max().date()}"
          f"  ({len(close)} closes)")
    return close


def read_eua(path=None) -> pd.Series:
    """Daily EUA closes (December contract, Investing.com), in €/t, indexed by date.

    CSV downloaded by hand (redistribution not allowed): data/raw/eua_investing*.csv. Several
    overlapping files are accepted and merged.
    """
    files = [Path(path)] if path else sorted(RAW.glob("eua_investing*.csv"))
    if not files:
        raise SystemExit("EUA CSV missing: put it in data/raw/eua_investing.csv (see README).")
    df = pd.concat(pd.read_csv(f, encoding="utf-8-sig") for f in files)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["Price"] = pd.to_numeric(df["Price"].astype(str).str.replace(",", ""), errors="coerce")
    eua = df.dropna(subset=["Price"]).drop_duplicates("Date").set_index("Date")["Price"].sort_index()
    print(f"  ok    EUA Investing     {eua.index.min().date()} -> {eua.index.max().date()}"
          f"  ({len(eua)} closes)")
    return eua.rename("eua")


def close_for_delivery(close: pd.Series, delivery_dates: pd.Series) -> pd.Series:
    """For each delivery day D, the last close (TTF, EUA) dated D-2 or earlier.

    Calendar forward-fill: weekends and holidays carry the last quote.
    """
    days = pd.date_range(close.index.min(), pd.Timestamp(max(delivery_dates)), freq="D")
    known = close.reindex(days).ffill()          # value on day d = last close <= d
    at_delivery = known.shift(2, freq="D")       # for delivery on D: known on D-2
    return pd.to_datetime(delivery_dates).map(at_delivery).astype(float)


# ---------------------------------------------------------------- formatting
def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to UTC, then hourly mean (handles 15-min steps, e.g. prices since 2025-10-01)."""
    if df.empty:
        return df
    df = df.copy()
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC")
    return df.resample("1h").mean()


def pick(df: pd.DataFrame, *keywords) -> pd.Series:
    """Sum of the columns whose name contains one of the keywords (names vary by zone)."""
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

    out["time_local"] = out.index.tz_convert(TZ)  # for calendar features and the "day" grouping
    out["date_local"] = out["time_local"].dt.date
    out["ttf_d2"] = close_for_delivery(ttf, out["date_local"]).to_numpy()
    out["eua_d2"] = close_for_delivery(eua, out["date_local"]).to_numpy()
    return out


def report(df: pd.DataFrame):
    negative = (df["price_da"] < 0).where(df["price_da"].notna())
    print("\n=== Summary ===")
    print(f"Period: {df.index.min()} -> {df.index.max()}  ({len(df)} hours)")
    print(f"Negative-price hours: {int(negative.sum())} ({negative.mean():.2%})")
    print("\nShare of missing values by column:")
    print(df.drop(columns=["time_local", "date_local"]).isna().mean().map("{:.1%}".format).to_string())
    print("\nNegative-price hours by year:")
    print(negative.groupby(df["time_local"].dt.year).sum().astype(int).to_string())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=None, help="excluded; defaults to tomorrow 00:00")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise SystemExit("ENTSOE_API_KEY missing: set it in .env (see .env.example).")

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
    ttf = fetch_ttf(start, end)
    eua = read_eua()

    df = build_dataset(raw, ttf, eua)
    df.to_parquet(DATASET_PATH)
    report(df)
    print(f"\nWritten: {OUT / 'dataset_hourly.parquet'}")


if __name__ == "__main__":
    main()
