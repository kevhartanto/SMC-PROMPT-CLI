"""Public OHLCV sources for historical research and fresh signal scanning.

Research: Binance USD-M Futures archival 1h candles (data.binance.vision).
Live: Bitunix USD-M Futures 1h candles. Never silently splice two exchanges.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile
import json
import logging
import re
import time

import pandas as pd

LOG = logging.getLogger(__name__)
FUTURES_ARCHIVE = "https://data.binance.vision/data/futures/um"
BITUNIX_API = "https://fapi.bitunix.com/api/v1/futures/market/kline"
SYMBOL_RE = re.compile(r"^[A-Z0-9]{3,25}$")
FIELDS = ["open", "high", "low", "close", "volume"]


def _request(url: str, timeout: float = 18.0) -> bytes:
    last = None
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "SMC-MTF-research/0.1"})
            with urlopen(req, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code == 404:
                raise
            last = exc
        except (URLError, TimeoutError, ConnectionError, OSError) as exc:
            last = exc
        time.sleep((attempt + 1) * 0.8)
    raise RuntimeError(f"Download failed {url}: {last}")


def _dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=FIELDS, index=pd.DatetimeIndex([], tz="UTC", name="open_time"))
    frame = frame.iloc[:, :6].copy()
    frame.columns = ["open_time", *FIELDS]
    stamps = pd.to_numeric(frame["open_time"], errors="coerce")
    # USD-M archive timestamps are milliseconds; tolerate microsecond records.
    stamps = stamps.where(stamps < 1e15, stamps / 1000)
    frame.index = pd.to_datetime(stamps, unit="ms", utc=True, errors="coerce")
    for name in FIELDS:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame[FIELDS]
    frame = frame.loc[frame.index.notna()].dropna()
    frame = frame.loc[(frame["high"] >= frame[["open", "close", "low"]].max(axis=1)) &
                      (frame["low"] <= frame[["open", "close", "high"]].min(axis=1)) &
                      (frame["close"] > 0)]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame.index.name = "open_time"
    return frame


def _archive_csv(blob: bytes) -> pd.DataFrame:
    with ZipFile(BytesIO(blob)) as zip_file:
        csv_names = [x for x in zip_file.namelist() if x.lower().endswith(".csv")]
        if not csv_names:
            raise ValueError("Archive missing CSV")
        data = zip_file.read(csv_names[0])
    frame = pd.read_csv(BytesIO(data), header=None, usecols=range(6), dtype=str, low_memory=False)
    return _dataframe(frame)


def _monthly(start: datetime, end: datetime):
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield year, month
        month += 1
        if month == 13:
            year, month = year + 1, 1


def download_history(symbol: str, *, years: int = 5, cache_dir: str = ".cache/mtf",
                     workers: int = 8, now: datetime | None = None) -> tuple[pd.DataFrame, dict]:
    """Download 1h closed candles. Missing pre-listing archives are not fabricated.

    Cached completed monthly archives remain useful on subsequent runs.
    Daily archives for the current month are refreshed; they publish next day.
    """
    symbol = symbol.upper()
    if not SYMBOL_RE.fullmatch(symbol) or years < 1 or years > 6:
        raise ValueError("Invalid symbol or years")
    now = now or datetime.now(timezone.utc)
    end = now.replace(minute=0, second=0, microsecond=0)
    start = (pd.Timestamp(end) - pd.DateOffset(years=years)).to_pydatetime()
    completed_month_end = (end.replace(day=1) - timedelta(days=1))
    requests = []
    for year, month in _monthly(start, completed_month_end):
        key = f"{symbol}-1h-{year}-{month:02d}.zip"
        url = f"{FUTURES_ARCHIVE}/monthly/klines/{symbol}/1h/{key}"
        requests.append((url, key, True))
    # Fetch only the current month's completed days from Binance's daily archives.
    day = end.replace(day=1)
    while day.date() < end.date():
        key = f"{symbol}-1h-{day:%Y-%m-%d}.zip"
        url = f"{FUTURES_ARCHIVE}/daily/klines/{symbol}/1h/{key}"
        requests.append((url, key, False))
        day += timedelta(days=1)

    cache = Path(cache_dir) / symbol
    cache.mkdir(parents=True, exist_ok=True)

    def load_one(item):
        url, key, immutable = item
        destination = cache / key
        if destination.exists() and destination.stat().st_size > 30:
            blob = destination.read_bytes()
        else:
            try:
                blob = _request(url)
                destination.write_bytes(blob)
            except HTTPError as exc:
                if exc.code == 404:
                    return None, "not_available"
                raise
        try:
            return _archive_csv(blob), "ok"
        except Exception:
            destination.unlink(missing_ok=True)
            raise

    frames = []
    failed = []
    missing = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(load_one, request): request for request in requests}
        for future in as_completed(futures):
            url, key, _ = futures[future]
            try:
                frame, status = future.result()
                if status == "not_available":
                    missing += 1
                elif frame is not None and not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                failed.append({"key": key, "error": str(exc)[:200]})
    if not frames:
        raise RuntimeError(f"No historical archives for {symbol} ({len(failed)} download errors)")
    candles = pd.concat(frames).sort_index()
    candles = candles[~candles.index.duplicated(keep="last")]
    candles = candles.loc[(candles.index >= pd.Timestamp(start)) & (candles.index + pd.Timedelta(hours=1) <= pd.Timestamp(end))]
    if candles.empty:
        raise RuntimeError("No candles remain in requested historical range")
    gaps = candles.index.to_series().diff().dropna()
    large_gaps = int((gaps > pd.Timedelta(hours=3)).sum())
    meta = {
        "symbol": symbol,
        "source": "Binance USD-M Futures historical 1h archives",
        "requested_from": start.isoformat(),
        "requested_to": end.isoformat(),
        "observed_from": candles.index.min().isoformat(),
        "observed_to": candles.index.max().isoformat(),
        "years_available": round((candles.index.max() - candles.index.min()).total_seconds() / (86400 * 365.25), 3),
        "bars": len(candles),
        "missing_archive_files": missing,
        "errors": failed[:15],
        "large_gaps_over_3h": large_gaps
    }
    return candles, meta


def bitunix_latest(symbol: str, *, hours: int = 24 * 100) -> tuple[pd.DataFrame, dict]:
    symbol = symbol.upper()
    if not SYMBOL_RE.fullmatch(symbol):
        raise ValueError("Invalid symbol")
    rows = []
    end_time = None
    while len(rows) < hours:
        params = f"symbol={symbol}&interval=1h&limit={min(hours - len(rows), 200)}"
        if end_time is not None:
            params += f"&endTime={end_time}"
        payload = json.loads(_request(f"{BITUNIX_API}?{params}").decode("utf-8"))
        batch = payload.get("data")
        if not isinstance(batch, list) or not batch:
            break
        rows.extend(batch)
        oldest = min(int(r["time"]) for r in batch)
        end_time = oldest - 1
        if len(batch) < 200:
            break
        time.sleep(0.13)
    if not rows:
        raise RuntimeError(f"Bitunix returned no 1h candles for {symbol}")
    # Endpoint excludes current forming candle. Avoid assuming freshness without checking.
    raw = pd.DataFrame([{
        "open_time": r.get("time"), "open": r.get("open"),
        "high": r.get("high"), "low": r.get("low"),
        "close": r.get("close"), "volume": r.get("baseVol", 0)
    } for r in rows])
    df = _dataframe(raw)
    now = pd.Timestamp.now(tz="UTC")
    df = df.loc[df.index + pd.Timedelta(hours=1) <= now]
    return df, {
        "symbol": symbol, "source": "Bitunix public futures 1h API",
        "bars": len(df), "observed_from": df.index.min().isoformat(),
        "observed_to": df.index.max().isoformat()
    }
