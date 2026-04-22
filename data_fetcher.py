"""Bybit data fetcher — USDT perpetual futures only, read-only."""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv
from pybit.unified_trading import HTTP

log = logging.getLogger(__name__)

CATEGORY = "linear"
QUOTE = "USDT"
TIMEFRAMES = {"1H": "60", "4H": "240", "1D": "D"}
CANDLES_PER_TF = 500
TOP_N = 50


def _client() -> HTTP:
    load_dotenv()
    api_key = os.getenv("BYBIT_API_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("BYBIT_API_KEY / BYBIT_API_SECRET missing in .env")
    return HTTP(api_key=api_key, api_secret=api_secret, testnet=False)


def _retry(fn, *args, tries: int = 4, backoff: float = 1.5, **kwargs):
    last_exc = None
    delay = 1.0
    for i in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            log.warning("API call failed (%s/%s): %s", i + 1, tries, e)
            time.sleep(delay)
            delay *= backoff
    raise last_exc


def top_usdt_perpetuals(client: HTTP, n: int = TOP_N) -> List[str]:
    """Return top-n USDT perpetual symbols by 24h turnover (proxy for volume in USDT)."""
    resp = _retry(client.get_tickers, category=CATEGORY)
    rows = resp.get("result", {}).get("list", [])
    usdt = [r for r in rows if r.get("symbol", "").endswith(QUOTE)]
    usdt.sort(key=lambda r: float(r.get("turnover24h", 0) or 0), reverse=True)
    symbols = [r["symbol"] for r in usdt[:n]]
    log.info("Selected top %d USDT perpetuals by 24h turnover", len(symbols))
    return symbols


def fetch_ohlcv(client: HTTP, symbol: str, interval: str, limit: int = CANDLES_PER_TF) -> pd.DataFrame:
    """Fetch up to `limit` candles (may require pagination; Bybit caps at 1000/req)."""
    resp = _retry(client.get_kline, category=CATEGORY, symbol=symbol, interval=interval, limit=min(limit, 1000))
    rows = resp.get("result", {}).get("list", [])
    if not rows:
        return pd.DataFrame()
    # Bybit returns newest-first: [start, open, high, low, close, volume, turnover]
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
    df = df.astype({"ts": "int64", "open": "float64", "high": "float64", "low": "float64",
                    "close": "float64", "volume": "float64", "turnover": "float64"})
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_all(symbols: List[str], timeframes: Dict[str, str] | None = None) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Return {tf_label: {symbol: df}} for all symbols and timeframes."""
    timeframes = timeframes or TIMEFRAMES
    client = _client()
    out: Dict[str, Dict[str, pd.DataFrame]] = {tf: {} for tf in timeframes}
    for tf_label, tf_code in timeframes.items():
        for sym in symbols:
            try:
                df = fetch_ohlcv(client, sym, tf_code, CANDLES_PER_TF)
                if len(df) < 100:
                    log.warning("Skipping %s %s — only %d candles", sym, tf_label, len(df))
                    continue
                out[tf_label][sym] = df
            except Exception as e:
                log.error("Failed to fetch %s %s: %s", sym, tf_label, e)
        log.info("Fetched %s: %d symbols with enough data", tf_label, len(out[tf_label]))
    return out


def fetch_funding_rate(client: HTTP, symbol: str) -> float:
    """Latest funding rate as a float (e.g. 0.0001 = 0.01%/8h). Returns 0.0 on failure."""
    try:
        resp = _retry(client.get_tickers, category=CATEGORY, symbol=symbol)
        rows = resp.get("result", {}).get("list", [])
        if not rows:
            return 0.0
        return float(rows[0].get("fundingRate", 0) or 0)
    except Exception as e:
        log.warning("funding rate fetch failed for %s: %s", symbol, e)
        return 0.0


def fetch_instrument_info(client: HTTP, symbol: str) -> dict:
    """Return min order qty, qty step, last price for sizing."""
    try:
        info = _retry(client.get_instruments_info, category=CATEGORY, symbol=symbol)
        lst = info.get("result", {}).get("list", [])
        if not lst:
            return {}
        it = lst[0]
        lot = it.get("lotSizeFilter", {})
        price = _retry(client.get_tickers, category=CATEGORY, symbol=symbol)
        last = float(price.get("result", {}).get("list", [{}])[0].get("lastPrice", 0) or 0)
        return {
            "min_qty": float(lot.get("minOrderQty", 0) or 0),
            "qty_step": float(lot.get("qtyStep", 0) or 0),
            "last_price": last,
        }
    except Exception as e:
        log.warning("instrument info fetch failed for %s: %s", symbol, e)
        return {}
