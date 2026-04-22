"""Z-score signals + position sizing for qualified pairs."""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from cointegration import Pair
from data_fetcher import fetch_funding_rate, fetch_instrument_info

log = logging.getLogger(__name__)

ZSCORE_WINDOW = 30
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 3.5
FUNDING_WARN = 0.001  # 0.1% per 8h


@dataclass
class Signal:
    pair: Pair
    zscore: float
    direction: str  # "LONG_SPREAD", "SHORT_SPREAD", "EXIT", "NONE", "STOP"
    action: str     # human-readable, e.g. "KUP A, SPRZEDAJ B"
    sl_level: float  # z-score at which SL triggers (±3.5 -> absolute spread value)
    zscore_series: pd.Series


def _compute_zscore(spread: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    mu = spread.rolling(window).mean()
    sd = spread.rolling(window).std(ddof=0)
    return (spread - mu) / sd


def _classify(z: float) -> tuple[str, str]:
    if not np.isfinite(z):
        return "NONE", "-"
    if z >= STOP_Z:
        return "STOP", "STOP LOSS (short spread)"
    if z <= -STOP_Z:
        return "STOP", "STOP LOSS (long spread)"
    if z >= ENTRY_Z:
        return "SHORT_SPREAD", "SPRZEDAJ A, KUP B"
    if z <= -ENTRY_Z:
        return "LONG_SPREAD", "KUP A, SPRZEDAJ B"
    if -EXIT_Z <= z <= EXIT_Z:
        return "EXIT", "WYJDŹ Z POZYCJI"
    return "NONE", "-"


def build_signals(pairs: List[Pair]) -> List[Signal]:
    out: List[Signal] = []
    for p in pairs:
        z = _compute_zscore(p.spread)
        if z.dropna().empty:
            continue
        last_z = float(z.dropna().iloc[-1])
        direction, action = _classify(last_z)
        # SL level expressed as signed z-score threshold ±3.5
        sl = math.copysign(STOP_Z, last_z) if direction in ("LONG_SPREAD", "SHORT_SPREAD") else STOP_Z
        out.append(Signal(pair=p, zscore=last_z, direction=direction, action=action,
                          sl_level=sl, zscore_series=z))
    return out


@dataclass
class Sizing:
    symbol_a: str
    symbol_b: str
    qty_a: float
    qty_b: float
    notional_a: float
    notional_b: float
    price_a: float
    price_b: float
    funding_a: float
    funding_b: float
    warning: Optional[str]
    instructions: str


def _round_step(qty: float, step: float, min_qty: float) -> float:
    if step <= 0:
        return max(qty, min_qty)
    rounded = math.floor(qty / step) * step
    if rounded < min_qty:
        return 0.0
    # clean floating-point tail
    return float(f"{rounded:.10g}")


def size_pair(signal: Signal, capital_usdt: float, max_pct: float, client) -> Optional[Sizing]:
    """Compute leg quantities given capital * max_pct for the pair.

    Rule: allocate capital_per_pair/2 in notional to leg A; leg B notional = hedge_ratio * leg_A_notional.
    """
    if signal.direction not in ("LONG_SPREAD", "SHORT_SPREAD"):
        return None
    a = signal.pair.a
    b = signal.pair.b
    hr = signal.pair.hedge_ratio
    cap_pair = capital_usdt * max_pct

    info_a = fetch_instrument_info(client, a)
    info_b = fetch_instrument_info(client, b)
    if not info_a or not info_b:
        log.warning("Missing instrument info for %s or %s", a, b)
        return None

    pa = info_a["last_price"]
    pb = info_b["last_price"]
    if pa <= 0 or pb <= 0:
        return None

    notional_a = cap_pair / 2.0
    notional_b = abs(hr) * notional_a  # hedge ratio on log prices ≈ $ ratio when used on prices in log space

    qty_a_raw = notional_a / pa
    qty_b_raw = notional_b / pb

    qty_a = _round_step(qty_a_raw, info_a.get("qty_step", 0), info_a.get("min_qty", 0))
    qty_b = _round_step(qty_b_raw, info_b.get("qty_step", 0), info_b.get("min_qty", 0))
    if qty_a == 0 or qty_b == 0:
        log.warning("Below min order size: %s=%s %s=%s", a, qty_a, b, qty_b)
        return None

    funding_a = fetch_funding_rate(client, a)
    funding_b = fetch_funding_rate(client, b)
    warn = None
    if max(abs(funding_a), abs(funding_b)) > FUNDING_WARN:
        warn = (f"⚠️  High funding: {a}={funding_a*100:.3f}%/8h, "
                f"{b}={funding_b*100:.3f}%/8h — może zjeść edge")

    if signal.direction == "LONG_SPREAD":
        instr = f"KUP {qty_a} {a} @ {pa:.6g}  |  SPRZEDAJ {qty_b} {b} @ {pb:.6g}"
    else:
        instr = f"SPRZEDAJ {qty_a} {a} @ {pa:.6g}  |  KUP {qty_b} {b} @ {pb:.6g}"

    return Sizing(
        symbol_a=a, symbol_b=b,
        qty_a=qty_a, qty_b=qty_b,
        notional_a=qty_a * pa, notional_b=qty_b * pb,
        price_a=pa, price_b=pb,
        funding_a=funding_a, funding_b=funding_b,
        warning=warn,
        instructions=instr,
    )
