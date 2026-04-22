"""Table rendering, confluence detection, CSV export."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd
from tabulate import tabulate

from cointegration import Pair
from signals import Signal

log = logging.getLogger(__name__)

CSV_DIR = "."


def _signal_tag(direction: str) -> str:
    return {
        "LONG_SPREAD": "LONG spread",
        "SHORT_SPREAD": "SHORT spread",
        "EXIT": "EXIT",
        "STOP": "STOP",
        "NONE": "-",
    }.get(direction, direction)


def _confluence_map(signals_by_tf: Dict[str, List[Signal]]) -> Dict[str, Dict[str, list]]:
    """For each pair key 'A/B', track in which timeframes a directional signal appears.

    Returns {"A/B": {"LONG_SPREAD": [tf,...], "SHORT_SPREAD": [tf,...]}}
    """
    m: Dict[str, Dict[str, list]] = {}
    for tf, sigs in signals_by_tf.items():
        for s in sigs:
            if s.direction not in ("LONG_SPREAD", "SHORT_SPREAD"):
                continue
            key = f"{s.pair.a}/{s.pair.b}"
            m.setdefault(key, {"LONG_SPREAD": [], "SHORT_SPREAD": []})
            m[key][s.direction].append(tf)
    return m


def _confluence_badge(pair_key: str, direction: str, cmap: Dict[str, Dict[str, list]]) -> str:
    if direction not in ("LONG_SPREAD", "SHORT_SPREAD"):
        return "-"
    tfs = cmap.get(pair_key, {}).get(direction, [])
    n = len(tfs)
    if n >= 3:
        return "3TF 🔥"
    if n == 2:
        return "2TF ⚡"
    return "-"


def _row_for(signal: Signal, tf: str, cmap) -> dict:
    p = signal.pair
    key = f"{p.a}/{p.b}"
    return {
        "para": key,
        "korelacja": round(p.correlation, 3),
        "p-value": round(p.coint_pvalue, 4),
        "half-life": round(p.half_life, 1),
        "Z-score": round(signal.zscore, 2),
        "sygnał": _signal_tag(signal.direction),
        "akcja": signal.action,
        "hedge_ratio": round(p.hedge_ratio, 4),
        "SL (Z)": f"±{abs(signal.sl_level):.1f}",
        "CONFLUENCE": _confluence_badge(key, signal.direction, cmap),
    }


def render(signals_by_tf: Dict[str, List[Signal]]) -> None:
    cmap = _confluence_map(signals_by_tf)

    # Collect high-confluence rows (same direction on 2+ TFs); dedupe by pair/direction.
    high = []
    seen = set()
    for tf, sigs in signals_by_tf.items():
        for s in sigs:
            if s.direction not in ("LONG_SPREAD", "SHORT_SPREAD"):
                continue
            key = f"{s.pair.a}/{s.pair.b}"
            tfs = cmap.get(key, {}).get(s.direction, [])
            if len(tfs) >= 2 and (key, s.direction) not in seen:
                seen.add((key, s.direction))
                row = _row_for(s, tf, cmap)
                row["timeframes"] = ",".join(tfs)
                high.append(row)

    if high:
        print("\n=== WYSOKIE PRAWDOPODOBIEŃSTWO (CONFLUENCE) ===")
        print(tabulate(high, headers="keys", tablefmt="github"))
    else:
        print("\n=== WYSOKIE PRAWDOPODOBIEŃSTWO (CONFLUENCE) ===")
        print("Brak sygnałów z confluence na 2+ timeframe.")

    for tf in ("1H", "4H", "1D"):
        sigs = signals_by_tf.get(tf, [])
        # Show only pairs with an active signal (entry/exit/stop) to keep tables short.
        active = [s for s in sigs if s.direction != "NONE"]
        print(f"\n=== SYGNAŁY {tf} ===")
        if not active:
            print("Brak aktywnych sygnałów.")
            continue
        rows = [_row_for(s, tf, cmap) for s in active]
        print(tabulate(rows, headers="keys", tablefmt="github"))


def _ts_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_pair(pair_key: str) -> str:
    return pair_key.replace("/", "_")


def write_spread_csvs(signals_by_tf: Dict[str, List[Signal]], frames_by_tf: Dict[str, Dict[str, pd.DataFrame]]) -> List[str]:
    """Write spread_data_{pair}_{TF}_{ts}.csv with columns timestamp, price_A, price_B, spread, zscore."""
    ts = _ts_tag()
    paths: List[str] = []
    for tf, sigs in signals_by_tf.items():
        frames = frames_by_tf.get(tf, {})
        for s in sigs:
            p = s.pair
            if p.a not in frames or p.b not in frames:
                continue
            df_a = frames[p.a].set_index("timestamp")["close"].rename("price_A")
            df_b = frames[p.b].set_index("timestamp")["close"].rename("price_B")
            out = pd.concat([df_a, df_b], axis=1).dropna()
            out["spread"] = p.spread.reindex(out.index)
            out["zscore"] = s.zscore_series.reindex(out.index)
            out = out.dropna()
            fname = f"spread_data_{_safe_pair(f'{p.a}/{p.b}')}_{tf}_{ts}.csv"
            path = os.path.join(CSV_DIR, fname)
            out.to_csv(path, index=True, index_label="timestamp")
            paths.append(path)
    log.info("Wrote %d spread CSV files", len(paths))
    return paths
