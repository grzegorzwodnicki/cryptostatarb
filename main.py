"""Entry point: disclaimer, fetch → coint → signals → display → (optional) watch loop."""
from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

from cointegration import find_pairs
from data_fetcher import TIMEFRAMES, _client, fetch_all, top_usdt_perpetuals
from display import render, write_spread_csvs
from signals import Signal, build_signals, size_pair

DISCLAIMER = (
    "============================================================\n"
    "  To narzędzie analityczne, nie doradztwo inwestycyjne.\n"
    "  Kointegracja na crypto jest niestabilna — wyniki są ważne\n"
    "  tylko na moment testu. Funding rate może zjeść edge.\n"
    "============================================================"
)


def setup_logging() -> None:
    logging.basicConfig(
        filename="app.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Also echo warnings to stderr so the user sees problems without opening the log.
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(console)


def ask_capital() -> tuple[float, float]:
    try:
        cap = float(input("Całkowity kapitał (USDT) [10000]: ") or "10000")
    except ValueError:
        cap = 10000.0
    try:
        pct_in = input("Max % na parę [10]: ") or "10"
        pct = float(pct_in) / 100.0
    except ValueError:
        pct = 0.10
    return cap, pct


def telegram_alert(msg: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat, "text": msg},
            timeout=10,
        )
    except Exception as e:
        logging.getLogger(__name__).warning("telegram send failed: %s", e)


def run_once(capital: float, max_pct: float) -> Dict[str, List[Signal]]:
    log = logging.getLogger(__name__)
    client = _client()
    symbols = top_usdt_perpetuals(client, n=50)
    print(f"Pobrano top {len(symbols)} par USDT perpetual. Ściągam OHLCV dla 1H / 4H / 1D...")

    frames = fetch_all(symbols, TIMEFRAMES)

    signals_by_tf: Dict[str, List[Signal]] = {}
    for tf, fmap in frames.items():
        print(f"[{tf}] Analiza kointegracji na {len(fmap)} symbolach...")
        pairs = find_pairs(fmap)
        sigs = build_signals(pairs)
        signals_by_tf[tf] = sigs

    print(DISCLAIMER)
    print(f"\nData testu: {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    render(signals_by_tf)

    paths = write_spread_csvs(signals_by_tf, frames)
    if paths:
        print(f"\nZapisano {len(paths)} plików CSV.")

    # Sizing + instructions for active entry signals.
    any_active = False
    for tf, sigs in signals_by_tf.items():
        for s in sigs:
            if s.direction not in ("LONG_SPREAD", "SHORT_SPREAD"):
                continue
            any_active = True
            sizing = size_pair(s, capital, max_pct, client)
            print(f"\n[{tf}] {s.pair.a}/{s.pair.b}  Z={s.zscore:+.2f}  {s.action}")
            if sizing is None:
                print("  (pominięto — brak danych instrumentu lub poniżej min order size)")
                continue
            print(f"  {sizing.instructions}")
            print(f"  Notional: A={sizing.notional_a:.2f} USDT  B={sizing.notional_b:.2f} USDT")
            print(f"  Funding: {sizing.symbol_a}={sizing.funding_a*100:.4f}%/8h  "
                  f"{sizing.symbol_b}={sizing.funding_b*100:.4f}%/8h")
            if sizing.warning:
                print(f"  {sizing.warning}")
    if not any_active:
        print("\nBrak sygnałów wejścia w tym przebiegu.")

    return signals_by_tf


def _signal_key(tf: str, s: Signal) -> str:
    return f"{tf}|{s.pair.a}/{s.pair.b}|{s.direction}"


def watch_loop(capital: float, max_pct: float, interval_min: int) -> None:
    log = logging.getLogger(__name__)
    prev_state: Dict[str, str] = {}
    print(f"\nTryb watch: odświeżanie co {interval_min} min. Ctrl+C aby zakończyć.")
    while True:
        try:
            signals_by_tf = run_once(capital, max_pct)
        except Exception as e:
            log.exception("run_once failed: %s", e)
            print(f"[ERROR] {e}")
            time.sleep(interval_min * 60)
            continue

        new_state: Dict[str, str] = {}
        for tf, sigs in signals_by_tf.items():
            for s in sigs:
                if s.direction == "NONE":
                    continue
                key = f"{tf}|{s.pair.a}/{s.pair.b}"
                new_state[key] = s.direction
                prev = prev_state.get(key)
                if prev != s.direction:
                    msg = (f"[{tf}] {s.pair.a}/{s.pair.b}: {prev or 'NONE'} → {s.direction} "
                           f"(Z={s.zscore:+.2f})")
                    print(f"ALERT: {msg}")
                    telegram_alert(msg)
        prev_state = new_state
        time.sleep(interval_min * 60)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Bybit statistical arbitrage scanner")
    parser.add_argument("--watch", type=int, default=0, metavar="MIN",
                        help="Refresh loop every MIN minutes (0 = single run).")
    parser.add_argument("--capital", type=float, default=None, help="Total capital in USDT.")
    parser.add_argument("--max-pct", type=float, default=None, help="Max %% per pair (e.g. 10).")
    args = parser.parse_args(argv)

    setup_logging()
    load_dotenv()
    print(DISCLAIMER)

    if args.capital is not None and args.max_pct is not None:
        capital, max_pct = args.capital, args.max_pct / 100.0
    else:
        capital, max_pct = ask_capital()

    if args.watch > 0:
        watch_loop(capital, max_pct, args.watch)
    else:
        run_once(capital, max_pct)


if __name__ == "__main__":
    main()
