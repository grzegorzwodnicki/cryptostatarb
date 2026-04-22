"""Pair selection: Pearson corr filter -> Engle-Granger coint -> ADF -> half-life filter."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

log = logging.getLogger(__name__)

CORR_THRESHOLD = 0.85
COINT_PVAL = 0.05
ADF_PVAL = 0.05
HL_MIN = 2
HL_MAX = 100


@dataclass
class Pair:
    a: str
    b: str
    correlation: float
    coint_pvalue: float
    hedge_ratio: float
    adf_pvalue: float
    half_life: float
    spread: pd.Series  # indexed by timestamp

    def to_row(self) -> dict:
        return {
            "pair": f"{self.a}/{self.b}",
            "correlation": round(self.correlation, 4),
            "p_value": round(self.coint_pvalue, 4),
            "adf_p": round(self.adf_pvalue, 4),
            "half_life": round(self.half_life, 2),
            "hedge_ratio": round(self.hedge_ratio, 4),
        }


def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff().dropna()


def _align_close_matrix(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return DataFrame of close prices indexed by timestamp, columns = symbols."""
    s = {sym: df.set_index("timestamp")["close"] for sym, df in frames.items()}
    return pd.DataFrame(s).dropna(how="any")


def _half_life(spread: pd.Series) -> float:
    """AR(1) half-life: HL = -log(2)/log(|beta|) where d(spread) = alpha + beta*spread_lag + eps."""
    s = spread.dropna()
    if len(s) < 10:
        return np.nan
    s_lag = s.shift(1).dropna()
    s_diff = s.diff().dropna()
    x, y = s_lag.align(s_diff, join="inner")
    x_c = sm.add_constant(x)
    try:
        model = sm.OLS(y, x_c).fit()
        beta = model.params.iloc[1]
    except Exception:
        return np.nan
    if beta >= 0 or beta <= -2:
        return np.nan
    # mean-reversion speed: phi = 1 + beta; HL = -log(2)/log(phi)
    phi = 1.0 + beta
    if phi <= 0 or phi >= 1:
        return np.nan
    return float(-np.log(2) / np.log(phi))


def find_pairs(frames: Dict[str, pd.DataFrame]) -> List[Pair]:
    """Run the full Krok 2 pipeline on one timeframe's frames (sym -> ohlcv df)."""
    if len(frames) < 2:
        return []

    closes = _align_close_matrix(frames)
    if len(closes) < 100:
        log.warning("Not enough aligned candles (%d) for pair selection", len(closes))
        return []

    rets = np.log(closes).diff().dropna()
    corr = rets.corr()

    candidates = []
    for a, b in combinations(corr.columns, 2):
        c = corr.loc[a, b]
        if pd.notna(c) and abs(c) > CORR_THRESHOLD:
            candidates.append((a, b, float(c)))
    log.info("Pre-filter: %d pairs with |corr| > %.2f", len(candidates), CORR_THRESHOLD)

    pairs: List[Pair] = []
    for a, b, c in candidates:
        pa = np.log(closes[a])
        pb = np.log(closes[b])
        try:
            _, coint_p, _ = coint(pa, pb)
        except Exception as e:
            log.debug("coint failed %s/%s: %s", a, b, e)
            continue
        if coint_p >= COINT_PVAL:
            continue

        xc = sm.add_constant(pb)
        ols = sm.OLS(pa, xc).fit()
        beta = float(ols.params.iloc[1])
        spread = pa - beta * pb

        try:
            adf_p = float(adfuller(spread.dropna(), autolag="AIC")[1])
        except Exception as e:
            log.debug("adfuller failed %s/%s: %s", a, b, e)
            continue
        if adf_p >= ADF_PVAL:
            continue

        hl = _half_life(spread)
        if not np.isfinite(hl) or hl < HL_MIN or hl > HL_MAX:
            continue

        pairs.append(Pair(
            a=a, b=b,
            correlation=c,
            coint_pvalue=float(coint_p),
            hedge_ratio=beta,
            adf_pvalue=adf_p,
            half_life=hl,
            spread=spread,
        ))

    log.info("Qualified pairs after coint+ADF+half-life: %d", len(pairs))
    return pairs
