from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .binance import BinanceFuturesClient


@dataclass
class DerivativesSnapshot:
    funding_now: float
    funding_median: float
    oi_now: float
    oi_change_1h: float
    funding_flags: List[str]
    oi_flags: List[str]
    # Normalized 0–100 (50 = neutral). Higher = better alignment for the proposed side.
    score: float
    # Raw score in [-10, +10] before normalization (useful for debugging/tuning).
    raw_score: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return {
            "funding_now": self.funding_now,
            "funding_median": self.funding_median,
            "oi_now": self.oi_now,
            "oi_change_1h": self.oi_change_1h,
            "funding_flags": list(self.funding_flags),
            "oi_flags": list(self.oi_flags),
            "score": self.score,
            "raw_score": self.raw_score,
        }


def fetch_derivatives_snapshot(
    client: BinanceFuturesClient,
    symbol: str,
    side: str,
    funding_history_limit: int = 24,
    thresholds: Optional[Dict[str, float]] = None,
) -> DerivativesSnapshot:
    """Funding + OI overlay.

    Implementation note:
      - We compute a small raw score in [-10, +10] (for compact logic).
      - We expose `score` normalized to 0–100 (50 = neutral), because the
        rest of the pipeline expects component scores in 0–100.
    """
    funding_flags: List[str] = []
    oi_flags: List[str] = []

    prem = client.premium_index(symbol)
    funding_now = float(prem.get("lastFundingRate", 0.0))

    hist = client.funding_rate(symbol, limit=funding_history_limit)
    hist_vals = [float(x.get("fundingRate", 0.0)) for x in hist if x.get("fundingRate") is not None]
    funding_median = float(np.median(hist_vals)) if hist_vals else funding_now

    oi_now = float(client.open_interest(symbol).get("openInterest", 0.0))
    oi_hist = client.open_interest_hist(symbol, period="1h", limit=2)
    if len(oi_hist) >= 2:
        oi_prev = float(oi_hist[-2].get("sumOpenInterest", oi_hist[-2].get("openInterest", 0.0)))
    else:
        oi_prev = 0.0
    oi_change_1h = ((oi_now - oi_prev) / oi_prev) if oi_prev > 0 else 0.0

    # Funding scoring (crowding)
    score = 0.0
    thr = thresholds or {}
    funding_high = float(thr.get("funding_high_abs", 0.0005))
    funding_extreme = float(thr.get("funding_extreme_abs", 0.001))
    crowded_abs = float(thr.get("crowded_abs", 0.0005))
    oi_spike = float(thr.get("oi_spike_pct", 0.05))
    oi_flush = float(thr.get("oi_flush_pct", -0.05))

    abs_f = abs(funding_now)
    if abs_f >= funding_extreme:
        funding_flags.append("FUNDING_EXTREME")
        score -= 6.0
    elif abs_f >= funding_high:
        funding_flags.append("FUNDING_HIGH")
        score -= 3.0
    else:
        funding_flags.append("FUNDING_NEUTRAL")
        score += 1.0

    if side == "LONG" and funding_now > crowded_abs:
        funding_flags.append("CROWDED_LONG")
        score -= 2.0
    if side == "SHORT" and funding_now < -crowded_abs:
        funding_flags.append("CROWDED_SHORT")
        score -= 2.0

    # OI scoring
    if oi_change_1h >= oi_spike:
        oi_flags.append("OI_SPIKE")
        score -= 3.0
    elif oi_change_1h <= oi_flush:
        oi_flags.append("OI_FLUSH")
        score += 1.0
    else:
        oi_flags.append("OI_STABLE")
        score += 1.0

    raw = float(max(-10.0, min(10.0, score)))
    # Map [-10, +10] -> [0, 100]
    score_norm = float(max(0.0, min(100.0, 50.0 + raw * 5.0)))

    return DerivativesSnapshot(
        funding_now=funding_now,
        funding_median=funding_median,
        oi_now=oi_now,
        oi_change_1h=oi_change_1h,
        funding_flags=funding_flags,
        oi_flags=oi_flags,
        score=score_norm,
        raw_score=raw,
    )
