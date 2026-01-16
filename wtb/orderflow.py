from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .binance import BinanceFuturesClient


@dataclass
class OrderflowSnapshot:
    delta_15m: float
    delta_1h: float
    cvd_1h_slope: float
    flags: List[str]
    # Normalized 0–100 (50 = neutral). Higher = more buy-aggressor flow; lower = more sell-aggressor flow.
    score: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "delta_15m": self.delta_15m,
            "delta_1h": self.delta_1h,
            "cvd_1h_slope": self.cvd_1h_slope,
            "flags": list(self.flags),
            "score": self.score,
        }


def compute_orderflow(
    client: BinanceFuturesClient,
    symbol: str,
    now_ms: int,
    insecure_ssl: bool,
    window_15m_min: int = 15,
    window_1h_min: int = 60,
    limit: int = 1000,
) -> OrderflowSnapshot:
    """Orderflow-lite from aggTrades. Uses isBuyerMaker to proxy aggressor side.

    delta = sum(qty*price) signed by aggressor.
    """
    start_15m = now_ms - window_15m_min * 60 * 1000
    start_1h = now_ms - window_1h_min * 60 * 1000

    # Get a chunk of recent trades. Binance returns ascending by time.
    trades = client.agg_trades(symbol, limit=limit, start_time=start_1h, end_time=now_ms)
    if not trades:
        # Neutral if we have no data (avoid killing the pipeline / watchlist).
        return OrderflowSnapshot(0.0, 0.0, 0.0, ["NO_TRADES"], 50.0)

    delta_15m = 0.0
    delta_1h = 0.0
    cvd_points: List[float] = []
    cvd = 0.0

    for t in trades:
        ts = int(t.get("T"))
        price = float(t.get("p"))
        qty = float(t.get("q"))
        notional = price * qty
        # isBuyerMaker: True means buyer is maker => seller aggressor
        is_buyer_maker = bool(t.get("m"))
        signed = -notional if is_buyer_maker else notional

        if ts >= start_1h:
            delta_1h += signed
            cvd += signed
            cvd_points.append(cvd)
        if ts >= start_15m:
            delta_15m += signed

    # Slope proxy: last - first normalized
    if len(cvd_points) >= 2:
        cvd_1h_slope = (cvd_points[-1] - cvd_points[0]) / max(1.0, abs(cvd_points[0]) + 1e-9)
    else:
        cvd_1h_slope = 0.0

    flags: List[str] = []

    # Raw flow score in [-5, +5], then normalize to [0, 100].
    # -5 => strong sell pressure, +5 => strong buy pressure.
    raw = 0.0
    if abs(delta_1h) > 0:
        raw = float(np.clip(delta_15m / (abs(delta_1h) + 1e-9) * 10.0, -5.0, 5.0))

    # Small extra nudge based on CVD slope direction (kept tiny to avoid overfitting).
    if cvd_1h_slope > 0:
        raw += 0.5
    elif cvd_1h_slope < 0:
        raw -= 0.5
    raw = float(np.clip(raw, -5.0, 5.0))
    score = 50.0 + raw * 10.0

    if delta_15m > 0:
        flags.append("DELTA_POS_15M")
    elif delta_15m < 0:
        flags.append("DELTA_NEG_15M")

    if cvd_1h_slope > 0:
        flags.append("CVD_UP")
    elif cvd_1h_slope < 0:
        flags.append("CVD_DOWN")

    return OrderflowSnapshot(delta_15m, delta_1h, cvd_1h_slope, flags, score)
