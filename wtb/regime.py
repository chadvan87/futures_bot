from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .indicators import adx, atr, ema


@dataclass
class BtcRegime:
    btc_trend: str
    close: float
    ema50: float
    ema200: float
    ema50_slope: float
    adx14: float
    atrp14: float
    volatility_state: str
    notes: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "btc_trend": self.btc_trend,
            "close": self.close,
            "ema50": self.ema50,
            "ema200": self.ema200,
            "ema50_slope": self.ema50_slope,
            "adx14": self.adx14,
            "atrp14": self.atrp14,
            "volatility_state": self.volatility_state,
            "notes": list(self.notes),
        }


def detect_btc_regime(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, ema_fast: int = 50, ema_slow: int = 200) -> BtcRegime:
    close = close.astype(float)
    ema50_series = ema(close, ema_fast)
    ema200_series = ema(close, ema_slow)
    ema50 = float(ema50_series[-1])
    ema200 = float(ema200_series[-1])
    ema50_slope = float(ema50_series[-1] - ema50_series[-5]) if len(ema50_series) >= 5 else 0.0

    adx14_series = adx(high.astype(float), low.astype(float), close, period=14)
    adx14 = float(adx14_series[-1]) if len(adx14_series) else 0.0

    atr14 = atr(high.astype(float), low.astype(float), close, period=14)
    atrp14 = float(atr14[-1] / close[-1] * 100.0) if len(atr14) else 0.0
    vol_state = "HIGH" if atrp14 >= 4.0 else "NORMAL"

    trend = "RANGE"
    if close[-1] > ema50 > ema200 and ema50_slope > 0:
        trend = "BULLISH"
    elif close[-1] < ema50 < ema200 and ema50_slope < 0:
        trend = "BEARISH"

    notes = [
        f"BTC close={float(close[-1]):.2f}",
        f"EMA50={ema50:.2f}, EMA200={ema200:.2f}, EMA50_slope={ema50_slope:.4f}",
        f"ADX14={adx14:.2f}",
        f"ATR%14={atrp14:.2f} ({vol_state})",
    ]

    return BtcRegime(
        btc_trend=trend,
        close=float(close[-1]),
        ema50=ema50,
        ema200=ema200,
        ema50_slope=ema50_slope,
        adx14=adx14,
        atrp14=atrp14,
        volatility_state=vol_state,
        notes=notes,
    )
