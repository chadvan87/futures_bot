from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .indicators import atr


@dataclass
class ManageSuggestion:
    symbol: str
    side: str
    entry: float
    current: float
    stop_loss: float
    tp1: float
    atr4h: float
    r_now: float
    action: str
    new_sl: Optional[float]
    chandelier: Optional[float]
    time_stop_note: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry": self.entry,
            "current": self.current,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "atr4h": self.atr4h,
            "r_now": self.r_now,
            "action": self.action,
            "new_sl": self.new_sl,
            "chandelier": self.chandelier,
            "time_stop_note": self.time_stop_note,
        }


def compute_dynamic_exit(
    symbol: str,
    side: str,
    entry: float,
    stop_loss: float,
    take_profits: List[float],
    klines_4h: List[List[object]],
    max_candles_no_progress: int = 4,
    chandelier_mult: float = 3.0,
    trail_lookback: int = 14,
) -> ManageSuggestion:
    """Dynamic exits: BE + Chandelier + time stop.

    - Move to breakeven at >=1R or once TP1 is tagged.
    - Suggest chandelier trail level.
    - Time stop: if no >=0.5R after K candles, suggest reduce/exit.

    klines_4h: last candles, newest last.
    """
    highs = np.array([float(x[2]) for x in klines_4h], dtype=float)
    lows = np.array([float(x[3]) for x in klines_4h], dtype=float)
    closes = np.array([float(x[4]) for x in klines_4h], dtype=float)
    current = float(closes[-1])

    a = atr(highs, lows, closes, 14)
    atr4h = float(a) if a is not None else float(np.nan)

    tp1 = float(take_profits[0]) if take_profits else float("nan")
    risk = abs(entry - stop_loss)
    r_now = 0.0
    if risk > 0:
        if side.upper() == "LONG":
            r_now = (current - entry) / risk
        else:
            r_now = (entry - current) / risk

    action = "HOLD"
    new_sl = None

    # Breakeven rule
    hit_tp1 = False
    if take_profits:
        if side.upper() == "LONG" and current >= tp1:
            hit_tp1 = True
        if side.upper() == "SHORT" and current <= tp1:
            hit_tp1 = True

    if hit_tp1 or r_now >= 1.0:
        action = "MOVE_SL_TO_BE"
        # small buffer to reduce stop hunts
        buff = 0.1 * atr4h if np.isfinite(atr4h) else 0.0
        if side.upper() == "LONG":
            new_sl = entry - buff
        else:
            new_sl = entry + buff

    # Chandelier
    chandelier = None
    if np.isfinite(atr4h) and len(closes) >= trail_lookback:
        if side.upper() == "LONG":
            hh = float(np.max(highs[-trail_lookback:]))
            chandelier = hh - chandelier_mult * atr4h
        else:
            ll = float(np.min(lows[-trail_lookback:]))
            chandelier = ll + chandelier_mult * atr4h

    # Time stop note
    time_stop_note = ""
    if len(closes) >= max_candles_no_progress + 1 and risk > 0:
        recent = closes[-(max_candles_no_progress + 1):]
        if side.upper() == "LONG":
            best = float(np.max(recent))
            r_best = (best - entry) / risk
        else:
            best = float(np.min(recent))
            r_best = (entry - best) / risk
        if r_best < 0.5:
            time_stop_note = f"No >=0.5R progress over last {max_candles_no_progress} candles → consider exit/reduce."

    return ManageSuggestion(
        symbol=symbol,
        side=side.upper(),
        entry=float(entry),
        current=float(current),
        stop_loss=float(stop_loss),
        tp1=float(tp1) if take_profits else float("nan"),
        atr4h=float(atr4h),
        r_now=float(r_now),
        action=action,
        new_sl=new_sl,
        chandelier=chandelier,
        time_stop_note=time_stop_note,
    )
