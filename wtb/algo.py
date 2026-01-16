from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .indicators import adx, atr, ema
from .structure import recent_range


@dataclass
class AlgoPlan:
    symbol: str
    side: str
    setup_type: str
    entry_zone: str
    stop_loss: float
    take_profits: List[float]
    expected_r: float
    score_tradeability: float
    score_setup: float
    late_atr: float
    status_hint: str
    flags: List[str]
    current_price: float = 0.0  # Added for UX clarity
    distance_pct: float = 0.0  # Added for UX clarity
    distance_atr: float = 0.0  # Added for UX clarity
    distance_direction: str = ""  # Added for UX clarity

    def to_dict(self) -> Dict[str, object]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "setup_type": self.setup_type,
            "entry_zone": self.entry_zone,
            "stop_loss": self.stop_loss,
            "take_profits": list(self.take_profits),
            "expected_r": self.expected_r,
            "score_tradeability": self.score_tradeability,
            "score_setup": self.score_setup,
            "late_atr": self.late_atr,
            "status_hint": self.status_hint,
            "flags": list(self.flags),
            "current_price": self.current_price,
            "distance_to_entry": {
                "percent": self.distance_pct,
                "atr_units": self.distance_atr,
                "direction": self.distance_direction,
            },
        }


def _pct_spread(bid: float, ask: float) -> float:
    mid = (bid + ask) / 2.0 if (bid + ask) else 0.0
    if mid == 0:
        return 0.0
    return abs(ask - bid) / mid * 100.0


def prescore(volume_usdt: float, spread_pct: float, atrp_4h: float, adx_4h: float) -> float:
    """Soft prescore (0-100). Never hard-reject; we just rank."""
    # volume: log scale
    v = max(volume_usdt, 1.0)
    vol_score = np.clip((np.log10(v) - 7.0) * 25.0, 0, 40)  # ~10M=>0, 100M=>25, 1B=>50 cap
    spread_score = np.clip(20.0 - spread_pct * 80.0, 0, 20)  # 0.05%=>16, 0.20%=>4
    # ATR%: prefer 1.5-6%; too low chop, too high unstable
    atr_score = 0.0
    if atrp_4h < 0.8:
        atr_score = 2.0
    elif atrp_4h <= 6.0:
        atr_score = 20.0
    elif atrp_4h <= 15.0:
        atr_score = 12.0
    else:
        atr_score = 5.0
    # ADX: moderate/trending is good
    adx_score = np.clip(adx_4h, 0, 40) / 2.0  # 0-20
    return float(np.clip(vol_score + spread_score + atr_score + adx_score, 0, 100))


def choose_setup_type(
    side_mode: str,
    adx_4h: float,
    atrp_4h: float,
    btc_trend: str,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    current_price: float,
    rng_low: float,
    rng_high: float,
    rng_h: float,
    atr_last: float,
) -> str:
    """Heuristic setup label.

    Important: this label is *descriptive*; the algo still produces a full plan.

    Why we need more than ADX:
    - If we only use ADX, almost everything becomes TREND_PULLBACK in strong markets.
    - We also look for: range sweeps/reclaims, breakouts, and extreme volatility.
    """

    if len(close) < 3:
        # Safe fallback
        return "TREND_PULLBACK" if adx_4h >= 25 else "RANGE_SWEEP_RECLAIM"

    # Get last 2 candles
    last_close = float(close[-1])
    last_high = float(high[-1])
    last_low = float(low[-1])
    last_open = float(close[-1]) if len(close) < 2 else float(close[-2])  # approximation
    prev_close = float(close[-2]) if len(close) >= 2 else last_close

    atr = atr_last if atr_last > 0 else max(1e-9, float(atrp_4h) / 100.0 * current_price)

    rng_mid = (rng_low + rng_high) / 2.0
    rng_w_pct = (rng_h / max(1e-9, rng_mid)) * 100.0

    trending = adx_4h >= 25

    # "Range-like" override: ADX can remain elevated after a trend leg.
    # If the last N candles are compressed (tight range), treat it as range/coil.
    # This prevents the bot from labeling everything as TREND_PULLBACK.
    # Heuristic: range width (pct) small relative to current volatility.
    range_like = (rng_w_pct <= max(4.0, atrp_4h * 2.0))
    very_volatile = (atrp_4h >= 8.0) or ((last_high - last_low) >= 2.0 * atr)

    # Sweep + reclaim (wick beyond range edge, close back inside)
    # NOTE: We intentionally do NOT require (not trending) here.
    # In crypto, a single expansion leg can inflate ADX even while the market is still
    # operating like a range with liquidity sweeps.
    sweep_buf = 0.15 * atr
    reclaim_buf = 0.05 * atr
    if side_mode == "LONG":
        swept = last_low < (rng_low - sweep_buf)
        reclaimed = last_close > (rng_low + reclaim_buf)
        bullish_close = last_close >= last_open
        if swept and reclaimed and bullish_close:
            return "RANGE_SWEEP_RECLAIM"
    else:
        swept = last_high > (rng_high + sweep_buf)
        reclaimed = last_close < (rng_high - reclaim_buf)
        bearish_close = last_close <= last_open
        if swept and reclaimed and bearish_close:
            return "RANGE_SWEEP_RECLAIM"

    # Breakout / retest detection (prefer this label when price is at/above the range edge
    # and the range is relatively tight). Even if the *fresh* break already happened,
    # the plan is still to wait for the retest, so the label remains BREAKOUT_RETEST.
    breakout_buf = 0.10 * atr
    tight_range = rng_w_pct <= 8.0  # heuristic; works reasonably across alts
    near_edge_frac = 0.20  # fraction of range height considered "near" the edge
    # Breakout / retest can happen both in trending legs and in compressed ranges.
    # If we only gate on ADX>=25, everything becomes TREND_PULLBACK.
    if side_mode == "LONG":
        broke_fresh = last_close > (rng_high + breakout_buf) and prev_close <= rng_high
        near_high = last_close >= (rng_high - near_edge_frac * rng_h)
        if (tight_range or range_like) and (broke_fresh or near_high) and (not very_volatile):
            return "BREAKOUT_RETEST"
    else:
        broke_fresh = last_close < (rng_low - breakout_buf) and prev_close >= rng_low
        near_low = last_close <= (rng_low + near_edge_frac * rng_h)
        if (tight_range or range_like) and (broke_fresh or near_low) and (not very_volatile):
            return "BREAKOUT_RETEST"

    # Volatility fade: very high ATR% or oversized candle.
    # Allow this even when trending: late-stage blow-off candles can still be faded.
    if very_volatile and (atrp_4h >= 10.0 or (last_high - last_low) >= 2.5 * atr):
        return "VOLATILITY_FADE"

    # Default buckets
    if range_like:
        # compressed / coil market -> treat as range behavior by default
        return "RANGE_SWEEP_RECLAIM"
    if trending:
        # If BTC is clearly against our side, we still keep the label, but penalties are handled elsewhere.
        return "TREND_PULLBACK"

    # In non-trending markets, prefer range behaviors.
    return "RANGE_SWEEP_RECLAIM" if rng_w_pct >= 1.0 else "VOLATILITY_FADE"


def build_algo_plan(
    symbol: str,
    side_mode: str,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    current_price: float,
    bid: float,
    ask: float,
    volume_usdt: float,
    btc_trend: str,
    range_lookback: int = 48,
    atr_period: int = 14,
    adx_period: int = 14,
    late_ok_atr: float = 0.15,
    late_watch_atr: float = 0.25,
) -> AlgoPlan:
    # basic metrics
    tr = _pct_spread(bid, ask)
    atr_vals = atr(high, low, close, period=atr_period)
    atr_last = float(atr_vals[-1]) if len(atr_vals) else 0.0
    atrp_4h = (atr_last / current_price * 100.0) if current_price else 0.0
    adx_vals = adx(high, low, close, period=adx_period)
    adx_last = float(adx_vals[-1]) if len(adx_vals) else 0.0

    flags: List[str] = []
    if volume_usdt >= 500_000_000:
        flags.append("HIGH_LIQUIDITY")
    elif volume_usdt >= 100_000_000:
        flags.append("MID_LIQUIDITY")
    else:
        flags.append("LOW_LIQUIDITY")
    if tr <= 0.05:
        flags.append("TIGHT_SPREAD")
    if 1.0 <= atrp_4h <= 6.0:
        flags.append("GOOD_ATR")
    if adx_last >= 25:
        flags.append("TRENDING")

    tradeability = prescore(volume_usdt, tr, atrp_4h, adx_last)

    # range bounds (execution tf)
    rng = recent_range(high, low, lookback=range_lookback)
    rng_low = float(rng.low)
    rng_high = float(rng.high)
    rng_h = max(rng.height, 0.0)

    setup_type = choose_setup_type(
        side_mode,
        adx_last,
        atrp_4h,
        btc_trend,
        close,
        high,
        low,
        current_price,
        rng_low,
        rng_high,
        rng_h,
        atr_last,
    )

    # deterministic plan (always yields TP list)
    if rng_h == 0:
        # fallback to ATR bands
        rng_h = max(atr_last * 2.5, 1e-9)
        rng_low = current_price - rng_h / 2
        rng_high = current_price + rng_h / 2

    # --- Entry/SL/TP (deterministic)
    # IMPORTANT: These are *planning* levels (wait for price to come to the zone).
    # Late rule should only trigger when price has moved away from the zone *past it*.

    # EMA anchor (execution TF) — used for TREND_PULLBACK plans
    ema_fast = float(ema(close, 50)[-1]) if len(close) >= 50 else float("nan")
    if not np.isfinite(ema_fast):
        ema_fast = (rng_low + rng_high) / 2

    def _clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    if setup_type == "TREND_PULLBACK":
        # Anchor around EMA(50) on 4H; aim to buy/sell the pullback, not the extreme range edge.
        if side_mode == "LONG":
            entry_low = ema_fast - 0.60 * atr_last
            entry_high = ema_fast + 0.20 * atr_last
            sl = entry_low - 0.60 * atr_last
        else:
            entry_low = ema_fast - 0.20 * atr_last
            entry_high = ema_fast + 0.60 * atr_last
            sl = entry_high + 0.60 * atr_last

    elif setup_type == "BREAKOUT_RETEST":
        # Retest around the breakout level (range edge)
        if side_mode == "LONG":
            level = rng_high
            entry_low = level - 0.40 * atr_last
            entry_high = level + 0.10 * atr_last
            sl = level - 1.00 * atr_last
        else:
            level = rng_low
            entry_low = level - 0.10 * atr_last
            entry_high = level + 0.40 * atr_last
            sl = level + 1.00 * atr_last

    elif setup_type == "VOLATILITY_FADE":
        # Fade extremes: enter near range edges with smaller zone
        if side_mode == "LONG":
            entry_low = rng_low
            entry_high = rng_low + 0.15 * rng_h
            sl = rng_low - 0.35 * atr_last
        else:
            entry_low = rng_high - 0.15 * rng_h
            entry_high = rng_high
            sl = rng_high + 0.35 * atr_last

    else:  # RANGE_SWEEP_RECLAIM
        if side_mode == "LONG":
            entry_low = rng_low
            entry_high = rng_low + 0.25 * rng_h
            sl = rng_low - 0.35 * atr_last
        else:
            entry_low = rng_high - 0.25 * rng_h
            entry_high = rng_high
            sl = rng_high + 0.35 * atr_last

    # Clamp zone to reasonable bounds around the recent range (avoid nonsense zones)
    entry_low = _clamp(float(entry_low), rng_low - 0.50 * atr_last, rng_high + 0.50 * atr_last)
    entry_high = _clamp(float(entry_high), rng_low - 0.50 * atr_last, rng_high + 0.50 * atr_last)
    if entry_high < entry_low:
        entry_low, entry_high = entry_high, entry_low

    # TPs: deterministic + structure-aware.
    # We prefer structure targets (range mid / range edge / measured move) over pure R-multiples.
    entry_mid = (entry_low + entry_high) / 2
    risk = max(abs(entry_mid - sl), 1e-9)
    rng_mid = (rng_low + rng_high) / 2.0

    if setup_type == "BREAKOUT_RETEST":
        # measured move style
        if side_mode == "LONG":
            tp1 = rng_high + 0.50 * rng_h
            tp2 = rng_high + 1.00 * rng_h
            tp3 = rng_high + 1.50 * rng_h
        else:
            tp1 = rng_low - 0.50 * rng_h
            tp2 = rng_low - 1.00 * rng_h
            tp3 = rng_low - 1.50 * rng_h
    elif setup_type == "RANGE_SWEEP_RECLAIM":
        # range targets: mid -> far edge -> extension
        if side_mode == "LONG":
            tp1 = max(entry_mid + 0.8 * risk, min(rng_mid, rng_high))
            tp2 = max(tp1 + 0.2 * risk, rng_high)
            tp3 = rng_high + 0.50 * rng_h
        else:
            tp1 = min(entry_mid - 0.8 * risk, max(rng_mid, rng_low))
            tp2 = min(tp1 - 0.2 * risk, rng_low)
            tp3 = rng_low - 0.50 * rng_h
    elif setup_type == "TREND_PULLBACK":
        # trend targets: mid -> prior swing edge -> continuation extension
        if side_mode == "LONG":
            tp1 = max(entry_mid + 1.0 * risk, min(rng_mid, rng_high))
            tp2 = max(tp1 + 0.3 * risk, rng_high)
            tp3 = rng_high + 0.75 * rng_h
        else:
            tp1 = min(entry_mid - 1.0 * risk, max(rng_mid, rng_low))
            tp2 = min(tp1 - 0.3 * risk, rng_low)
            tp3 = rng_low - 0.75 * rng_h
    elif setup_type == "VOLATILITY_FADE":
        # volatility fade: use more conservative R targets
        if side_mode == "LONG":
            tp1 = entry_mid + 1.2 * risk
            tp2 = entry_mid + 2.0 * risk
            tp3 = entry_mid + 2.8 * risk
        else:
            tp1 = entry_mid - 1.2 * risk
            tp2 = entry_mid - 2.0 * risk
            tp3 = entry_mid - 2.8 * risk
    else:
        # fallback R-multiples
        if side_mode == "LONG":
            tp1 = entry_mid + 1.5 * risk
            tp2 = entry_mid + 2.5 * risk
            tp3 = entry_mid + 3.5 * risk
        else:
            tp1 = entry_mid - 1.5 * risk
            tp2 = entry_mid - 2.5 * risk
            tp3 = entry_mid - 3.5 * risk

    # Late distance: only counts if price has moved beyond the zone (i.e., chase / missed entry)
    if side_mode == "LONG":
        late_atr = (current_price - entry_high) / atr_last if current_price > entry_high and atr_last else 0.0
    else:
        late_atr = (entry_low - current_price) / atr_last if current_price < entry_low and atr_last else 0.0

    # status hint from late rule
    status_hint = "OK"
    if late_atr > late_watch_atr:
        status_hint = "WATCH_PULLBACK"
        flags.append("LATE_PULLBACK")
    elif late_atr > late_ok_atr:
        status_hint = "WATCH_LATE"
        flags.append("LATE")

    # expected R uses tp2 by default
    reward = abs(tp2 - entry_mid)
    expected_r = float(reward / risk) if risk else 0.0

    # Distance metrics for UX clarity
    distance_pct = abs(current_price - entry_mid) / max(current_price, 1e-9) * 100.0
    distance_atr_units = abs(current_price - entry_mid) / max(atr_last, 1e-9)
    if side_mode == "LONG":
        distance_direction = "BELOW" if current_price < entry_low else ("INSIDE" if current_price <= entry_high else "ABOVE")
    else:
        distance_direction = "ABOVE" if current_price > entry_high else ("INSIDE" if current_price >= entry_low else "BELOW")

    # setup score (0–100): intent is *ranking*, not rejecting.
    # Trend pullback prefers strong ADX but not insane ATR. Ranges prefer low ADX + clean volatility.
    setup_score = 55.0
    if setup_type == "TREND_PULLBACK":
        setup_score = 60.0 + _clamp(adx_last - 20.0, 0.0, 30.0) * 1.0
    elif setup_type == "BREAKOUT_RETEST":
        setup_score = 58.0 + _clamp(adx_last - 18.0, 0.0, 35.0) * 0.8
    elif setup_type == "RANGE_SWEEP_RECLAIM":
        setup_score = 56.0 + _clamp(25.0 - adx_last, 0.0, 20.0) * 1.0
    else:  # VOLATILITY_FADE
        setup_score = 55.0 + _clamp(25.0 - adx_last, 0.0, 20.0) * 0.8

    # volatility band (prefer 1–6% ATR on 4H)
    if 1.0 <= atrp_4h <= 6.0:
        setup_score += 6.0
    elif atrp_4h > 15.0:
        setup_score -= 8.0

    if btc_trend == "BEARISH" and side_mode == "LONG":
        # not reject; just down-weight
        setup_score -= 10.0
        flags.append("BTC_BEAR_HEADWIND")
    if btc_trend == "BULLISH" and side_mode == "SHORT":
        setup_score -= 10.0
        flags.append("BTC_BULL_HEADWIND")

    return AlgoPlan(
        symbol=symbol,
        side=side_mode,
        setup_type=setup_type,
        entry_zone=f"{entry_low:.4g}-{entry_high:.4g}",
        stop_loss=float(sl),
        take_profits=[float(tp1), float(tp2), float(tp3)],
        expected_r=float(expected_r),
        score_tradeability=float(tradeability),
        score_setup=float(np.clip(setup_score, 0, 100)),
        late_atr=float(late_atr),
        status_hint=status_hint,
        flags=flags,
        current_price=float(current_price),
        distance_pct=float(distance_pct),
        distance_atr=float(distance_atr_units),
        distance_direction=distance_direction,
    )
