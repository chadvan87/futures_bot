"""DCA Discovery scoring components and penalties.

All scoring is deterministic. Each component returns a score in [0, 100].
Penalties are applied after weighted sum of components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .indicators import atr, ema, adx


@dataclass
class DCAScoreResult:
    """Result of DCA scoring for a single symbol."""
    symbol: str
    side: str
    tier: str  # CORE, MID, EXPLORE
    dca_score: float
    score_components: Dict[str, float]
    penalties: Dict[str, Any]
    eligibility_flags: List[str]
    risk_flags: List[str]
    status: str  # RUN, WATCH, SKIP

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "tier": self.tier,
            "dca_score": round(self.dca_score, 2),
            "score_components": {k: round(v, 2) for k, v in self.score_components.items()},
            "penalties": self.penalties,
            "eligibility_flags": self.eligibility_flags,
            "risk_flags": self.risk_flags,
            "status": self.status,
        }


def assign_tier(
    vol24h_usdt: float,
    spread_pct: float,
    tier_cfg: Dict[str, Dict[str, float]],
) -> Tuple[str, bool]:
    """Assign tier based on volume and spread thresholds.

    Returns (tier_name, is_eligible).
    """
    core = tier_cfg.get("core", {"min_volume_usdt": 500_000_000, "max_spread_pct": 0.08})
    mid = tier_cfg.get("mid", {"min_volume_usdt": 100_000_000, "max_spread_pct": 0.15})
    explore = tier_cfg.get("explore", {"min_volume_usdt": 25_000_000, "max_spread_pct": 0.30})

    # Check CORE first (most stringent)
    if vol24h_usdt >= core["min_volume_usdt"] and spread_pct <= core["max_spread_pct"]:
        return "CORE", True

    # Check MID
    if vol24h_usdt >= mid["min_volume_usdt"] and spread_pct <= mid["max_spread_pct"]:
        return "MID", True

    # Check EXPLORE
    if vol24h_usdt >= explore["min_volume_usdt"] and spread_pct <= explore["max_spread_pct"]:
        return "EXPLORE", True

    # Does not meet minimum thresholds
    return "INELIGIBLE", False


def microstructure_score(
    vol24h_usdt: float,
    spread_pct: float,
    tier: str,
) -> float:
    """Score based on spread quality and volume persistence.

    Higher volume and tighter spread = better for DCA execution.
    Score range: 0-100.
    """
    score = 50.0

    # Spread component (lower is better)
    # Ideal spread < 0.05%, penalize beyond 0.15%
    if spread_pct <= 0.03:
        score += 25.0
    elif spread_pct <= 0.05:
        score += 20.0
    elif spread_pct <= 0.10:
        score += 10.0
    elif spread_pct <= 0.15:
        score += 0.0
    elif spread_pct <= 0.25:
        score -= 10.0
    else:
        score -= 20.0

    # Volume component (log scale)
    # Higher volume = better liquidity for DCA fills
    log_vol = np.log10(max(vol24h_usdt, 1.0))
    # 8 = $100M, 9 = $1B, 10 = $10B
    if log_vol >= 9.5:  # > $3B
        score += 25.0
    elif log_vol >= 9.0:  # > $1B
        score += 20.0
    elif log_vol >= 8.5:  # > $300M
        score += 15.0
    elif log_vol >= 8.0:  # > $100M
        score += 10.0
    elif log_vol >= 7.5:  # > $30M
        score += 5.0
    else:
        score -= 5.0

    # Tier bonus/penalty
    if tier == "CORE":
        score += 5.0
    elif tier == "EXPLORE":
        score -= 5.0

    return float(max(0.0, min(100.0, score)))


def mean_reversion_score(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ema_period: int = 50,
) -> float:
    """Score based on mean reversion tendency.

    Measures wick reclaim frequency and displacement/reversion around EMA.
    Higher score = more mean-reverting behavior (good for DCA grids).
    """
    if len(close) < ema_period + 20:
        return 50.0  # Neutral if insufficient data

    score = 50.0

    # Calculate EMA
    ema50 = ema(close, ema_period)

    # 1. Wick reclaim tendency
    # Count bars where wick exceeded body but closed back
    wick_reclaims = 0
    total_wicks = 0
    for i in range(-50, 0):
        body_top = max(close[i], close[i-1] if i > -len(close) else close[i])
        body_bot = min(close[i], close[i-1] if i > -len(close) else close[i])
        wick_up = high[i] - body_top
        wick_down = body_bot - low[i]
        body_size = abs(close[i] - close[i-1]) if i > -len(close) else 0.001

        # Significant wick = wick > 0.5 * body
        if body_size > 0 and (wick_up > 0.5 * body_size or wick_down > 0.5 * body_size):
            total_wicks += 1
            # Reclaim = close inside previous range
            if low[i-1] <= close[i] <= high[i-1] if i > -len(close) else True:
                wick_reclaims += 1

    if total_wicks > 0:
        reclaim_ratio = wick_reclaims / total_wicks
        score += (reclaim_ratio - 0.5) * 30.0  # [-15, +15]

    # 2. EMA touch/bounce frequency
    # Count touches of EMA band (within 1 ATR) that bounced
    atr_series = atr(high, low, close, 14)
    if len(atr_series) > 0:
        recent_atr = float(atr_series[-1])
        ema_touches = 0
        ema_bounces = 0

        for i in range(-40, -1):
            if i + 1 >= 0:
                continue
            price = close[i]
            ema_val = ema50[i]
            dist = abs(price - ema_val)

            # Touch = within 0.5 ATR of EMA
            if dist <= 0.5 * recent_atr:
                ema_touches += 1
                # Bounce = moved away by at least 0.3 ATR next bar
                next_dist = abs(close[i + 1] - ema50[i + 1])
                if next_dist > 0.3 * recent_atr:
                    ema_bounces += 1

        if ema_touches > 0:
            bounce_ratio = ema_bounces / ema_touches
            score += (bounce_ratio - 0.5) * 20.0  # [-10, +10]

    # 3. Displacement tendency
    # Measure how often large moves revert
    returns = np.diff(close) / np.maximum(close[:-1], 1e-12)
    if len(returns) > 20:
        large_moves = np.where(np.abs(returns) > 0.02)[0]  # > 2% moves
        reversions = 0
        for idx in large_moves:
            if idx + 3 < len(returns):
                move_dir = np.sign(returns[idx])
                # Reversion = opposite move in next 3 bars
                subsequent = returns[idx + 1 : idx + 4]
                if np.any(np.sign(subsequent) == -move_dir):
                    reversions += 1

        if len(large_moves) > 0:
            reversion_ratio = reversions / len(large_moves)
            score += (reversion_ratio - 0.4) * 15.0  # [-6, +9]

    return float(max(0.0, min(100.0, score)))


def volatility_fit_score(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr_period: int = 14,
) -> float:
    """Score based on ATR% suitability for DCA grids.

    Ideal ATR%: 2-6% (enough movement for grid profit, not too explosive).
    Too low (<1%): Dead, no profit opportunity.
    Too high (>10%): Too risky, liquidation danger.
    """
    if len(close) < atr_period + 10:
        return 50.0

    atr_series = atr(high, low, close, atr_period)
    atr_pct = (atr_series[-1] / close[-1]) * 100.0 if close[-1] > 0 else 0.0

    # Also compute volatility consistency (std of recent ATR%)
    recent_atrs = atr_series[-20:]
    recent_closes = close[-20:]
    atr_pcts = (recent_atrs / np.maximum(recent_closes, 1e-12)) * 100.0
    atr_std = float(np.std(atr_pcts)) if len(atr_pcts) > 1 else 0.0

    score = 50.0

    # ATR% scoring (bell curve around 3-5%)
    if 3.0 <= atr_pct <= 5.0:
        score += 30.0  # Ideal range
    elif 2.0 <= atr_pct <= 6.0:
        score += 20.0  # Good range
    elif 1.5 <= atr_pct <= 8.0:
        score += 10.0  # Acceptable
    elif 1.0 <= atr_pct <= 10.0:
        score += 0.0   # Borderline
    elif atr_pct < 1.0:
        score -= 20.0  # Too dead
    else:  # > 10%
        score -= 25.0  # Too explosive

    # Volatility consistency bonus
    # Lower std = more predictable for grid sizing
    if atr_std < 0.5:
        score += 15.0
    elif atr_std < 1.0:
        score += 10.0
    elif atr_std < 2.0:
        score += 5.0
    elif atr_std > 4.0:
        score -= 10.0

    return float(max(0.0, min(100.0, score)))


def derivatives_health_score(
    funding_now: float,
    funding_median: float,
    oi_change_1h: float,
    side: str,
) -> Tuple[float, List[str]]:
    """Score based on funding neutrality and OI stability.

    For DCA: prefer neutral funding (not crowded either direction).
    Penalize OI blow-off spikes that indicate potential cascade liquidations.

    Returns (score, flags).
    """
    score = 50.0
    flags: List[str] = []

    # Funding neutrality (best = close to 0)
    abs_funding = abs(funding_now)
    if abs_funding < 0.0001:
        score += 25.0
        flags.append("FUNDING_NEUTRAL")
    elif abs_funding < 0.0003:
        score += 15.0
        flags.append("FUNDING_LOW")
    elif abs_funding < 0.0005:
        score += 5.0
        flags.append("FUNDING_MODERATE")
    elif abs_funding < 0.001:
        score -= 5.0
        flags.append("FUNDING_HIGH")
    else:
        score -= 15.0
        flags.append("FUNDING_EXTREME")

    # Side alignment with funding
    # For LONG DCA: negative funding is favorable (paid to hold)
    # For SHORT DCA: positive funding is favorable
    if side == "LONG":
        if funding_now < -0.0003:
            score += 10.0
            flags.append("FUNDING_FAVORABLE_LONG")
        elif funding_now > 0.0005:
            score -= 10.0
            flags.append("FUNDING_UNFAVORABLE_LONG")
    elif side == "SHORT":
        if funding_now > 0.0003:
            score += 10.0
            flags.append("FUNDING_FAVORABLE_SHORT")
        elif funding_now < -0.0005:
            score -= 10.0
            flags.append("FUNDING_UNFAVORABLE_SHORT")

    # OI stability (penalize spikes)
    if abs(oi_change_1h) < 0.02:
        score += 15.0
        flags.append("OI_STABLE")
    elif abs(oi_change_1h) < 0.05:
        score += 5.0
        flags.append("OI_MODERATE_CHANGE")
    elif oi_change_1h > 0.08:
        score -= 15.0
        flags.append("OI_SPIKE")
    elif oi_change_1h < -0.08:
        score -= 10.0
        flags.append("OI_FLUSH")
    else:
        flags.append("OI_VOLATILE")

    return float(max(0.0, min(100.0, score))), flags


def context_score(
    side: str,
    btc_trend: str,
    btc_atrp: float,
) -> float:
    """Score based on BTC regime compatibility.

    For DCA:
    - LONG: prefer BULLISH or RANGE BTC (accumulation friendly)
    - SHORT: prefer BEARISH or RANGE BTC

    BTC volatility state also matters for DCA timing.
    """
    score = 50.0

    # BTC trend alignment
    if side == "LONG":
        if btc_trend == "BULLISH":
            score += 25.0
        elif btc_trend == "NEUTRAL" or btc_trend == "RANGE":
            score += 10.0
        elif btc_trend == "BEARISH":
            score -= 10.0  # Softer penalty for DCA (accumulation in downtrend can work)
    elif side == "SHORT":
        if btc_trend == "BEARISH":
            score += 25.0
        elif btc_trend == "NEUTRAL" or btc_trend == "RANGE":
            score += 10.0
        elif btc_trend == "BULLISH":
            score -= 10.0

    # BTC volatility adjustment
    # Moderate BTC volatility is good for DCA (movement creates opportunities)
    if 1.5 <= btc_atrp <= 4.0:
        score += 10.0
    elif 1.0 <= btc_atrp <= 6.0:
        score += 5.0
    elif btc_atrp < 0.8:
        score -= 5.0  # Too quiet
    elif btc_atrp > 8.0:
        score -= 10.0  # Too chaotic

    return float(max(0.0, min(100.0, score)))


def compute_penalties(
    side: str,
    btc_trend: str,
    funding_now: float,
    spread_pct: float,
    atr_pct: float,
    adx_val: float,
    tier: str,
    penalty_cfg: Dict[str, float],
) -> Tuple[float, List[Dict[str, Any]]]:
    """Compute penalty deductions for DCA score.

    Returns (total_penalty, penalty_items).
    """
    items: List[Dict[str, Any]] = []
    total = 0.0

    btc_headwind = float(penalty_cfg.get("btc_headwind", 6.0))
    extreme_funding = float(penalty_cfg.get("extreme_funding", 4.0))
    liquidity_stress = float(penalty_cfg.get("liquidity_stress", 4.0))
    trend_runaway = float(penalty_cfg.get("trend_runaway", 5.0))
    max_total = float(penalty_cfg.get("max_total", 15.0))

    # BTC headwind penalty
    if side == "LONG" and btc_trend == "BEARISH":
        total += btc_headwind
        items.append({"name": "btc_headwind", "value": btc_headwind, "reason": "LONG in BEARISH BTC"})
    elif side == "SHORT" and btc_trend == "BULLISH":
        total += btc_headwind
        items.append({"name": "btc_headwind", "value": btc_headwind, "reason": "SHORT in BULLISH BTC"})

    # Extreme funding penalty
    if abs(funding_now) >= 0.001:
        total += extreme_funding
        items.append({"name": "extreme_funding", "value": extreme_funding, "reason": f"funding={funding_now:.6f}"})

    # Liquidity stress penalty (high spread for tier)
    spread_thresholds = {"CORE": 0.06, "MID": 0.12, "EXPLORE": 0.25}
    tier_spread_limit = spread_thresholds.get(tier, 0.15)
    if spread_pct > tier_spread_limit * 1.5:
        total += liquidity_stress
        items.append({"name": "liquidity_stress", "value": liquidity_stress, "reason": f"spread={spread_pct:.3f}%"})

    # Trend runaway penalty
    # High ADX + high ATR = strong trend, bad for DCA grids
    if adx_val >= 35 and atr_pct >= 6.0:
        total += trend_runaway
        items.append({"name": "trend_runaway", "value": trend_runaway, "reason": f"ADX={adx_val:.1f}, ATR%={atr_pct:.1f}"})

    # Cap total penalty
    total = min(total, max_total)

    return total, items


def compute_dca_score(
    symbol: str,
    side: str,
    tier: str,
    vol24h_usdt: float,
    spread_pct: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    funding_now: float,
    funding_median: float,
    oi_change_1h: float,
    btc_trend: str,
    btc_atrp: float,
    weights_cfg: Dict[str, float],
    penalty_cfg: Dict[str, float],
    min_dca_score: float = 55.0,
) -> DCAScoreResult:
    """Compute full DCA score for a symbol.

    Returns DCAScoreResult with all components and final classification.
    """
    eligibility_flags: List[str] = []
    risk_flags: List[str] = []

    # Check basic eligibility
    if tier == "INELIGIBLE":
        return DCAScoreResult(
            symbol=symbol,
            side=side,
            tier=tier,
            dca_score=0.0,
            score_components={},
            penalties={"items": [], "total": 0.0},
            eligibility_flags=["BELOW_MIN_THRESHOLDS"],
            risk_flags=[],
            status="SKIP",
        )

    eligibility_flags.append(f"TIER_{tier}")

    # Compute component scores
    micro_score = microstructure_score(vol24h_usdt, spread_pct, tier)

    reversion_score = mean_reversion_score(high, low, close)

    vol_fit_score = volatility_fit_score(high, low, close)

    deriv_score, deriv_flags = derivatives_health_score(funding_now, funding_median, oi_change_1h, side)
    risk_flags.extend([f for f in deriv_flags if "EXTREME" in f or "UNFAVORABLE" in f or "SPIKE" in f])

    ctx_score = context_score(side, btc_trend, btc_atrp)

    # Get weights
    w_micro = float(weights_cfg.get("microstructure", 30.0))
    w_reversion = float(weights_cfg.get("mean_reversion", 25.0))
    w_vol = float(weights_cfg.get("volatility_fit", 20.0))
    w_deriv = float(weights_cfg.get("derivatives_health", 15.0))
    w_ctx = float(weights_cfg.get("context", 10.0))

    total_weight = w_micro + w_reversion + w_vol + w_deriv + w_ctx
    if total_weight == 0:
        total_weight = 100.0

    # Weighted sum
    raw_score = (
        micro_score * w_micro +
        reversion_score * w_reversion +
        vol_fit_score * w_vol +
        deriv_score * w_deriv +
        ctx_score * w_ctx
    ) / total_weight

    # Compute penalties
    atr_series = atr(high, low, close, 14)
    atr_pct = (atr_series[-1] / close[-1]) * 100.0 if len(close) > 0 and close[-1] > 0 else 0.0
    adx_series = adx(high, low, close, 14)
    adx_val = float(adx_series[-1]) if len(adx_series) > 0 else 0.0

    total_penalty, penalty_items = compute_penalties(
        side=side,
        btc_trend=btc_trend,
        funding_now=funding_now,
        spread_pct=spread_pct,
        atr_pct=atr_pct,
        adx_val=adx_val,
        tier=tier,
        penalty_cfg=penalty_cfg,
    )

    for item in penalty_items:
        risk_flags.append(f"PENALTY_{item['name'].upper()}")

    # Final score
    final_score = max(0.0, min(100.0, raw_score - total_penalty))

    # Determine status
    if final_score >= min_dca_score and not any("EXTREME" in f for f in risk_flags):
        status = "RUN"
    elif final_score >= min_dca_score * 0.85:
        status = "WATCH"
    else:
        status = "SKIP"

    # Check for hard blockers
    if abs(funding_now) >= 0.002:  # 0.2% funding = hard block
        status = "SKIP"
        risk_flags.append("HARD_BLOCK_EXTREME_FUNDING")

    if spread_pct > 0.5:  # 0.5% spread = hard block
        status = "SKIP"
        risk_flags.append("HARD_BLOCK_SPREAD")

    return DCAScoreResult(
        symbol=symbol,
        side=side,
        tier=tier,
        dca_score=final_score,
        score_components={
            "microstructure": micro_score,
            "mean_reversion": reversion_score,
            "volatility_fit": vol_fit_score,
            "derivatives_health": deriv_score,
            "context": ctx_score,
        },
        penalties={"items": penalty_items, "total": total_penalty},
        eligibility_flags=eligibility_flags,
        risk_flags=risk_flags,
        status=status,
    )


def suggest_dca_profile(
    atr_pct: float,
    tier: str,
    dca_score: float,
) -> Dict[str, Any]:
    """Suggest DCA grid profile based on volatility and tier.

    Returns profile hints for grid step, max layers, size multiplier.
    """
    # Profile selection
    if dca_score >= 70 and tier in ("CORE", "MID"):
        profile = "BALANCED"
    elif dca_score >= 60 or tier == "CORE":
        profile = "CONSERVATIVE"
    else:
        profile = "AGGRESSIVE"  # Higher risk, need tighter management

    # Grid step based on ATR%
    # Rule: grid_step ~= 0.3-0.5 * ATR%
    if atr_pct < 2.0:
        grid_step_pct = 0.5
        max_layers = 8
    elif atr_pct < 4.0:
        grid_step_pct = round(atr_pct * 0.35, 2)
        max_layers = 10
    elif atr_pct < 6.0:
        grid_step_pct = round(atr_pct * 0.40, 2)
        max_layers = 12
    else:
        grid_step_pct = round(atr_pct * 0.45, 2)
        max_layers = 15

    # Size multiplier for DCA layers
    if profile == "CONSERVATIVE":
        size_mult = 1.2
    elif profile == "BALANCED":
        size_mult = 1.5
    else:  # AGGRESSIVE
        size_mult = 2.0

    # Tier adjustments
    if tier == "EXPLORE":
        max_layers = min(max_layers, 8)  # Cap layers for risky coins
        size_mult = min(size_mult, 1.3)  # Smaller position scaling

    return {
        "profile": profile,
        "grid_step_pct_hint": grid_step_pct,
        "max_layers_hint": max_layers,
        "size_multiplier_hint": size_mult,
    }


def compute_kill_switch_conditions(
    side: str,
    current_price: float,
    atr_val: float,
    tier: str,
) -> List[str]:
    """Generate kill-switch conditions for DCA bot safety.

    These are deterministic rules to stop the DCA bot.
    """
    conditions: List[str] = []

    # Price-based kill switch
    atr_mult = 4.0 if tier == "CORE" else (3.5 if tier == "MID" else 3.0)

    if side == "LONG":
        kill_price = current_price - (atr_mult * atr_val)
        conditions.append(f"STOP if price < {kill_price:.6g} (current - {atr_mult}*ATR)")
        conditions.append("STOP if 3 consecutive 4H closes below entry average")
    else:
        kill_price = current_price + (atr_mult * atr_val)
        conditions.append(f"STOP if price > {kill_price:.6g} (current + {atr_mult}*ATR)")
        conditions.append("STOP if 3 consecutive 4H closes above entry average")

    # Funding-based kill switch
    conditions.append("STOP if |funding| > 0.1% for 8+ consecutive hours")

    # Max layers exceeded
    max_layers = 15 if tier == "CORE" else (12 if tier == "MID" else 8)
    conditions.append(f"STOP if layers > {max_layers}")

    return conditions


@dataclass
class DCAExecutionPlan:
    """Deterministic execution plan with Entry, SL, and TPs."""
    entry_zone_lower: Optional[float]
    entry_zone_upper: Optional[float]
    stop_loss: Optional[float]
    take_profit_1: Optional[float]
    take_profit_2: Optional[float]
    take_profit_3: Optional[float]
    rr_tp1: Optional[float]  # Risk-reward ratio for TP1
    rr_tp2: Optional[float]  # Risk-reward ratio for TP2
    rr_tp3: Optional[float]  # Risk-reward ratio for TP3
    levels_valid: bool  # True if all levels are on correct side
    data_sufficient: bool  # True if we had enough data to compute
    validation_flags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_zone_lower": round(self.entry_zone_lower, 8) if self.entry_zone_lower is not None else None,
            "entry_zone_upper": round(self.entry_zone_upper, 8) if self.entry_zone_upper is not None else None,
            "stop_loss": round(self.stop_loss, 8) if self.stop_loss is not None else None,
            "take_profit_1": round(self.take_profit_1, 8) if self.take_profit_1 is not None else None,
            "take_profit_2": round(self.take_profit_2, 8) if self.take_profit_2 is not None else None,
            "take_profit_3": round(self.take_profit_3, 8) if self.take_profit_3 is not None else None,
            "rr_tp1": round(self.rr_tp1, 2) if self.rr_tp1 is not None else None,
            "rr_tp2": round(self.rr_tp2, 2) if self.rr_tp2 is not None else None,
            "rr_tp3": round(self.rr_tp3, 2) if self.rr_tp3 is not None else None,
            "levels_valid": self.levels_valid,
            "data_sufficient": self.data_sufficient,
            "validation_flags": self.validation_flags,
        }


def compute_execution_plan(
    side: str,
    current_price: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ema50_val: float,
    atr_val: float,
    tier: str,
) -> DCAExecutionPlan:
    """Compute deterministic Entry Zone, Stop Loss, and Take Profits.

    Rules (deterministic, no LLM):
    - Entry zone: based on EMA50 +/- ATR offset
    - Stop loss: based on ATR multiple from entry zone edge
    - Take profits: based on ATR multiples from entry (1R, 2R, 3R targets)

    For LONG:
      - Entry zone BELOW or AT current price (buy the dip)
      - SL BELOW entry zone
      - TPs ABOVE entry zone

    For SHORT:
      - Entry zone ABOVE or AT current price (sell the bounce)
      - SL ABOVE entry zone
      - TPs BELOW entry zone
    """
    validation_flags: List[str] = []

    # Check data sufficiency
    if len(close) < 50 or atr_val <= 0 or current_price <= 0:
        return DCAExecutionPlan(
            entry_zone_lower=None,
            entry_zone_upper=None,
            stop_loss=None,
            take_profit_1=None,
            take_profit_2=None,
            take_profit_3=None,
            rr_tp1=None,
            rr_tp2=None,
            rr_tp3=None,
            levels_valid=False,
            data_sufficient=False,
            validation_flags=["INSUFFICIENT_DATA"],
        )

    # ATR multipliers based on tier (tighter for CORE, wider for EXPLORE)
    if tier == "CORE":
        sl_atr_mult = 1.5
        tp1_atr_mult = 1.0
        tp2_atr_mult = 2.0
        tp3_atr_mult = 3.0
        entry_offset_mult = 0.5
    elif tier == "MID":
        sl_atr_mult = 1.8
        tp1_atr_mult = 1.2
        tp2_atr_mult = 2.4
        tp3_atr_mult = 3.5
        entry_offset_mult = 0.6
    else:  # EXPLORE
        sl_atr_mult = 2.0
        tp1_atr_mult = 1.5
        tp2_atr_mult = 2.8
        tp3_atr_mult = 4.0
        entry_offset_mult = 0.7

    if side == "LONG":
        # Entry zone: EMA50 - offset to EMA50 + small buffer
        entry_zone_lower = ema50_val - (entry_offset_mult * atr_val)
        entry_zone_upper = ema50_val + (0.3 * atr_val)

        # Use entry zone midpoint for SL/TP calculations
        entry_mid = (entry_zone_lower + entry_zone_upper) / 2.0

        # Stop loss below entry zone
        stop_loss = entry_zone_lower - (sl_atr_mult * atr_val)

        # Take profits above entry
        take_profit_1 = entry_mid + (tp1_atr_mult * atr_val)
        take_profit_2 = entry_mid + (tp2_atr_mult * atr_val)
        take_profit_3 = entry_mid + (tp3_atr_mult * atr_val)

        # Validate: SL < entry < TPs
        if stop_loss >= entry_zone_lower:
            validation_flags.append("SL_NOT_BELOW_ENTRY")
        if take_profit_1 <= entry_zone_upper:
            validation_flags.append("TP1_NOT_ABOVE_ENTRY")
        if take_profit_2 <= take_profit_1:
            validation_flags.append("TP2_NOT_ABOVE_TP1")
        if take_profit_3 <= take_profit_2:
            validation_flags.append("TP3_NOT_ABOVE_TP2")

    else:  # SHORT
        # Entry zone: EMA50 - small buffer to EMA50 + offset
        entry_zone_lower = ema50_val - (0.3 * atr_val)
        entry_zone_upper = ema50_val + (entry_offset_mult * atr_val)

        # Use entry zone midpoint for SL/TP calculations
        entry_mid = (entry_zone_lower + entry_zone_upper) / 2.0

        # Stop loss above entry zone
        stop_loss = entry_zone_upper + (sl_atr_mult * atr_val)

        # Take profits below entry
        take_profit_1 = entry_mid - (tp1_atr_mult * atr_val)
        take_profit_2 = entry_mid - (tp2_atr_mult * atr_val)
        take_profit_3 = entry_mid - (tp3_atr_mult * atr_val)

        # Validate: TPs < entry < SL
        if stop_loss <= entry_zone_upper:
            validation_flags.append("SL_NOT_ABOVE_ENTRY")
        if take_profit_1 >= entry_zone_lower:
            validation_flags.append("TP1_NOT_BELOW_ENTRY")
        if take_profit_2 >= take_profit_1:
            validation_flags.append("TP2_NOT_BELOW_TP1")
        if take_profit_3 >= take_profit_2:
            validation_flags.append("TP3_NOT_BELOW_TP2")

    # Calculate risk-reward ratios
    # Risk = distance from entry_mid to SL
    # Reward = distance from entry_mid to TP
    risk = abs(entry_mid - stop_loss)

    if risk > 0:
        rr_tp1 = abs(take_profit_1 - entry_mid) / risk
        rr_tp2 = abs(take_profit_2 - entry_mid) / risk
        rr_tp3 = abs(take_profit_3 - entry_mid) / risk
    else:
        rr_tp1 = None
        rr_tp2 = None
        rr_tp3 = None
        validation_flags.append("ZERO_RISK_DISTANCE")

    levels_valid = len(validation_flags) == 0

    return DCAExecutionPlan(
        entry_zone_lower=entry_zone_lower,
        entry_zone_upper=entry_zone_upper,
        stop_loss=stop_loss,
        take_profit_1=take_profit_1,
        take_profit_2=take_profit_2,
        take_profit_3=take_profit_3,
        rr_tp1=rr_tp1,
        rr_tp2=rr_tp2,
        rr_tp3=rr_tp3,
        levels_valid=levels_valid,
        data_sufficient=True,
        validation_flags=validation_flags,
    )


def format_execution_levels_text(
    symbol: str,
    side: str,
    plan: DCAExecutionPlan,
) -> str:
    """Format execution levels for text output."""
    if not plan.data_sufficient:
        return f"{symbol} ({side}): INSUFFICIENT DATA - Cannot compute levels"

    lines = []
    lines.append(f"{symbol} ({side}):")

    if plan.entry_zone_lower is not None and plan.entry_zone_upper is not None:
        lines.append(f"  Entry Zone: {plan.entry_zone_lower:.8g} - {plan.entry_zone_upper:.8g}")
    else:
        lines.append("  Entry Zone: NULL")

    lines.append(f"  Stop Loss: {plan.stop_loss:.8g}" if plan.stop_loss else "  Stop Loss: NULL")
    lines.append(f"  Take Profit 1: {plan.take_profit_1:.8g} (RR: {plan.rr_tp1:.2f})" if plan.take_profit_1 and plan.rr_tp1 else "  Take Profit 1: NULL")
    lines.append(f"  Take Profit 2: {plan.take_profit_2:.8g} (RR: {plan.rr_tp2:.2f})" if plan.take_profit_2 and plan.rr_tp2 else "  Take Profit 2: NULL")
    lines.append(f"  Take Profit 3: {plan.take_profit_3:.8g} (RR: {plan.rr_tp3:.2f})" if plan.take_profit_3 and plan.rr_tp3 else "  Take Profit 3: NULL")

    if not plan.levels_valid:
        lines.append(f"  ⚠️ VALIDATION FLAGS: {', '.join(plan.validation_flags)}")

    return "\n".join(lines)
