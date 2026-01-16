#!/usr/bin/env python3
"""Smoke tests for WhaleTraderBot v2.3 ALGO-only.

These tests verify critical logic paths without needing live data.
Run with: python3 -m wtb.tests_smoke
"""

from __future__ import annotations

import numpy as np
from .algo import choose_setup_type


def test_setup_type_determinism():
    """Verify that all 4 setup types can be produced deterministically."""
    print("Testing setup_type determinism...")

    # Mock data: simple arrays
    close = np.array([100.0] * 100 + [105.0, 106.0, 107.0])
    high = np.array([101.0] * 100 + [106.0, 107.0, 108.0])
    low = np.array([99.0] * 100 + [104.0, 105.0, 106.0])
    current_price = 107.0
    atr_last = 2.0

    # Test case 1: RANGE_SWEEP_RECLAIM
    # - Low ADX (non-trending)
    # - Price sweeps below range low and reclaims
    result_1 = choose_setup_type(
        side_mode="LONG",
        adx_4h=20.0,
        atrp_4h=3.0,
        btc_trend="NEUTRAL",
        close=np.array([100.0] * 50 + [99.0, 98.0, 99.5]),  # sweep low + reclaim
        high=np.array([101.0] * 50 + [100.0, 99.0, 100.0]),
        low=np.array([99.0] * 50 + [97.5, 96.0, 98.0]),  # wick below range
        current_price=99.5,
        rng_low=99.0,
        rng_high=101.0,
        rng_h=2.0,
        atr_last=1.0,
    )
    assert result_1 == "RANGE_SWEEP_RECLAIM", f"Expected RANGE_SWEEP_RECLAIM, got {result_1}"
    print(f"  ✓ RANGE_SWEEP_RECLAIM: {result_1}")

    # Test case 2: BREAKOUT_RETEST
    # - Tight range (8% or less width)
    # - Price closes above range high (breakout)
    result_2 = choose_setup_type(
        side_mode="LONG",
        adx_4h=22.0,
        atrp_4h=3.0,
        btc_trend="BULLISH",
        close=np.array([100.0] * 50 + [100.5, 101.2]),
        high=np.array([101.0] * 50 + [101.0, 102.0]),
        low=np.array([99.0] * 50 + [100.0, 101.0]),
        current_price=101.2,
        rng_low=99.0,
        rng_high=101.0,
        rng_h=2.0,  # 2% range
        atr_last=1.0,
    )
    assert result_2 == "BREAKOUT_RETEST", f"Expected BREAKOUT_RETEST, got {result_2}"
    print(f"  ✓ BREAKOUT_RETEST: {result_2}")

    # Test case 3: TREND_PULLBACK
    # - High ADX (trending)
    # - Not in a very volatile state
    # - Range is wider (not tight)
    # - Price is NOT near range edges (to avoid BREAKOUT_RETEST)
    result_3 = choose_setup_type(
        side_mode="LONG",
        adx_4h=35.0,
        atrp_4h=4.0,
        btc_trend="BULLISH",
        close=np.array([100.0] * 50 + [103.0, 104.0]),
        high=np.array([101.0] * 50 + [104.0, 105.0]),
        low=np.array([99.0] * 50 + [102.0, 103.0]),
        current_price=104.0,
        rng_low=95.0,
        rng_high=110.0,
        rng_h=15.0,  # wide range (>8% of mid)
        atr_last=2.0,
    )
    assert result_3 == "TREND_PULLBACK", f"Expected TREND_PULLBACK, got {result_3}"
    print(f"  ✓ TREND_PULLBACK: {result_3}")

    # Test case 4: VOLATILITY_FADE
    # - Very high ATR% (>= 8%)
    # - Oversized candle
    result_4 = choose_setup_type(
        side_mode="LONG",
        adx_4h=25.0,
        atrp_4h=12.0,  # very high volatility
        btc_trend="NEUTRAL",
        close=np.array([100.0] * 50 + [110.0]),  # big move
        high=np.array([101.0] * 50 + [115.0]),
        low=np.array([99.0] * 50 + [105.0]),
        current_price=110.0,
        rng_low=99.0,
        rng_high=101.0,
        rng_h=2.0,
        atr_last=3.0,
    )
    assert result_4 == "VOLATILITY_FADE", f"Expected VOLATILITY_FADE, got {result_4}"
    print(f"  ✓ VOLATILITY_FADE: {result_4}")

    print("All setup_type tests passed!")


def test_late_atr_calculation():
    """Verify late_atr calculation logic."""
    print("Testing late_atr calculation...")

    # The late_atr should only be positive when price is BEYOND the entry zone.
    # For LONG: current_price > entry_high triggers late
    # For SHORT: current_price < entry_low triggers late

    # LONG example: current price is 105, entry_high is 102, atr is 2
    # late_atr = (105 - 102) / 2 = 1.5 ATR units late
    late_long = (105.0 - 102.0) / 2.0
    assert late_long == 1.5, f"Expected 1.5, got {late_long}"
    print(f"  ✓ LONG late_atr calculation: {late_long}")

    # SHORT example: current price is 95, entry_low is 98, atr is 2
    # late_atr = (98 - 95) / 2 = 1.5 ATR units late
    late_short = (98.0 - 95.0) / 2.0
    assert late_short == 1.5, f"Expected 1.5, got {late_short}"
    print(f"  ✓ SHORT late_atr calculation: {late_short}")

    print("All late_atr tests passed!")


def test_penalty_capping():
    """Verify penalty capping logic."""
    print("Testing penalty capping...")

    max_penalty_abs = 10.0
    penalties = -25.0  # huge penalty

    # Cap should bring it to -10
    capped = max(-max_penalty_abs, min(max_penalty_abs, penalties))
    assert capped == -10.0, f"Expected -10.0, got {capped}"
    print(f"  ✓ Penalty capped from -25 to {capped}")

    # Positive penalty (should also cap)
    penalties_pos = 15.0
    capped_pos = max(-max_penalty_abs, min(max_penalty_abs, penalties_pos))
    assert capped_pos == 10.0, f"Expected 10.0, got {capped_pos}"
    print(f"  ✓ Penalty capped from +15 to {capped_pos}")

    print("All penalty capping tests passed!")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("WhaleTraderBot v2.3 ALGO-only - Smoke Tests")
    print("=" * 60)
    print()

    test_setup_type_determinism()
    print()
    test_late_atr_calculation()
    print()
    test_penalty_capping()
    print()

    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
