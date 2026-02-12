from __future__ import annotations

import json
import pathlib
from typing import Any, Dict


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)  # type: ignore
        else:
            out[k] = v
    return out


def default_config() -> Dict[str, Any]:
    # Keep defaults aligned with config.example.json
    return {
        "binance": {
            "base_url": "https://fapi.binance.com",
            "timeout_sec": 15,
            "insecure_ssl": False,
        },
        "scan": {
            "scan_top": 120,
            "min_volume_usdt": 50_000_000,
            "max_spread_pct": 0.2,
            "shortlist_n": 40,
            "watchlist_k": 10,
            "min_watch_score": 60,
            "side_default": "LONG",
            "timeframes": {"context": "1d", "execution": "4h", "refine": "1h"},
        },
        "indicators": {
            "ema_fast": 50,
            "ema_slow": 200,
            "atr_period": 14,
            "adx_period": 14,
            "rsi_period": 14,
            "bb_period": 20,
            "bb_std": 2.0,
            "pivot_w": 2,
            "range_lookback_bars": 48,
        },
        "breath": {
            "enabled": True,
            "top_n": 20,
            "thresholds": {
                "oi_drop_pct": -0.04,
                "median_corr": 0.85,
                "funding_abs": 0.0005,
                "btc_vol_share_jump": 0.08,
            },
            "min_watch_score_risk_off": 75,
        },
        "whales": {
            "enabled": False,
            "hyperliquid": {
                "base_url": "https://api.hyperliquid.xyz",
                "timeout_sec": 15,
            },
            "assets": ["BTC", "ETH"],
            "addresses": [],
            "cache_sec": 60,
            "thresholds": {
                "net_notional_strong": 25_000_000,
                "net_notional_extreme": 50_000_000,
            },
        },
        "derivatives": {
            "enabled": True,
            "funding_history_limit": 24,
            "oi_hist_period": "1h",
            "oi_hist_limit": 6,
            "thresholds": {
                "funding_high_abs": 0.0005,
                "funding_extreme_abs": 0.001,
                "crowded_abs": 0.0005,
                "oi_spike_pct": 0.05,
                "oi_flush_pct": -0.05,
            },
        },
        "orderflow": {
            "enabled": True,
            "top_n": 15,
            "window_min_15": 15,
            "window_min_60": 60,
            "aggtrades_limit": 1000,
        },
        "outputs": {
            "base_dir": "outputs",
            "keep_latest": 5,
            "compact_prompt": True,
        },
        "backtest": {
            "enabled": False,
            "cache_dir": "data_cache",
        },
        "scoring": {
            "weights": {
                "tradeability": 35,
                "setup_quality": 25,
                "derivatives": 20,
                "orderflow": 10,
                "context": 5,
                "whales": 5,
            },
            "late": {
                # Measured in ATR multiples away from the entry-zone boundary.
                # 0.5 = still acceptable; 1.5+ = too stretched (watch pullback).
                "ok_atr": 0.5,
                "watch_atr": 1.5,
                # Penalties are intentionally SMALL. Watch statuses are meant to guide execution,
                # not to nuke ranking.
                "penalty_watch_late": -2.0,
                "penalty_watch_pullback": 0.0,
            },
            # Legacy fields (kept for backward compatibility). Context effects are handled via
            # the explicit "context" component score rather than hard penalties.
            "regime_penalty": 0.0,
            "breath_penalty": 0.0,
        },
        "manual": {
            "enabled": True,
            "default_symbols": ["BTCUSDT", "ETHUSDT"],
        },
        "dca": {
            "side_default": "LONG",
            "scan_top": 180,
            "watchlist_k": 15,
            "min_dca_score": 55,
            "explore_quota": 0.2,
            "timeframe": "4h",
            "tiers": {
                "core": {"min_volume_usdt": 500_000_000, "max_spread_pct": 0.08},
                "mid": {"min_volume_usdt": 100_000_000, "max_spread_pct": 0.15},
                "explore": {"min_volume_usdt": 25_000_000, "max_spread_pct": 0.30},
            },
            "weights": {
                "microstructure": 30,
                "mean_reversion": 25,
                "volatility_fit": 20,
                "derivatives_health": 15,
                "context": 10,
            },
            "penalties": {
                "btc_headwind": 6,
                "extreme_funding": 4,
                "liquidity_stress": 4,
                "trend_runaway": 5,
                "max_total": 15,
            },
        },
    }


def load_config(path: str | None) -> Dict[str, Any]:
    cfg = default_config()
    if path is None:
        return cfg
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    override = json.loads(p.read_text(encoding="utf-8"))
    return deep_merge(cfg, override)
