"""DCA Discovery pipeline.

Separate mode focused on finding symbols suitable for DCA futures bot operation.
Does not replace existing scan logic - this is a parallel discovery mode.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

from .binance import BinanceFuturesClient
from .config import load_config
from .regime import detect_btc_regime
from .indicators import atr, adx, ema
from .derivatives import fetch_derivatives_snapshot
from .dca_scoring import (
    assign_tier,
    compute_dca_score,
    suggest_dca_profile,
    compute_kill_switch_conditions,
    compute_execution_plan,
    DCAScoreResult,
)
from .utils import ensure_dir, json_dumps, utc_now_iso, write_text


console = Console()


def build_dca_universe(
    client: BinanceFuturesClient,
    dca_cfg: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Build universe of symbols for DCA evaluation.

    Returns list of dicts with symbol, volume, spread info.
    """
    tickers = client.ticker_24hr()
    books = client.book_ticker()
    book_by = {b["symbol"]: b for b in books}

    # USDT perpetual symbols only
    usdt = [t for t in tickers if t.get("symbol", "").endswith("USDT") and not t.get("symbol", "").endswith("BUSD")]

    # Volume sort
    usdt.sort(key=lambda x: float(x.get("quoteVolume", 0.0)), reverse=True)

    scan_top = int(dca_cfg.get("scan_top", 180))
    top = usdt[:scan_top]

    out: List[Dict[str, Any]] = []
    for t in top:
        sym = t["symbol"]
        vol = float(t.get("quoteVolume", 0.0))
        b = book_by.get(sym)
        if not b:
            continue
        bid = float(b.get("bidPrice", 0.0) or 0.0)
        ask = float(b.get("askPrice", 0.0) or 0.0)
        if bid <= 0 or ask <= 0:
            continue
        mid = (bid + ask) / 2.0
        spread_pct = (ask - bid) / mid * 100.0 if mid > 0 else 999.0
        out.append({
            "symbol": sym,
            "vol24h_usdt": vol,
            "spread_pct": spread_pct,
            "bid": bid,
            "ask": ask,
            "mid_price": mid,
        })

    return out


def evaluate_symbol_for_dca(
    client: BinanceFuturesClient,
    item: Dict[str, Any],
    side: str,
    btc_trend: str,
    btc_atrp: float,
    dca_cfg: Dict[str, Any],
    ind_cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Evaluate a single symbol for DCA suitability.

    Returns enriched dict with scores and profile, or None if ineligible.
    """
    sym = item["symbol"]
    vol24h = item["vol24h_usdt"]
    spread_pct = item["spread_pct"]

    tier_cfg = dca_cfg.get("tiers", {})
    tier, is_eligible = assign_tier(vol24h, spread_pct, tier_cfg)

    if not is_eligible:
        return None

    # Fetch klines for indicator computation
    tf = str(dca_cfg.get("timeframe", "4h"))
    klines = client.klines(sym, interval=tf, limit=200)
    if len(klines) < 60:
        return None

    high = np.array([float(x[2]) for x in klines], dtype=float)
    low = np.array([float(x[3]) for x in klines], dtype=float)
    close = np.array([float(x[4]) for x in klines], dtype=float)

    current_price = float(close[-1]) if len(close) > 0 else item["mid_price"]

    # Fetch derivatives data
    try:
        deriv_cfg = dca_cfg.get("derivatives_thresholds", {})
        deriv = fetch_derivatives_snapshot(client, sym, side, thresholds=deriv_cfg)
        funding_now = deriv.funding_now
        funding_median = deriv.funding_median
        oi_change_1h = deriv.oi_change_1h
    except Exception:
        funding_now = 0.0
        funding_median = 0.0
        oi_change_1h = 0.0

    # Compute DCA score
    weights_cfg = dca_cfg.get("weights", {})
    penalty_cfg = dca_cfg.get("penalties", {})
    min_dca_score = float(dca_cfg.get("min_dca_score", 55.0))

    score_result = compute_dca_score(
        symbol=sym,
        side=side,
        tier=tier,
        vol24h_usdt=vol24h,
        spread_pct=spread_pct,
        high=high,
        low=low,
        close=close,
        funding_now=funding_now,
        funding_median=funding_median,
        oi_change_1h=oi_change_1h,
        btc_trend=btc_trend,
        btc_atrp=btc_atrp,
        weights_cfg=weights_cfg,
        penalty_cfg=penalty_cfg,
        min_dca_score=min_dca_score,
    )

    # Compute ATR for profile suggestion
    atr_series = atr(high, low, close, int(ind_cfg.get("atr_period", 14)))
    atr_val = float(atr_series[-1]) if len(atr_series) > 0 else 0.0
    atr_pct = (atr_val / current_price) * 100.0 if current_price > 0 else 0.0

    # Compute EMA for entry reference
    ema50 = ema(close, int(ind_cfg.get("ema_fast", 50)))
    ema200 = ema(close, int(ind_cfg.get("ema_slow", 200)))
    ema50_val = float(ema50[-1]) if len(ema50) > 0 else current_price
    ema200_val = float(ema200[-1]) if len(ema200) > 0 else current_price

    # Compute deterministic execution plan (Entry, SL, TPs)
    exec_plan = compute_execution_plan(
        side=side,
        current_price=current_price,
        high=high,
        low=low,
        close=close,
        ema50_val=ema50_val,
        atr_val=atr_val,
        tier=tier,
    )

    # Entry reference zone string (for backward compatibility)
    if exec_plan.data_sufficient and exec_plan.entry_zone_lower and exec_plan.entry_zone_upper:
        entry_zone = f"{exec_plan.entry_zone_lower:.6g} - {exec_plan.entry_zone_upper:.6g}"
    else:
        # Fallback to old logic if execution plan failed
        if side == "LONG":
            entry_low = ema50_val - 0.5 * atr_val
            entry_high = ema50_val + 0.3 * atr_val
        else:
            entry_low = ema50_val - 0.3 * atr_val
            entry_high = ema50_val + 0.5 * atr_val
        entry_zone = f"{entry_low:.6g} - {entry_high:.6g}"

    # Suggest DCA profile
    profile = suggest_dca_profile(atr_pct, tier, score_result.dca_score)

    # Kill switch conditions
    kill_switches = compute_kill_switch_conditions(side, current_price, atr_val, tier)

    # Build result
    result = {
        "symbol": sym,
        "recommended_side": side,
        "tier": tier,
        "dca_score": score_result.dca_score,
        "score_components": score_result.score_components,
        "penalties": score_result.penalties,
        "eligibility_flags": score_result.eligibility_flags,
        "risk_flags": score_result.risk_flags,
        "current_price": current_price,
        "entry_reference_zone": entry_zone,
        # Execution plan with Entry/SL/TPs
        "execution_plan": exec_plan.to_dict(),
        "suggested_dca_profile": profile,
        "kill_switch_conditions": kill_switches,
        "status": score_result.status,
        # Additional data for output
        "vol24h_usdt": vol24h,
        "spread_pct": spread_pct,
        "atr_pct": atr_pct,
        "atr_value": atr_val,
        "ema50": ema50_val,
        "ema200": ema200_val,
        "funding_now": funding_now,
        "oi_change_1h": oi_change_1h,
    }

    return result


def evaluate_both_sides(
    client: BinanceFuturesClient,
    item: Dict[str, Any],
    btc_trend: str,
    btc_atrp: float,
    dca_cfg: Dict[str, Any],
    ind_cfg: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Evaluate symbol for both LONG and SHORT, return best side."""
    long_result = evaluate_symbol_for_dca(
        client, item, "LONG", btc_trend, btc_atrp, dca_cfg, ind_cfg
    )
    short_result = evaluate_symbol_for_dca(
        client, item, "SHORT", btc_trend, btc_atrp, dca_cfg, ind_cfg
    )

    if long_result is None and short_result is None:
        return None

    if long_result is None:
        return short_result
    if short_result is None:
        return long_result

    # Return side with higher score
    if long_result["dca_score"] >= short_result["dca_score"]:
        return long_result
    else:
        return short_result


def apply_explore_quota(
    candidates: List[Dict[str, Any]],
    watchlist_k: int,
    explore_quota: float,
) -> List[Dict[str, Any]]:
    """Apply explore quota cap to final selection.

    Ensures EXPLORE tier symbols don't exceed quota proportion.
    """
    # Separate by tier
    core_mid = [c for c in candidates if c["tier"] in ("CORE", "MID")]
    explore = [c for c in candidates if c["tier"] == "EXPLORE"]

    # Calculate max explore slots
    max_explore = max(1, int(watchlist_k * explore_quota))

    # Build final list respecting quota
    final: List[Dict[str, Any]] = []
    explore_count = 0

    for c in candidates:
        if len(final) >= watchlist_k:
            break

        if c["tier"] == "EXPLORE":
            if explore_count < max_explore:
                final.append(c)
                explore_count += 1
        else:
            final.append(c)

    return final


def run_dca_pipeline(
    cfg_or_path: str | Dict[str, Any],
    side_mode: str,
    print_prompt: bool = False,
) -> str | None:
    """Run DCA Discovery pipeline.

    Args:
        cfg_or_path: Config dict or path to config.json
        side_mode: "LONG", "SHORT", or "BOTH"
        print_prompt: Whether to print ChatGPT prompt to stdout

    Returns:
        Path to output directory or None on failure.
    """
    cfg = load_config(cfg_or_path) if isinstance(cfg_or_path, str) else cfg_or_path

    # Get DCA config with defaults
    dca_cfg = cfg.get("dca", {})
    ind_cfg = cfg.get("indicators", {})

    client = BinanceFuturesClient(
        base_url=str(cfg["binance"]["base_url"]),
        timeout_sec=int(cfg["binance"]["timeout_sec"]),
        insecure_ssl=bool(cfg["binance"].get("insecure_ssl", False)),
    )

    # Detect BTC regime
    console.print("Fetching BTC 1D to detect regime...")
    btc_kl = client.klines("BTCUSDT", interval="1d", limit=250)
    btc_open = np.array([float(x[1]) for x in btc_kl], dtype=float)
    btc_high = np.array([float(x[2]) for x in btc_kl], dtype=float)
    btc_low = np.array([float(x[3]) for x in btc_kl], dtype=float)
    btc_close = np.array([float(x[4]) for x in btc_kl], dtype=float)

    btc_regime = detect_btc_regime(
        btc_open, btc_high, btc_low, btc_close,
        ema_fast=int(ind_cfg.get("ema_fast", 50)),
        ema_slow=int(ind_cfg.get("ema_slow", 200)),
    )

    console.print(
        "BTC regime:",
        f"trend={btc_regime.btc_trend}",
        f"close={btc_regime.close:.2f}",
        f"atr%={btc_regime.atrp14:.2f}",
    )

    # Build universe
    console.print("Building DCA candidate universe...")
    universe = build_dca_universe(client, dca_cfg)
    console.print(f"Evaluating {len(universe)} symbols for DCA suitability...")

    # Evaluate each symbol
    candidates: List[Dict[str, Any]] = []
    for i, item in enumerate(universe):
        if (i + 1) % 20 == 0:
            console.print(f"  Progress: {i + 1}/{len(universe)}")

        if side_mode == "BOTH":
            result = evaluate_both_sides(
                client, item, btc_regime.btc_trend, btc_regime.atrp14,
                dca_cfg, ind_cfg,
            )
        else:
            result = evaluate_symbol_for_dca(
                client, item, side_mode, btc_regime.btc_trend, btc_regime.atrp14,
                dca_cfg, ind_cfg,
            )

        if result is not None:
            candidates.append(result)

    console.print(f"Found {len(candidates)} eligible candidates")

    # Sort by score
    candidates.sort(key=lambda x: x["dca_score"], reverse=True)

    # Apply watchlist limit and explore quota
    watchlist_k = int(dca_cfg.get("watchlist_k", 15))
    explore_quota = float(dca_cfg.get("explore_quota", 0.2))
    min_dca_score = float(dca_cfg.get("min_dca_score", 55.0))

    # Filter by minimum score
    qualified = [c for c in candidates if c["dca_score"] >= min_dca_score]

    # Apply explore quota
    final_list = apply_explore_quota(qualified, watchlist_k, explore_quota)

    console.print(f"Final watchlist: {len(final_list)} symbols")

    # Render table
    _render_dca_table(final_list, side_mode)

    # Build config snapshot for output
    config_snapshot = {
        "scan_top": dca_cfg.get("scan_top", 180),
        "watchlist_k": watchlist_k,
        "min_dca_score": min_dca_score,
        "explore_quota": explore_quota,
        "tiers": dca_cfg.get("tiers", {}),
        "weights": dca_cfg.get("weights", {}),
        "penalties": dca_cfg.get("penalties", {}),
    }

    # Build payload
    payload = {
        "analysis_timestamp_utc": utc_now_iso(),
        "mode": "DCA_DISCOVERY",
        "side_mode": side_mode,
        "market_regime": btc_regime.to_dict(),
        "config_snapshot": config_snapshot,
        "candidates": final_list,
    }

    # Output directory
    base_out = pathlib.Path(cfg.get("outputs", {}).get("base_dir", "outputs"))
    latest = base_out / "latest"
    ensure_dir(str(latest))

    # Write outputs
    (latest / "dca_payload.json").write_text(json_dumps(payload, pretty=True), encoding="utf-8")

    watchlist_txt = _build_dca_watchlist_text(final_list, side_mode, btc_regime)
    write_text(str(latest / "dca_watchlist.txt"), watchlist_txt)

    chatgpt_prompt = _build_dca_chatgpt_prompt(payload)
    write_text(str(latest / "dca_chatgpt_prompt.txt"), chatgpt_prompt)

    console.print(f"Saved DCA outputs to {latest.resolve()}")

    if print_prompt:
        console.print("\n" + "=" * 60)
        console.print("DCA ChatGPT Prompt:")
        console.print("=" * 60)
        console.print(chatgpt_prompt)

    return str(latest.resolve())


def _render_dca_table(items: List[Dict[str, Any]], side_mode: str) -> None:
    """Render DCA candidates table to console."""
    table = Table(title=f"DCA Discovery (side_mode={side_mode})")
    table.add_column("#", justify="right")
    table.add_column("Symbol")
    table.add_column("Score", justify="right")
    table.add_column("Tier")
    table.add_column("Side")
    table.add_column("Vol24h(M)", justify="right")
    table.add_column("Spread%", justify="right")
    table.add_column("ATR%", justify="right")
    table.add_column("Profile")
    table.add_column("Status")

    for idx, it in enumerate(items, 1):
        vol_m = it.get("vol24h_usdt", 0) / 1_000_000
        table.add_row(
            str(idx),
            str(it.get("symbol")),
            f"{it.get('dca_score', 0):.1f}",
            str(it.get("tier")),
            str(it.get("recommended_side")),
            f"{vol_m:.0f}",
            f"{it.get('spread_pct', 0):.3f}",
            f"{it.get('atr_pct', 0):.2f}",
            str(it.get("suggested_dca_profile", {}).get("profile", "")),
            str(it.get("status")),
        )

    console.print(table)


def _build_dca_watchlist_text(
    items: List[Dict[str, Any]],
    side_mode: str,
    btc_regime: Any,
) -> str:
    """Build human-readable DCA watchlist text."""
    lines = []
    lines.append("=" * 70)
    lines.append("DCA DISCOVERY WATCHLIST")
    lines.append(f"Mode: {side_mode}")
    lines.append(f"Generated: {utc_now_iso()}")
    lines.append(f"BTC Regime: {btc_regime.btc_trend} (ATR%: {btc_regime.atrp14:.2f})")
    lines.append("=" * 70)
    lines.append("")

    # Summary by tier
    core = [i for i in items if i["tier"] == "CORE"]
    mid = [i for i in items if i["tier"] == "MID"]
    explore = [i for i in items if i["tier"] == "EXPLORE"]
    lines.append(f"Summary: {len(core)} CORE | {len(mid)} MID | {len(explore)} EXPLORE")
    lines.append("")

    for idx, item in enumerate(items, 1):
        lines.append("-" * 70)
        lines.append(f"[{idx}] {item['symbol']} ({item['tier']}) - {item['recommended_side']}")
        lines.append(f"    DCA Score: {item['dca_score']:.1f} | Status: {item['status']}")
        lines.append(f"    Current Price: {item['current_price']:.6g}")
        lines.append(f"    Vol24h: ${item['vol24h_usdt']/1e6:.1f}M | Spread: {item['spread_pct']:.3f}%")
        lines.append(f"    ATR%: {item['atr_pct']:.2f} | Funding: {item['funding_now']:.6f}")
        lines.append("")

        # Execution Plan (Entry, SL, TPs)
        exec_plan = item.get("execution_plan", {})
        lines.append("    === EXECUTION LEVELS ===")
        if exec_plan.get("data_sufficient", False):
            entry_lo = exec_plan.get("entry_zone_lower")
            entry_hi = exec_plan.get("entry_zone_upper")
            sl = exec_plan.get("stop_loss")
            tp1 = exec_plan.get("take_profit_1")
            tp2 = exec_plan.get("take_profit_2")
            tp3 = exec_plan.get("take_profit_3")
            rr1 = exec_plan.get("rr_tp1")
            rr2 = exec_plan.get("rr_tp2")
            rr3 = exec_plan.get("rr_tp3")

            lines.append(f"    Entry Zone: {entry_lo:.8g} - {entry_hi:.8g}" if entry_lo and entry_hi else "    Entry Zone: NULL")
            lines.append(f"    Stop Loss:  {sl:.8g}" if sl else "    Stop Loss: NULL")
            lines.append(f"    TP1: {tp1:.8g} (RR: {rr1:.2f})" if tp1 and rr1 else "    TP1: NULL")
            lines.append(f"    TP2: {tp2:.8g} (RR: {rr2:.2f})" if tp2 and rr2 else "    TP2: NULL")
            lines.append(f"    TP3: {tp3:.8g} (RR: {rr3:.2f})" if tp3 and rr3 else "    TP3: NULL")

            if not exec_plan.get("levels_valid", True):
                lines.append(f"    ⚠️ VALIDATION: {', '.join(exec_plan.get('validation_flags', []))}")
        else:
            lines.append("    ⚠️ INSUFFICIENT DATA - Cannot compute execution levels")
        lines.append("")

        # Score components
        comp = item.get("score_components", {})
        lines.append(f"    Components: micro={comp.get('microstructure', 0):.1f} | "
                    f"reversion={comp.get('mean_reversion', 0):.1f} | "
                    f"volatility={comp.get('volatility_fit', 0):.1f} | "
                    f"derivatives={comp.get('derivatives_health', 0):.1f} | "
                    f"context={comp.get('context', 0):.1f}")

        # Penalties
        pen = item.get("penalties", {})
        if pen.get("items"):
            pen_str = ", ".join([f"{p['name']}={p['value']}" for p in pen["items"]])
            lines.append(f"    Penalties: {pen_str} (total: {pen.get('total', 0):.1f})")

        # Risk flags
        if item.get("risk_flags"):
            lines.append(f"    Risk Flags: {', '.join(item['risk_flags'])}")

        # Profile
        profile = item.get("suggested_dca_profile", {})
        lines.append(f"    Profile: {profile.get('profile')} | "
                    f"Grid: {profile.get('grid_step_pct_hint', 0):.2f}% | "
                    f"MaxLayers: {profile.get('max_layers_hint', 0)} | "
                    f"SizeMult: {profile.get('size_multiplier_hint', 1):.1f}x")

        # Kill switches
        lines.append("    Kill Switches:")
        for ks in item.get("kill_switch_conditions", []):
            lines.append(f"      - {ks}")
        lines.append("")

    lines.append("=" * 70)
    lines.append("DISCLAIMER: This is for research only. Not financial advice.")
    lines.append("=" * 70)

    return "\n".join(lines)


def _build_dca_chatgpt_prompt(payload: Dict[str, Any]) -> str:
    """Build ChatGPT audit prompt for DCA analysis."""
    lines = []

    lines.append("=" * 70)
    lines.append("DCA DISCOVERY AUDIT PROMPT")
    lines.append("=" * 70)
    lines.append("")
    lines.append("You are a senior crypto futures trader reviewing DCA bot candidates.")
    lines.append("IMPORTANT: All metrics below are deterministic. AI overlay is advisory only.")
    lines.append("")

    # Regime summary
    regime = payload.get("market_regime", {})
    lines.append("## MARKET REGIME")
    lines.append(f"- BTC Trend: {regime.get('btc_trend', 'N/A')}")
    lines.append(f"- BTC Close: {regime.get('close', 0):.2f}")
    lines.append(f"- BTC ATR%: {regime.get('atrp14', 0):.2f}")
    lines.append(f"- BTC Volatility: {regime.get('volatility_state', 'N/A')}")
    lines.append("")

    # Config
    config = payload.get("config_snapshot", {})
    lines.append("## CONFIG")
    lines.append(f"- Min DCA Score: {config.get('min_dca_score', 55)}")
    lines.append(f"- Explore Quota: {config.get('explore_quota', 0.2)*100:.0f}%")
    lines.append("")

    # Candidates summary table
    candidates = payload.get("candidates", [])
    lines.append("## TOP CANDIDATES SUMMARY")
    lines.append("")
    lines.append("| # | Symbol | Tier | Side | Score | Status | Profile |")
    lines.append("|---|--------|------|------|-------|--------|---------|")

    for idx, c in enumerate(candidates[:15], 1):
        profile = c.get("suggested_dca_profile", {}).get("profile", "")
        lines.append(
            f"| {idx} | {c['symbol']} | {c['tier']} | {c['recommended_side']} | "
            f"{c['dca_score']:.1f} | {c['status']} | {profile} |"
        )
    lines.append("")

    # Detailed breakdown with EXECUTION LEVELS for all candidates
    lines.append("## DETAILED BREAKDOWN WITH EXECUTION LEVELS")
    lines.append("")

    for idx, c in enumerate(candidates, 1):
        lines.append(f"### [{idx}] {c['symbol']} ({c['tier']})")
        lines.append(f"- Side: {c['recommended_side']}")
        lines.append(f"- DCA Score: {c['dca_score']:.1f}")
        lines.append(f"- Status: {c['status']}")
        lines.append(f"- Current Price: {c['current_price']:.6g}")
        lines.append("")

        # Execution levels (Entry, SL, TPs)
        exec_plan = c.get("execution_plan", {})
        lines.append("**EXECUTION LEVELS:**")
        if exec_plan.get("data_sufficient", False):
            entry_lo = exec_plan.get("entry_zone_lower")
            entry_hi = exec_plan.get("entry_zone_upper")
            sl = exec_plan.get("stop_loss")
            tp1 = exec_plan.get("take_profit_1")
            tp2 = exec_plan.get("take_profit_2")
            tp3 = exec_plan.get("take_profit_3")
            rr1 = exec_plan.get("rr_tp1")
            rr2 = exec_plan.get("rr_tp2")
            rr3 = exec_plan.get("rr_tp3")

            if entry_lo and entry_hi:
                lines.append(f"- Entry Zone: {entry_lo:.8g} - {entry_hi:.8g}")
            else:
                lines.append("- Entry Zone: NULL")

            lines.append(f"- Stop Loss: {sl:.8g}" if sl else "- Stop Loss: NULL")
            lines.append(f"- Take Profit 1: {tp1:.8g} (RR: {rr1:.2f})" if tp1 and rr1 else "- Take Profit 1: NULL")
            lines.append(f"- Take Profit 2: {tp2:.8g} (RR: {rr2:.2f})" if tp2 and rr2 else "- Take Profit 2: NULL")
            lines.append(f"- Take Profit 3: {tp3:.8g} (RR: {rr3:.2f})" if tp3 and rr3 else "- Take Profit 3: NULL")

            if not exec_plan.get("levels_valid", True):
                lines.append(f"- ⚠️ VALIDATION FLAGS: {', '.join(exec_plan.get('validation_flags', []))}")
        else:
            lines.append("- ⚠️ INSUFFICIENT DATA - Cannot compute execution levels")
        lines.append("")

        # Score components (abbreviated for non-top-5)
        if idx <= 5:
            comp = c.get("score_components", {})
            lines.append("Score Components:")
            lines.append(f"  - Microstructure: {comp.get('microstructure', 0):.1f}")
            lines.append(f"  - Mean Reversion: {comp.get('mean_reversion', 0):.1f}")
            lines.append(f"  - Volatility Fit: {comp.get('volatility_fit', 0):.1f}")
            lines.append(f"  - Derivatives Health: {comp.get('derivatives_health', 0):.1f}")
            lines.append(f"  - Context: {comp.get('context', 0):.1f}")
            lines.append("")

            if c.get("risk_flags"):
                lines.append(f"Risk Flags: {', '.join(c['risk_flags'])}")
                lines.append("")

            profile = c.get("suggested_dca_profile", {})
            lines.append(f"Suggested Profile: {profile.get('profile')}")
            lines.append(f"  - Grid Step: {profile.get('grid_step_pct_hint', 0):.2f}%")
            lines.append(f"  - Max Layers: {profile.get('max_layers_hint', 0)}")
            lines.append(f"  - Size Multiplier: {profile.get('size_multiplier_hint', 1):.1f}x")
            lines.append("")

            lines.append("Kill Switch Conditions:")
            for ks in c.get("kill_switch_conditions", []):
                lines.append(f"  - {ks}")
            lines.append("")

    # Footer with explicit instruction
    lines.append("=" * 70)
    lines.append("## MANDATORY OUTPUT INSTRUCTION")
    lines.append("")
    lines.append("ChatGPT must list out Entry, SL, and TP for all selected coins.")
    lines.append("")
    lines.append("For each coin you recommend, provide:")
    lines.append("1. Symbol and Side (LONG/SHORT)")
    lines.append("2. Entry Zone (lower - upper)")
    lines.append("3. Stop Loss price")
    lines.append("4. Take Profit 1 with RR ratio")
    lines.append("5. Take Profit 2 with RR ratio")
    lines.append("6. Take Profit 3 with RR ratio (or NULL if not applicable)")
    lines.append("")
    lines.append("=" * 70)
    lines.append("STRICT NOTE: Deterministic metrics are source of truth.")
    lines.append("AI overlay is advisory only - do not override algo scores.")
    lines.append("All Entry/SL/TP levels are rule-based and deterministic.")
    lines.append("=" * 70)

    return "\n".join(lines)
