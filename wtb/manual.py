"""Manual mode: analyze user-specified symbols using the same pipeline as scan mode.

This module provides the run_manual_pipeline() function which:
1. Validates and normalizes user-provided symbols
2. Runs the SAME analysis pipeline as scan mode (regime, breath, derivatives, orderflow, scoring)
3. Generates identical outputs (watchlist.txt, payload.json, chatgpt_prompt.txt)

The key difference from scan mode: symbols come from user input instead of volume-based scanning.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console

from .binance import (
    BinanceFuturesClient,
    parse_symbols_file,
    parse_symbols_input,
    validate_symbols,
)
from .config import load_config
from .regime import detect_btc_regime
from .breath import compute_market_breath
from .indicators import atr, adx
from .algo import build_algo_plan
from .derivatives import fetch_derivatives_snapshot as get_derivatives_snapshot
from .orderflow import compute_orderflow as get_orderflow_snapshot
from .whales import build_whale_context, whales_component_score
from .plutus import run_plutus_batch
from .prompts import build_chatgpt_teamlead_prompt
from .utils import ensure_dir, json_dumps, utc_now_iso, write_text
from .pipeline import prescore_symbol, _default_confirm, _score_breakdown, _render_table

import pathlib

console = Console()


def run_manual_pipeline(
    cfg_or_path: str | Dict[str, Any],
    side_mode: str,
    symbols: Optional[List[str]] = None,
    symbols_file: Optional[str] = None,
    print_prompt: bool = False,
) -> str | None:
    """Run manual mode pipeline on user-specified symbols.

    This function reuses ALL the same analysis logic as scan mode:
    - BTC regime detection
    - Market breath gauge
    - Whale context (if enabled)
    - Prescore calculation
    - Algo plan building (setup detection, entry/SL/TP)
    - Derivatives overlay
    - Orderflow overlay
    - Final scoring with penalties
    - Watchlist selection
    - Output generation

    Args:
        cfg_or_path: Config dict or path to config.json
        side_mode: "LONG" or "SHORT"
        symbols: List of symbols from --symbols (e.g., ["ETH", "BTC", "PEPE"])
        symbols_file: Path to symbols file from --symbols-file
        print_prompt: Whether to print ChatGPT prompt to stdout

    Returns:
        Output directory path on success, None on failure
    """
    cfg = load_config(cfg_or_path) if isinstance(cfg_or_path, str) else cfg_or_path

    # Initialize client
    client = BinanceFuturesClient(
        base_url=str(cfg["binance"]["base_url"]),
        timeout_sec=int(cfg["binance"]["timeout_sec"]),
        insecure_ssl=bool(cfg["binance"]["insecure_ssl"]),
    )

    # Parse and validate symbols
    raw_symbols: List[str] = []

    if symbols:
        raw_symbols.extend(symbols)
    if symbols_file:
        try:
            raw_symbols.extend(parse_symbols_file(symbols_file))
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            return None

    # Also check config for default symbols if none provided
    if not raw_symbols:
        manual_cfg = cfg.get("manual", {})
        default_symbols = manual_cfg.get("default_symbols", [])
        if default_symbols:
            raw_symbols.extend(default_symbols)
            console.print(f"Using default symbols from config: {default_symbols}")

    if not raw_symbols:
        console.print("[red]Error:[/red] No symbols provided. Use --symbols or --symbols-file")
        return None

    console.print(f"Validating {len(raw_symbols)} symbol(s)...")
    valid_symbols, warnings, errors = validate_symbols(client, raw_symbols)

    # Print warnings
    for original, warning in warnings:
        console.print(f"[yellow]Warning:[/yellow] {warning}")

    # Print errors
    for err in errors:
        console.print(f"[red]Error:[/red] Symbol not found on Binance USDT-M perpetual: {err}")

    if not valid_symbols:
        console.print("[red]Error:[/red] No valid symbols to analyze after validation.")
        return None

    console.print(f"Analyzing {len(valid_symbols)} valid symbol(s): {valid_symbols}")

    # ========== BTC REGIME ==========
    console.print("Fetching BTC 1D to detect regime (BTC/USDT)...")
    btc_kl = client.klines("BTCUSDT", interval="1d", limit=250)
    btc_open = np.array([float(x[1]) for x in btc_kl], dtype=float)
    btc_high = np.array([float(x[2]) for x in btc_kl], dtype=float)
    btc_low = np.array([float(x[3]) for x in btc_kl], dtype=float)
    btc_close = np.array([float(x[4]) for x in btc_kl], dtype=float)
    btc_regime = detect_btc_regime(
        btc_open,
        btc_high,
        btc_low,
        btc_close,
        ema_fast=int(cfg["indicators"]["ema_fast"]),
        ema_slow=int(cfg["indicators"]["ema_slow"]),
    )
    console.print(
        "BTC regime:",
        f"trend={btc_regime.btc_trend}",
        f"close={btc_regime.close:.2f}",
        f"ema50={btc_regime.ema50:.2f}",
        f"ema200={btc_regime.ema200:.2f}",
        f"adx14={btc_regime.adx14:.2f}",
        f"atr%14={btc_regime.atrp14:.2f}",
        f"vol={btc_regime.volatility_state}",
    )

    # ========== FETCH TICKER DATA FOR VALIDATED SYMBOLS ==========
    console.print("Fetching ticker data for validated symbols...")
    tickers = client.ticker_24hr()
    books = client.book_ticker()

    ticker_by_sym = {t["symbol"]: t for t in tickers}
    book_by_sym = {b["symbol"]: b for b in books}

    # Build universe-like structure for validated symbols
    universe: List[Dict[str, Any]] = []
    for sym in valid_symbols:
        t = ticker_by_sym.get(sym)
        b = book_by_sym.get(sym)
        if not t or not b:
            console.print(f"[yellow]Warning:[/yellow] Could not fetch data for {sym}, skipping.")
            continue

        vol = float(t.get("quoteVolume", 0.0))
        bid = float(b.get("bidPrice", 0.0) or 0.0)
        ask = float(b.get("askPrice", 0.0) or 0.0)

        if bid <= 0 or ask <= 0:
            console.print(f"[yellow]Warning:[/yellow] Invalid bid/ask for {sym}, skipping.")
            continue

        spread_pct = (ask - bid) / ((ask + bid) / 2.0) * 100.0
        universe.append({
            "symbol": sym,
            "vol24h_usdt": vol,
            "spread_pct": spread_pct,
            "bid": bid,
            "ask": ask,
        })

    if not universe:
        console.print("[red]Error:[/red] No symbols with valid ticker data.")
        return None

    # ========== MARKET BREATH ==========
    breath = None
    if bool(cfg.get("breath", {}).get("enabled", True)):
        console.print("Computing Market Breath Gauge...")
        thresholds = cfg.get("breath", {}).get("thresholds", {})
        btc_symbol = str(cfg.get("breath", {}).get("btc_symbol", "BTCUSDT"))
        timeframe = str(cfg.get("breath", {}).get("timeframe", cfg["scan"]["timeframes"]["execution"]))
        breath = compute_market_breath(
            client,
            [u["symbol"] for u in universe],
            btc_symbol=btc_symbol,
            timeframe=timeframe,
            thresholds=thresholds,
        )

    # ========== WHALES CONTEXT ==========
    whale_ctx = None
    if bool(cfg.get("whales", {}).get("enabled", False)):
        console.print("Fetching Hyperliquid whale context (BTC/ETH only)...")
        try:
            whale_ctx = build_whale_context(cfg.get("whales", {}), now_utc=utc_now_iso())
            console.print(f"Whale context: state={whale_ctx.state}, bullish_score={whale_ctx.bullish_score:.1f}, flags={whale_ctx.flags}")
        except Exception as e:
            console.print(f"Warning: Whale context failed: {e}")
            whale_ctx = None

    # ========== COMPUTE 4H FEATURES (PRESCORE) ==========
    tf = str(cfg["scan"]["timeframes"]["execution"])
    ind = cfg["indicators"]

    scored: List[Dict[str, Any]] = []
    for item in universe:
        sym = item["symbol"]
        console.print(f"  Fetching {sym} {tf} klines...")
        kl = client.klines(sym, interval=tf, limit=200)
        if len(kl) < 60:
            console.print(f"[yellow]Warning:[/yellow] Not enough klines for {sym} (got {len(kl)}), skipping.")
            continue

        high = np.array([float(x[2]) for x in kl], dtype=float)
        low = np.array([float(x[3]) for x in kl], dtype=float)
        close = np.array([float(x[4]) for x in kl], dtype=float)

        atr4_series = atr(high, low, close, int(ind["atr_period"]))
        adx4_series = adx(high, low, close, int(ind["adx_period"]))

        if len(atr4_series) == 0 or len(adx4_series) == 0:
            console.print(f"[yellow]Warning:[/yellow] Could not compute indicators for {sym}, skipping.")
            continue

        atr4 = float(atr4_series[-1])
        adx4 = float(adx4_series[-1])
        atrp4 = atr4 / close[-1] * 100.0

        pscore = prescore_symbol(item["vol24h_usdt"], item["spread_pct"], float(atrp4), adx4)
        scored.append({
            **item,
            "atrp4h": float(atrp4),
            "adx4h": adx4,
            "prescore": float(pscore),
            "klines_4h": kl,
        })

    if not scored:
        console.print("[red]Error:[/red] No symbols passed indicator computation.")
        return None

    # Sort by prescore (for consistent ordering, though we process all)
    scored.sort(key=lambda x: x["prescore"], reverse=True)

    # In manual mode, we don't apply shortlist_n limit - we analyze all provided symbols
    shortlist = scored

    console.print(f"Building algo plans for {len(shortlist)} symbol(s)...")

    # ========== BUILD ALGO PLANS ==========
    late_cfg = cfg.get("scoring", {}).get("late", {})
    late_ok_atr = float(late_cfg.get("ok_atr", 0.15))
    late_watch_atr = float(late_cfg.get("watch_atr", 0.25))
    range_lookback = int(ind.get("range_lookback_bars", ind.get("range_lookback", 48)))

    plans: List[Dict[str, Any]] = []
    for s in shortlist:
        kl = s["klines_4h"]
        open_ = np.array([float(x[1]) for x in kl], dtype=float)
        high = np.array([float(x[2]) for x in kl], dtype=float)
        low = np.array([float(x[3]) for x in kl], dtype=float)
        close = np.array([float(x[4]) for x in kl], dtype=float)
        current_price = float(close[-1]) if len(close) else float(s["ask"])

        plan = build_algo_plan(
            symbol=s["symbol"],
            side_mode=side_mode,
            close=close,
            high=high,
            low=low,
            current_price=current_price,
            bid=float(s["bid"]),
            ask=float(s["ask"]),
            volume_usdt=float(s["vol24h_usdt"]),
            btc_trend=btc_regime.btc_trend,
            range_lookback=range_lookback,
            atr_period=int(cfg["indicators"]["atr_period"]),
            adx_period=int(cfg["indicators"]["adx_period"]),
            late_ok_atr=late_ok_atr,
            late_watch_atr=late_watch_atr,
        )

        plan_dict = plan.to_dict()
        plan_dict["late_status"] = plan_dict.get("status_hint")
        plan_dict["vol24h_usdt"] = float(s.get("vol24h_usdt", 0.0))
        plan_dict["spread_pct"] = float(s.get("spread_pct", 0.0))
        plan_dict["atrp4h"] = float(s.get("atrp4h", 0.0))
        plan_dict["adx4h"] = float(s.get("adx4h", 0.0))
        plans.append(plan_dict)

    # ========== DERIVATIVES OVERLAY ==========
    if bool(cfg.get("derivatives", {}).get("enabled", True)):
        console.print("Fetching derivatives data...")
        funding_limit = int(cfg.get("derivatives", {}).get("funding_history_limit", 24))
        for p in plans:
            try:
                thresholds = cfg.get("derivatives", {}).get("thresholds", {})
                snap = get_derivatives_snapshot(
                    client,
                    p["symbol"],
                    side_mode,
                    funding_history_limit=funding_limit,
                    thresholds=thresholds,
                )
                p["derivatives"] = snap.to_dict()
            except Exception as e:
                p["derivatives"] = {"error": str(e)}

    # ========== ORDERFLOW OVERLAY ==========
    if bool(cfg.get("orderflow", {}).get("enabled", True)):
        console.print("Fetching orderflow data...")
        now_ms = int(time.time() * 1000)
        # In manual mode, apply orderflow to all symbols (not just top_n)
        w15 = cfg.get("orderflow", {}).get("window_15m_min", cfg.get("orderflow", {}).get("window_min_15", 15))
        w1h = cfg.get("orderflow", {}).get("window_1h_min", cfg.get("orderflow", {}).get("window_min_60", 60))
        limit = int(cfg.get("orderflow", {}).get("aggtrades_limit", 1000))

        for p in plans:
            try:
                snap = get_orderflow_snapshot(
                    client,
                    p["symbol"],
                    now_ms=now_ms,
                    insecure_ssl=bool(cfg.get("binance", {}).get("insecure_ssl", False)),
                    window_15m_min=int(w15),
                    window_1h_min=int(w1h),
                    limit=limit,
                )
                p["orderflow"] = snap.to_dict()
            except Exception as e:
                p["orderflow"] = {"error": str(e)}

    # ========== WHALES SCORING OVERLAY ==========
    for p in plans:
        if whale_ctx:
            try:
                score_for_side = whales_component_score(whale_ctx, side_mode=side_mode, symbol=p["symbol"])
                flags = []
                if side_mode == "LONG" and whale_ctx.state == "BEARISH":
                    flags.append("WHALES_OPPOSING")
                elif side_mode == "SHORT" and whale_ctx.state == "BULLISH":
                    flags.append("WHALES_OPPOSING")
                p["whales"] = {
                    "score_for_side": float(score_for_side),
                    "state": whale_ctx.state,
                    "bullish_score": float(whale_ctx.bullish_score),
                    "flags": flags,
                }
            except Exception as e:
                p["whales"] = {"error": str(e), "score_for_side": 50.0, "flags": []}
        else:
            p["whales"] = {"score_for_side": 50.0, "state": "NEUTRAL", "flags": []}

    # ========== COMPUTE FINAL SCORES ==========
    console.print("Computing final scores...")
    for p in plans:
        p["psych_notes"] = {"psych_score": None, "biases": [], "manipulation_flags": [], "comment": "ALGO_ONLY"}
        p["confirm"] = _default_confirm(p)
        p["overlay_flags"] = []
        p["overlay_score"] = 0.0
        final_score, score_detail = _score_breakdown(p, btc_regime, breath, cfg)
        p["final_score"] = float(final_score)
        p["score_detail"] = score_detail

    # Sort plans by score
    plans.sort(key=lambda x: x["final_score"], reverse=True)

    # ========== OPTIONAL PLUTUS (OLLAMA) OVERLAY ==========
    plutus_enabled = bool(cfg.get("plutus", {}).get("enabled", False))
    if plutus_enabled:
        max_overlay_candidates = int(cfg.get("plutus", {}).get("max_candidates", 10))
        top_plans = plans[:max_overlay_candidates]

        console.print(f"Running Ollama psychology overlay on top {len(top_plans)} candidates...")
        try:
            whale_ctx_dict = {}
            if whale_ctx:
                whale_ctx_dict = {
                    "state": whale_ctx.state,
                    "bullish_score": float(whale_ctx.bullish_score),
                    "flags": whale_ctx.flags,
                }

            market_ctx = {
                "btc_regime": btc_regime.to_dict() if btc_regime else {},
                "breath": breath.to_dict() if breath else {},
                "whale_ctx": whale_ctx_dict,
            }
            overlay_result, overlay_meta = run_plutus_batch(
                cfg.get("plutus", {}),
                side_mode,
                market_ctx,
                candidates=top_plans,
                insecure_ssl=bool(cfg.get("binance", {}).get("insecure_ssl", False)),
            )

            if overlay_meta.ok:
                console.print(f"Ollama overlay succeeded: {len(overlay_result.get('items', []))} items")
                overlay_by_symbol = {item["symbol"]: item for item in overlay_result.get("items", [])}
                for p in top_plans:
                    overlay = overlay_by_symbol.get(p["symbol"])
                    if overlay:
                        p["psych_notes"] = {
                            "psych_score": overlay.get("psych_score"),
                            "biases": overlay.get("biases", []),
                            "manipulation_flags": overlay.get("manipulation_flags", []),
                            "comment": overlay.get("notes", ""),
                        }
                        p["confirm"] = overlay.get("confirm_checklist", _default_confirm(p))
                    else:
                        p["psych_notes"] = {"psych_score": None, "biases": [], "manipulation_flags": [], "comment": "NO_OVERLAY"}
            else:
                console.print(f"Ollama overlay failed: {overlay_meta.error} (tried {overlay_meta.models_tried})")
        except Exception as e:
            console.print(f"Ollama overlay error: {e}")

    # ========== WATCHLIST SELECTION ==========
    min_score = float(cfg["scan"]["min_watch_score"])
    watch_k = int(cfg["scan"]["watchlist_k"])

    # Breath override
    if breath and breath.state == "RISK_OFF":
        min_score = max(min_score, float(cfg.get("breath", {}).get("min_watch_score_risk_off", 75)))
        watch_k = max(1, int(round(watch_k * 0.5)))

    candidates = [p for p in plans if p["final_score"] >= min_score]
    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    if not candidates and plans:
        # Keep output useful even if scores are low
        plans_sorted = sorted(plans, key=lambda x: x.get("final_score", 0.0), reverse=True)
        candidates = plans_sorted[:max(1, watch_k)]
        for p in candidates:
            flags = list(p.get("flags", []))
            if "MIN_SCORE_BYPASS" not in flags:
                flags.append("MIN_SCORE_BYPASS")
            p["flags"] = flags

    # In manual mode, include all analyzed symbols in watchlist (up to watchlist_k)
    # but mark those below min_score
    watchlist = candidates[:watch_k]

    # ========== OUTPUT GENERATION ==========
    base_out = pathlib.Path(cfg["outputs"].get("base_dir", cfg["outputs"].get("root", "outputs")))
    latest = base_out / "latest"
    ensure_dir(str(latest))

    # Render table to console
    _render_table(watchlist, side_mode)

    # Build payload
    payload = {
        "analysis_timestamp_utc": utc_now_iso(),
        "mode": "manual",
        "side_mode": side_mode,
        "input_symbols": valid_symbols,
        "errors": errors,  # Include symbols that failed validation
        "market_regime": btc_regime.to_dict(),
        "breath": breath.to_dict() if breath else None,
        "watchlist": watchlist,
        "all_plans": plans,  # Include all analyzed plans for manual mode
    }

    (latest / "payload.json").write_text(json_dumps(payload), encoding="utf-8")

    watch_txt = _manual_watchlist_text(watchlist, side_mode, valid_symbols, errors)
    write_text(str(latest / "watchlist.txt"), watch_txt)

    shortlist_txt = _manual_shortlist_table_text(shortlist)
    write_text(str(latest / "shortlist_table.txt"), shortlist_txt)

    chatgpt_prompt = build_chatgpt_teamlead_prompt(payload)
    write_text(str(latest / "chatgpt_prompt.txt"), chatgpt_prompt)

    if print_prompt:
        console.print("\n" + "=" * 60)
        console.print("ChatGPT Prompt:")
        console.print("=" * 60)
        console.print(chatgpt_prompt)

    console.print(f"\nSaved outputs to {latest.resolve()}")
    return str(latest.resolve())


def _manual_watchlist_text(
    watchlist: List[Dict[str, Any]],
    side_mode: str,
    input_symbols: List[str],
    errors: List[str],
) -> str:
    """Generate watchlist text with manual mode header."""
    lines = []
    lines.append(f"WATCHLIST - MANUAL MODE (side_mode={side_mode})")
    lines.append(f"Input symbols: {', '.join(input_symbols)}")
    if errors:
        lines.append(f"Errors (symbols not found): {', '.join(errors)}")
    lines.append("")

    for i, w in enumerate(watchlist, 1):
        lines.append(f"[{i}] {w['symbol']} {w['side']} | score={w['final_score']:.1f} | setup={w['setup_type']}")
        lines.append(f"Entry: {w['entry_zone']} | SL: {w['stop_loss']} | TP: {w['take_profits']}")
        lines.append(f"ExpectedR(tp2): {w['expected_r']:.2f} | Late: {w['late_status']} (late_atr={w['late_atr']:.2f})")

        if w.get("derivatives"):
            d = w["derivatives"]
            if isinstance(d, dict) and "score" in d:
                lines.append(f"Derivs: score={d['score']} funding_now={d['funding_now']} oi_change_1h={d['oi_change_1h']} flags={d.get('funding_flags', [])+d.get('oi_flags', [])}")

        if w.get("orderflow"):
            o = w["orderflow"]
            if isinstance(o, dict) and "score" in o:
                lines.append(f"Orderflow: score={o['score']} delta_15m={o['delta_15m']} delta_1h={o['delta_1h']} flags={o.get('flags', [])}")

        p = w.get("psych_notes", {})
        lines.append(f"Psych: {p}")

        conf = w.get("confirm", [])
        if conf:
            lines.append(f"Confirm: {conf}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def _manual_shortlist_table_text(shortlist: List[Dict[str, Any]]) -> str:
    """Generate shortlist table text for manual mode."""
    if not shortlist:
        return "SHORTLIST - MANUAL MODE (empty)\n"

    header = "SHORTLIST - MANUAL MODE (prescore ranking)"
    cols = ["#", "Symbol", "PreScore", "Vol24h(USDT)", "Spread%", "ATR%4H", "ADX4H"]
    rows = []

    for i, s in enumerate(shortlist, 1):
        rows.append([
            str(i),
            str(s.get("symbol", "")),
            f"{float(s.get('prescore', 0.0)):.1f}",
            f"{float(s.get('vol24h_usdt', 0.0)):.0f}",
            f"{float(s.get('spread_pct', 0.0)):.2f}",
            f"{float(s.get('atrp4h', 0.0)):.2f}",
            f"{float(s.get('adx4h', 0.0)):.2f}",
        ])

    widths = [max(len(cols[i]), max(len(r[i]) for r in rows)) for i in range(len(cols))]
    lines = [header, ""]
    lines.append(" ".join(cols[i].ljust(widths[i]) for i in range(len(cols))))
    lines.append(" ".join("-" * widths[i] for i in range(len(cols))))

    for r in rows:
        lines.append(" ".join(r[i].ljust(widths[i]) for i in range(len(cols))))

    return "\n".join(lines).strip() + "\n"
