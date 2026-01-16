from __future__ import annotations

import os
import pathlib
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from rich.console import Console
from rich.table import Table

from .binance import BinanceFuturesClient
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


console = Console()


def prescore_symbol(vol_usdt: float, spread_pct: float, atrp4h: float, adx4h: float) -> float:
    """Soft prescore 0–100 to rank before LLM.

    No hard reject except very high spread or missing data.
    """
    score = 50.0
    # volume: log scale
    score += min(25.0, max(0.0, (np.log10(max(vol_usdt, 1.0)) - 6.0) * 6.0))
    # spread: penalize beyond 0.03%
    score -= min(25.0, max(0.0, (spread_pct - 0.03) * 120.0))
    # atr%: prefer 1.0–8.0
    if atrp4h < 0.7:
        score -= 10.0
    elif atrp4h > 15.0:
        score -= 12.0
    else:
        score += 6.0
    # adx
    if adx4h >= 25.0:
        score += 6.0
    elif adx4h < 15.0:
        score -= 4.0

    return float(max(0.0, min(100.0, score)))


def scan_universe(client: BinanceFuturesClient, cfg: Dict[str, Any], manual_symbol: str | None = None) -> List[Dict[str, Any]]:
    # tickers and book ticker for spreads
    tickers = client.ticker_24hr()
    books = client.book_ticker()
    book_by = {b["symbol"]: b for b in books}

    # USDT perpetual symbols only
    usdt = [t for t in tickers if t.get("symbol", "").endswith("USDT") and not t.get("symbol", "").endswith("BUSD")]

    # volume sort
    usdt.sort(key=lambda x: float(x.get("quoteVolume", 0.0)), reverse=True)
    top = usdt[: int(cfg["scan"]["scan_top"])]

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
        spread_pct = (ask - bid) / ((ask + bid) / 2.0) * 100.0
        out.append({"symbol": sym, "vol24h_usdt": vol, "spread_pct": spread_pct, "bid": bid, "ask": ask})

    if manual_symbol:
        sym = manual_symbol.replace("/", "").upper()
        return [x for x in out if x["symbol"] == sym]

    # soft filter by min volume and max spread
    min_vol = float(cfg["scan"]["min_volume_usdt"])
    max_spread = float(cfg["scan"]["max_spread_pct"])
    out = [x for x in out if x["vol24h_usdt"] >= min_vol and x["spread_pct"] <= max_spread]
    return out


def run_pipeline(
    cfg_or_path: str | Dict[str, Any],
    side_mode: str,
    manual_symbol: str | None = None,
    insecure_ssl: bool = False,
    enable_plutus: bool = False,
    enable_derivatives: bool = True,
    enable_orderflow: bool = True,
    print_prompt: bool = False,
) -> str | None:
    cfg = load_config(cfg_or_path) if isinstance(cfg_or_path, str) else cfg_or_path
    # CLI override
    if insecure_ssl:
        cfg["binance"]["insecure_ssl"] = True

    client = BinanceFuturesClient(
        base_url=str(cfg["binance"]["base_url"]),
        timeout_sec=int(cfg["binance"]["timeout_sec"]),
        insecure_ssl=bool(cfg["binance"]["insecure_ssl"]),
    )

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

    console.print("Loading symbol universe from Binance Futures...")
    universe = scan_universe(client, cfg, manual_symbol=manual_symbol)
    if manual_symbol and not universe:
        console.print(f"Symbol {manual_symbol} not found in futures universe.")
        return None
    console.print(f"Scanning {len(universe)} symbols (after min volume filter)...")

    # Market breath
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

    # Whales context (Hyperliquid)
    whale_ctx = None
    if bool(cfg.get("whales", {}).get("enabled", False)):
        console.print("Fetching Hyperliquid whale context (BTC/ETH only)...")
        try:
            whale_ctx = build_whale_context(cfg.get("whales", {}), now_utc=utc_now_iso())
            console.print(f"Whale context: state={whale_ctx.state}, bullish_score={whale_ctx.bullish_score:.1f}, flags={whale_ctx.flags}")
        except Exception as e:
            console.print(f"Warning: Whale context failed: {e}")
            whale_ctx = None

    # compute 4H features for prescore
    tf = str(cfg["scan"]["timeframes"]["execution"])
    ind = cfg["indicators"]
    max_shortlist = int(cfg["scan"]["shortlist_n"])

    scored: List[Dict[str, Any]] = []
    for item in universe:
        sym = item["symbol"]
        kl = client.klines(sym, interval=tf, limit=200)
        if len(kl) < 60:
            continue
        high = np.array([float(x[2]) for x in kl], dtype=float)
        low = np.array([float(x[3]) for x in kl], dtype=float)
        close = np.array([float(x[4]) for x in kl], dtype=float)
        atr4_series = atr(high, low, close, int(ind["atr_period"]))
        adx4_series = adx(high, low, close, int(ind["adx_period"]))
        if len(atr4_series) == 0 or len(adx4_series) == 0:
            continue
        atr4 = float(atr4_series[-1])
        adx4 = float(adx4_series[-1])
        atrp4 = atr4 / close[-1] * 100.0
        pscore = prescore_symbol(item["vol24h_usdt"], item["spread_pct"], float(atrp4), adx4)
        scored.append({**item, "atrp4h": float(atrp4), "adx4h": adx4, "prescore": float(pscore), "klines_4h": kl})

    scored.sort(key=lambda x: x["prescore"], reverse=True)
    shortlist = scored[:max_shortlist]

    # Build algo plans
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

    # Derivatives + orderflow overlays
    if enable_derivatives and bool(cfg.get("derivatives", {}).get("enabled", True)):
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

    if enable_orderflow and bool(cfg.get("orderflow", {}).get("enabled", True)):
        now_ms = int(time.time() * 1000)
        top_n = int(cfg.get("orderflow", {}).get("top_n", 15))
        w15 = cfg.get("orderflow", {}).get("window_15m_min", cfg.get("orderflow", {}).get("window_min_15", 15))
        w1h = cfg.get("orderflow", {}).get("window_1h_min", cfg.get("orderflow", {}).get("window_min_60", 60))
        limit = int(cfg.get("orderflow", {}).get("aggtrades_limit", 1000))
        for p in plans[:top_n]:
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

    # Whales scoring overlay (BTC/ETH context only)
    for p in plans:
        if whale_ctx:
            try:
                score_for_side = whales_component_score(whale_ctx, side_mode=side_mode, symbol=p["symbol"])
                flags = []
                # Check for opposing whale bias
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

    # Compute final scores FIRST (before Ollama overlay)
    # This allows us to only send top candidates to Ollama
    for p in plans:
        # Set default psych_notes and confirm for scoring
        p["psych_notes"] = {"psych_score": None, "biases": [], "manipulation_flags": [], "comment": "ALGO_ONLY"}
        p["confirm"] = _default_confirm(p)
        p["overlay_flags"] = []
        p["overlay_score"] = 0.0
        final_score, score_detail = _score_breakdown(p, btc_regime, breath, cfg)
        p["final_score"] = float(final_score)
        p["score_detail"] = score_detail

    # Sort plans by score to identify top candidates
    plans.sort(key=lambda x: x["final_score"], reverse=True)

    # Optional: Plutus (Ollama) psychology overlay on TOP candidates only
    plutus_enabled = bool(cfg.get("plutus", {}).get("enabled", False))
    if plutus_enabled:
        # Only analyze top N candidates (default: 10)
        max_overlay_candidates = int(cfg.get("plutus", {}).get("max_candidates", 10))
        top_plans = plans[:max_overlay_candidates]

        console.print(f"Running Ollama psychology overlay on top {len(top_plans)} candidates...")
        try:
            # Prepare whale context dict (WhaleContext is a dataclass without to_dict method)
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
                # Merge overlay results into top_plans only
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
                # Top plans already have ALGO_ONLY defaults from above
        except Exception as e:
            console.print(f"Ollama overlay error: {e}")
            # Top plans already have ALGO_ONLY defaults from above

    # Watchlist selection
    min_score = float(cfg["scan"]["min_watch_score"])
    watch_k = int(cfg["scan"]["watchlist_k"])

    # breath override
    if breath and breath.state == "RISK_OFF":
        min_score = max(min_score, float(cfg.get("breath", {}).get("min_watch_score_risk_off", 75)))
        watch_k = max(1, int(round(watch_k * 0.5)))

    candidates = [p for p in plans if p["final_score"] >= min_score]
    candidates.sort(key=lambda x: x["final_score"], reverse=True)
    if not candidates and plans:
        # Keep algo-only output useful even if scores are low.
        plans_sorted = sorted(plans, key=lambda x: x.get("final_score", 0.0), reverse=True)
        candidates = plans_sorted[: max(1, watch_k)]
        for p in candidates:
            flags = list(p.get("flags", []))
            if "MIN_SCORE_BYPASS" not in flags:
                flags.append("MIN_SCORE_BYPASS")
            p["flags"] = flags
    watchlist = candidates[:watch_k]

    # output dir
    base_out = pathlib.Path(cfg["outputs"].get("base_dir", cfg["outputs"].get("root", "outputs")))
    latest = base_out / "latest"
    ensure_dir(str(latest))

    # human table
    _render_table(watchlist, side_mode)

    # write outputs
    payload = {
        "analysis_timestamp_utc": utc_now_iso(),
        "side_mode": side_mode,
        "market_regime": btc_regime.to_dict(),
        "breath": breath.to_dict() if breath else None,
        "watchlist": watchlist,
    }

    (latest / "payload.json").write_text(json_dumps(payload), encoding="utf-8")

    watch_txt = _watchlist_text(watchlist, side_mode)
    write_text(str(latest / "watchlist.txt"), watch_txt)

    shortlist_txt = _shortlist_table_text(shortlist)
    write_text(str(latest / "shortlist_table.txt"), shortlist_txt)

    chatgpt_prompt = build_chatgpt_teamlead_prompt(payload)
    write_text(str(latest / "chatgpt_prompt.txt"), chatgpt_prompt)

    console.print(f"Saved outputs to {latest.resolve()}")
    return str(latest.resolve())


def _watchlist_text(watchlist: List[Dict[str, Any]], side_mode: str) -> str:
    lines = []
    lines.append(f"WATCHLIST (side_mode={side_mode})")
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


def _shortlist_table_text(shortlist: List[Dict[str, Any]]) -> str:
    if not shortlist:
        return "SHORTLIST (empty)\n"
    header = "SHORTLIST (prescore ranking)"
    cols = ["#", "Symbol", "PreScore", "Vol24h(USDT)", "Spread%", "ATR%4H", "ADX4H"]
    rows = []
    for i, s in enumerate(shortlist, 1):
        rows.append(
            [
                str(i),
                str(s.get("symbol", "")),
                f"{float(s.get('prescore', 0.0)):.1f}",
                f"{float(s.get('vol24h_usdt', 0.0)):.0f}",
                f"{float(s.get('spread_pct', 0.0)):.2f}",
                f"{float(s.get('atrp4h', 0.0)):.2f}",
                f"{float(s.get('adx4h', 0.0)):.2f}",
            ]
        )
    widths = [max(len(cols[i]), max(len(r[i]) for r in rows)) for i in range(len(cols))]
    lines = [header, ""]
    lines.append(" ".join(cols[i].ljust(widths[i]) for i in range(len(cols))))
    lines.append(" ".join("-" * widths[i] for i in range(len(cols))))
    for r in rows:
        lines.append(" ".join(r[i].ljust(widths[i]) for i in range(len(cols))))
    return "\n".join(lines).strip() + "\n"


def _default_confirm(plan: Dict[str, Any]) -> List[str]:
    st = str(plan.get("setup_type", ""))
    side = str(plan.get("side", ""))
    base = []
    if st == "BREAKOUT_RETEST":
        base.append("Retest holds (no 4H close back inside range)")
        base.append("Reclaim candle closes in breakout direction")
    elif st == "RANGE_SWEEP_RECLAIM":
        base.append("Sweep wick then reclaim inside range")
        base.append("No immediate acceptance below/above swept level")
    elif st == "VOLATILITY_FADE":
        base.append("Mean reversion candle (body back inside bands)")
        base.append("Avoid trading into high funding crowding")
    else:
        base.append("Pullback respects structure")
        base.append("No chase: wait entry zone")

    if side == "LONG":
        base.append("If BTC is BEARISH: require extra confirmation / smaller size")
    else:
        base.append("If BTC is BULLISH: require extra confirmation / smaller size")
    return base[:4]


def _final_score(plan: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    return float(_score_breakdown(plan, cfg)[0])



def _score_breakdown(plan: Dict[str, Any], btc_regime, breath, cfg: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Compute final score from component scores.

    This is *algo-only*. There is no local LLM overlay.
    """
    scoring_cfg = cfg.get("scoring", {})
    weight_cfg = scoring_cfg.get("weights", {})

    w_trade = float(weight_cfg.get("tradeability", 35.0))
    w_setup = float(weight_cfg.get("setup_quality", 25.0))
    w_deriv = float(weight_cfg.get("derivatives", 20.0))
    w_of = float(weight_cfg.get("orderflow", 10.0))
    w_context = float(weight_cfg.get("context", 5.0))
    w_whales = float(weight_cfg.get("whales", 5.0))

    tradeability = float(plan.get("score_tradeability", 0.0))
    setup = float(plan.get("score_setup", 0.0))
    derivatives = float(plan.get("derivatives", {}).get("score", 50.0))
    orderflow = float(plan.get("orderflow", {}).get("score", 50.0))
    whales = float(plan.get("whales", {}).get("score_for_side", 50.0))

    # Context score from BTC regime and breath
    context = 50.0
    if btc_regime:
        btc_trend = str(getattr(btc_regime, "btc_trend", "NEUTRAL")).upper()
        side = str(plan.get("side", "LONG")).upper()
        # Bullish BTC helps LONG, bearish helps SHORT
        if side == "LONG":
            if btc_trend == "BULLISH":
                context = 70.0
            elif btc_trend == "BEARISH":
                context = 30.0
        else:  # SHORT
            if btc_trend == "BEARISH":
                context = 70.0
            elif btc_trend == "BULLISH":
                context = 30.0

    # Adjust context by breath state
    if breath and hasattr(breath, "state"):
        if breath.state == "RISK_OFF":
            context = max(0.0, context - 15.0)
        elif breath.state == "RISK_ON":
            context = min(100.0, context + 10.0)

    denom = max(1e-9, (w_trade + w_setup + w_deriv + w_of + w_context + w_whales))
    raw = (
        tradeability * w_trade
        + setup * w_setup
        + derivatives * w_deriv
        + orderflow * w_of
        + context * w_context
        + whales * w_whales
    ) / denom

    penalties = 0.0

    # Late/no-chase penalties
    late_status = plan.get("late_status")
    if late_status in ("WATCH_LATE", "WATCH_PULLBACK"):
        penalties += float(scoring_cfg.get("late_penalty", -4.0))

    # BTC regime headwind
    if btc_regime:
        btc_trend = str(getattr(btc_regime, "btc_trend", "")).upper()
        if btc_trend in ("BEARISH",) and plan.get("side") == "LONG":
            penalties += float(scoring_cfg.get("btc_bear_headwind_penalty", -6.0))
        elif btc_trend in ("BULLISH",) and plan.get("side") == "SHORT":
            penalties += float(scoring_cfg.get("btc_bear_headwind_penalty", -6.0))

    # Whale conflict penalty (soft)
    if "WHALES_OPPOSING" in set(plan.get("whales", {}).get("flags", [])):
        penalties += float(scoring_cfg.get("whales_conflict_penalty", -4.0))

    # Cap total penalty magnitude ("penalty tối đa")
    max_pen = float(scoring_cfg.get("max_penalty_abs", 10.0))
    if max_pen > 0:
        penalties = max(-max_pen, min(max_pen, penalties))

    final = max(0.0, min(100.0, raw + penalties))
    detail = {
        "components": {
            "tradeability": tradeability,
            "setup": setup,
            "derivatives": derivatives,
            "orderflow": orderflow,
            "context": context,
            "whales": whales,
        },
        "weights": {
            "tradeability": w_trade,
            "setup": w_setup,
            "derivatives": w_deriv,
            "orderflow": w_of,
            "context": w_context,
            "whales": w_whales,
        },
        "raw": raw,
        "penalties": penalties,
        "final": final,
    }
    return final, detail



def _render_table(items: List[Dict[str, Any]], side_mode: str) -> None:
    table = Table(title=f"Watchlist (side_mode={side_mode})")
    table.add_column("#", justify="right")
    table.add_column("Symbol")
    table.add_column("Score", justify="right")
    table.add_column("Vol24h(USDT)", justify="right")
    table.add_column("Spread%", justify="right")
    table.add_column("ATR%4H", justify="right")
    table.add_column("ADX4H", justify="right")
    table.add_column("Setup")
    table.add_column("Flags")

    for idx, it in enumerate(items, 1):
        flags = it.get("flags", [])
        if it.get("overlay_flags"):
            flags = list(flags) + list(it.get("overlay_flags", []))
        table.add_row(
            str(idx),
            str(it.get("symbol")),
            f"{it.get('final_score', 0):.1f}",
            f"{it.get('vol24h_usdt', 0):.0f}",
            f"{it.get('spread_pct', 0):.2f}",
            f"{it.get('atrp4h', 0):.2f}",
            f"{it.get('adx4h', 0):.2f}",
            str(it.get("setup_type")),
            ",".join(flags[:6]),
        )

    console.print(table)
