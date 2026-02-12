from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .pipeline import run_pipeline
from .manual import run_manual_pipeline
from .dca import run_dca_pipeline
from .config import load_config
from .binance import BinanceFuturesClient, parse_symbols_input
from .backtest import (
    cache_klines, run_backtest, run_multi_backtest,
    get_top_symbols_by_volume, parse_date_to_ms,
    save_backtest_outputs, build_backtest_chatgpt_prompt,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="whaletraderbot", description="WhaleTraderBot v2.3 (ALGO-only + whales)")
    p.add_argument("--config", default="config.json", help="Path to config.json")
    p.add_argument("--insecure-ssl", action="store_true", help="Disable SSL verification (corporate proxy/self-signed cert)")

    sub = p.add_subparsers(dest="cmd", required=True)

    def add_scan_overrides(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--top", type=int, default=None, help="Override scan_top")
        sp.add_argument("--watchlist-k", type=int, default=None, help="Override watchlist_k")
        sp.add_argument("--min-score", type=int, default=None, help="Override min_watch_score")

    scan = sub.add_parser("scan", help="Scan market and build watchlist")
    scan.add_argument("--side", choices=["LONG", "SHORT"], default=None, help="Override side mode")
    scan.add_argument("--print-prompt", action="store_true", help="Print ChatGPT prompt to stdout")
    add_scan_overrides(scan)

    longp = sub.add_parser("long", help="Shortcut = scan --side LONG")
    longp.add_argument("--print-prompt", action="store_true")
    add_scan_overrides(longp)

    shortp = sub.add_parser("short", help="Shortcut = scan --side SHORT")
    shortp.add_argument("--print-prompt", action="store_true")
    add_scan_overrides(shortp)

    # New manual mode with multi-symbol support
    manual = sub.add_parser("manual", help="Analyze specific symbols (same pipeline as scan)")
    manual.add_argument("--symbols", "-s", type=str, default=None,
                       help='Comma-separated symbols, e.g. "ETH,BTC,PEPE" or "ETHUSDT,BTCUSDT"')
    manual.add_argument("--symbols-file", "-f", type=str, default=None,
                       help="Path to file with symbols (one per line, # for comments)")
    manual.add_argument("--side", choices=["LONG", "SHORT"], default=None,
                       help="Override side mode")
    manual.add_argument("--print-prompt", action="store_true",
                       help="Print ChatGPT prompt to stdout")
    add_scan_overrides(manual)

    cache = sub.add_parser("cache", help="Cache klines for offline backtest")
    cache.add_argument("symbol")
    cache.add_argument("interval", choices=["1d", "4h", "1h", "15m"])
    cache.add_argument("start_ms", type=int, help="UTC ms")
    cache.add_argument("end_ms", type=int, help="UTC ms")
    cache.add_argument("out", help="Output parquet path")

    # Backtest subcommand
    bt = sub.add_parser("backtest", help="Run walk-forward backtest on historical klines")
    bt.add_argument("--symbols", "-s", type=str, default=None,
                     help='Comma-separated symbols (e.g., "BTC,ETH,SOL")')
    bt.add_argument("--top", type=int, default=None,
                     help="Backtest top N symbols by 24h volume (e.g., --top 50)")
    bt.add_argument("--side", choices=["LONG", "SHORT"], default="LONG",
                     help="Trade side (default: LONG)")
    bt.add_argument("--interval", choices=["1d", "4h", "1h", "15m"], default="4h",
                     help="Kline interval (default: 4h)")
    bt.add_argument("--start-date", type=str, default=None,
                     help="Start date for backtest (YYYY-MM-DD or YYYY-MM-DD HH:MM)")
    bt.add_argument("--end-date", type=str, default=None,
                     help="End date for backtest (YYYY-MM-DD or YYYY-MM-DD HH:MM, default: now)")
    bt.add_argument("--bars", type=int, default=500,
                     help="Total bars to download if no date range specified (default: 500)")
    bt.add_argument("--cooldown", type=int, default=12,
                     help="Min bars between signals (default: 12)")
    bt.add_argument("--fill-window", type=int, default=12,
                     help="Bars to wait for fill (default: 12)")
    bt.add_argument("--timeout", type=int, default=48,
                     help="Max bars to hold trade (default: 48)")
    bt.add_argument("--print-prompt", action="store_true",
                     help="Print ChatGPT analysis prompt to stdout")
    bt.add_argument("--save-outputs", action="store_true",
                     help="Save results to outputs/latest/ (JSON, text, ChatGPT prompt)")

    # DCA Discovery subcommand
    dca = sub.add_parser("dca", help="DCA Discovery - find symbols suitable for DCA bot operation")
    dca.add_argument("--side", choices=["LONG", "SHORT", "BOTH"], default=None,
                     help="Side mode: LONG, SHORT, or BOTH (default: from config)")
    dca.add_argument("--top", type=int, default=None,
                     help="Override scan_top (number of symbols to evaluate)")
    dca.add_argument("--watchlist-k", type=int, default=None,
                     help="Override watchlist_k (max symbols in final list)")
    dca.add_argument("--min-score", type=int, default=None,
                     help="Override min_dca_score threshold")
    dca.add_argument("--print-prompt", action="store_true",
                     help="Print ChatGPT prompt to stdout")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = load_config(args.config)
    if args.insecure_ssl:
        cfg.setdefault("binance", {})["insecure_ssl"] = True
    if args.cmd in ("scan", "long", "short", "manual"):
        scan_cfg = cfg.setdefault("scan", {})
        if getattr(args, "top", None) is not None:
            scan_cfg["scan_top"] = int(args.top)
        if getattr(args, "watchlist_k", None) is not None:
            scan_cfg["watchlist_k"] = int(args.watchlist_k)
        if getattr(args, "min_score", None) is not None:
            scan_cfg["min_watch_score"] = int(args.min_score)

    if args.cmd in ("scan", "long", "short"):
        if args.cmd == "scan":
            side = args.side or cfg.get("scan", {}).get("side_default", "LONG")
        elif args.cmd == "long":
            side = "LONG"
        else:  # short
            side = "SHORT"

        result = run_pipeline(cfg, side_mode=side, print_prompt=bool(getattr(args, "print_prompt", False)))
        if result is None:
            return 2
        return 0

    if args.cmd == "manual":
        side = args.side or cfg.get("scan", {}).get("side_default", "LONG")

        # Parse symbols from --symbols argument
        symbols = None
        if args.symbols:
            symbols = parse_symbols_input(args.symbols)

        result = run_manual_pipeline(
            cfg,
            side_mode=side,
            symbols=symbols,
            symbols_file=args.symbols_file,
            print_prompt=bool(getattr(args, "print_prompt", False)),
        )
        if result is None:
            return 2
        return 0

    if args.cmd == "cache":
        bcfg = cfg.get("binance", {})
        client = BinanceFuturesClient(
            base_url=bcfg.get("base_url", "https://fapi.binance.com"),
            timeout_sec=int(bcfg.get("timeout_sec", 15)),
            insecure_ssl=bool(bcfg.get("insecure_ssl", False)),
        )
        out_path = pathlib.Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cache_klines(client, args.symbol.replace("/", ""), args.interval, args.start_ms, args.end_ms, out_path)
        print(str(out_path))
        return 0

    if args.cmd == "backtest":
        bcfg = cfg.get("binance", {})
        client = BinanceFuturesClient(
            base_url=bcfg.get("base_url", "https://fapi.binance.com"),
            timeout_sec=int(bcfg.get("timeout_sec", 15)),
            insecure_ssl=bool(bcfg.get("insecure_ssl", False)),
        )

        # Determine symbols: --top N, --symbols, or default to BTCUSDT
        if args.top is not None:
            symbols = get_top_symbols_by_volume(client, args.top)
        elif args.symbols:
            symbols = [s.strip().upper().replace("/", "") for s in args.symbols.split(",") if s.strip()]
            for i, s in enumerate(symbols):
                if not s.endswith("USDT"):
                    symbols[i] = s + "USDT"
        else:
            symbols = ["BTCUSDT"]

        # Parse date range if specified
        start_ms = None
        end_ms = None
        if args.start_date:
            try:
                start_ms = parse_date_to_ms(args.start_date)
            except ValueError as e:
                print(f"Error: {e}")
                return 1
        if args.end_date:
            try:
                end_ms = parse_date_to_ms(args.end_date)
            except ValueError as e:
                print(f"Error: {e}")
                return 1

        ind = cfg.get("indicators", {})
        late_cfg = cfg.get("scoring", {}).get("late", {})

        common_kwargs = {
            "interval": args.interval,
            "lookback_bars": args.bars,
            "signal_cooldown": args.cooldown,
            "fill_window": args.fill_window,
            "trade_timeout": args.timeout,
            "atr_period": int(ind.get("atr_period", 14)),
            "adx_period": int(ind.get("adx_period", 14)),
            "range_lookback": int(ind.get("range_lookback_bars", 48)),
            "late_ok_atr": float(late_cfg.get("ok_atr", 0.5)),
            "late_watch_atr": float(late_cfg.get("watch_atr", 1.5)),
            "start_ms": start_ms,
            "end_ms": end_ms,
        }

        if len(symbols) == 1:
            result = run_backtest(client, symbols[0], args.side, **common_kwargs)
            results = [result]
        else:
            results = run_multi_backtest(client, symbols, args.side, **common_kwargs)

        # Save outputs if requested
        if args.save_outputs or args.print_prompt:
            save_backtest_outputs(
                results, args.side, args.interval,
                start_date=args.start_date,
                end_date=args.end_date,
            )

        # Print ChatGPT prompt if requested
        if args.print_prompt:
            prompt = build_backtest_chatgpt_prompt(
                results, args.side, args.interval,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            print("\n" + prompt)

        return 0

    if args.cmd == "dca":
        dca_cfg = cfg.setdefault("dca", {})

        # Apply CLI overrides
        if getattr(args, "top", None) is not None:
            dca_cfg["scan_top"] = int(args.top)
        if getattr(args, "watchlist_k", None) is not None:
            dca_cfg["watchlist_k"] = int(args.watchlist_k)
        if getattr(args, "min_score", None) is not None:
            dca_cfg["min_dca_score"] = int(args.min_score)

        side = args.side or dca_cfg.get("side_default", "LONG")

        result = run_dca_pipeline(
            cfg,
            side_mode=side,
            print_prompt=bool(getattr(args, "print_prompt", False)),
        )
        if result is None:
            return 2
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
