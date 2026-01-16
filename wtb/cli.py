from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .pipeline import run_pipeline
from .config import load_config
from .binance import BinanceFuturesClient
from .backtest import cache_klines


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

    manual = sub.add_parser("manual", help="Build a single-symbol packet")
    manual.add_argument("symbol", help="e.g. ETHUSDT or ETH/USDT")
    manual.add_argument("--side", choices=["LONG", "SHORT"], default=None)
    manual.add_argument("--print-prompt", action="store_true")
    add_scan_overrides(manual)

    cache = sub.add_parser("cache", help="Cache klines for offline backtest")
    cache.add_argument("symbol")
    cache.add_argument("interval", choices=["1d", "4h", "1h", "15m"])
    cache.add_argument("start_ms", type=int, help="UTC ms")
    cache.add_argument("end_ms", type=int, help="UTC ms")
    cache.add_argument("out", help="Output parquet path")

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

    if args.cmd in ("scan", "long", "short", "manual"):
        if args.cmd == "scan":
            side = args.side or cfg.get("scan", {}).get("side_default", "LONG")
        elif args.cmd == "long":
            side = "LONG"
        elif args.cmd == "short":
            side = "SHORT"
        else:
            side = args.side or cfg.get("scan", {}).get("side_default", "LONG")

        symbol = getattr(args, "symbol", None)
        result = run_pipeline(cfg, side_mode=side, manual_symbol=symbol, print_prompt=bool(getattr(args, "print_prompt", False)))
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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
