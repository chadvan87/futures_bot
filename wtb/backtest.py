from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from .binance import BinanceFuturesClient
from .algo import build_algo_plan, AlgoPlan
from .indicators import atr, adx, ema
from .regime import detect_btc_regime
from .utils import ensure_dir, json_dumps, utc_now_iso, write_text


console = Console()


# ---------------------------------------------------------------------------
# Date parsing utilities
# ---------------------------------------------------------------------------

def parse_date_to_ms(date_str: str) -> int:
    """Parse date string (YYYY-MM-DD or YYYY-MM-DD HH:MM) to UTC milliseconds."""
    formats = ["%Y-%m-%d %H:%M", "%Y-%m-%d"]
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD or YYYY-MM-DD HH:MM")


def ms_to_date_str(ms: int) -> str:
    """Convert milliseconds to human-readable date string."""
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# Top N symbols by volume
# ---------------------------------------------------------------------------

def get_top_symbols_by_volume(client: BinanceFuturesClient, top_n: int = 100) -> List[str]:
    """Fetch top N USDT perpetual symbols ranked by 24h quote volume."""
    console.print(f"[cyan]Fetching top {top_n} symbols by 24h volume...[/cyan]")

    # Get exchange info for all symbols
    exchange_info = client.exchange_info()
    usdt_perps = set()
    for sym_info in exchange_info.get("symbols", []):
        if (sym_info.get("quoteAsset") == "USDT" and
            sym_info.get("contractType") == "PERPETUAL" and
            sym_info.get("status") == "TRADING"):
            usdt_perps.add(sym_info["symbol"])

    # Get 24h ticker for volume data
    tickers = client.ticker_24h()

    # Filter and sort by quote volume
    volume_data = []
    for t in tickers:
        symbol = t.get("symbol", "")
        if symbol in usdt_perps:
            quote_vol = float(t.get("quoteVolume", 0))
            volume_data.append((symbol, quote_vol))

    volume_data.sort(key=lambda x: x[1], reverse=True)
    top_symbols = [s[0] for s in volume_data[:top_n]]

    console.print(f"[green]Found {len(top_symbols)} symbols[/green]")
    return top_symbols


def _download_klines_range(
    client: BinanceFuturesClient,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    quiet: bool = False,
) -> List[List]:
    """Download all klines between start_ms and end_ms (paginated)."""
    rows: List[List] = []
    cursor = start_ms
    step = 1500  # max limit per Binance API call

    while cursor < end_ms:
        batch = client.klines(symbol, interval, limit=step, start_ms=cursor, end_ms=end_ms)
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        if last_open == cursor:
            break
        cursor = last_open + 1

    if not quiet and rows:
        console.print(f"  Downloaded {len(rows)} bars for {symbol}")
    return rows


# ---------------------------------------------------------------------------
# Parquet caching (original functionality)
# ---------------------------------------------------------------------------

def cache_klines(client: BinanceFuturesClient, symbol: str, interval: str, start_ms: int, end_ms: int, out_path: pathlib.Path) -> pathlib.Path:
    """Download klines into a Parquet cache (offline backtest support)."""
    rows: List[List] = []
    cursor = start_ms
    step = 1500  # max limit per Binance API call
    while True:
        batch = client.klines(symbol, interval, limit=step, start_ms=cursor, end_ms=end_ms)
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        if last_open == cursor:
            break
        cursor = last_open + 1
        if cursor >= end_ms:
            break
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"
    ])
    if df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        return out_path
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_base", "taker_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Backtest engine — replays WhaleTraderBot algo signals on historical klines
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """A single completed or open trade."""
    bar_index: int           # bar where signal was generated
    fill_index: int          # bar where entry was filled (-1 if never filled)
    exit_index: int          # bar where trade was closed (-1 if still open)
    symbol: str
    side: str
    setup_type: str
    entry_low: float
    entry_high: float
    fill_price: float        # actual fill price (mid of zone touch)
    stop_loss: float
    take_profits: List[float]
    tp_hit: int              # which TP was hit (0=none, 1/2/3)
    exit_price: float
    exit_reason: str         # "TP1", "TP2", "TP3", "SL", "TIMEOUT", "OPEN"
    pnl_pct: float           # percentage PnL (before fees)
    r_multiple: float        # realized R-multiple
    algo_score: float        # algo score_setup at signal time


@dataclass
class BacktestResult:
    """Aggregated backtest statistics."""
    symbol: str
    side: str
    interval: str
    total_bars: int
    signals_generated: int
    trades_filled: int
    trades_won: int
    trades_lost: int
    trades_breakeven: int
    trades_timeout: int
    win_rate: float
    avg_r: float
    total_r: float
    max_r: float
    min_r: float
    profit_factor: float
    avg_pnl_pct: float
    total_pnl_pct: float
    max_drawdown_r: float
    trades: List[Trade]
    equity_curve: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "interval": self.interval,
            "total_bars": self.total_bars,
            "signals_generated": self.signals_generated,
            "trades_filled": self.trades_filled,
            "trades_won": self.trades_won,
            "trades_lost": self.trades_lost,
            "trades_breakeven": self.trades_breakeven,
            "trades_timeout": self.trades_timeout,
            "win_rate": round(self.win_rate, 4),
            "avg_r": round(self.avg_r, 4),
            "total_r": round(self.total_r, 4),
            "max_r": round(self.max_r, 4),
            "min_r": round(self.min_r, 4),
            "profit_factor": round(self.profit_factor, 4),
            "avg_pnl_pct": round(self.avg_pnl_pct, 4),
            "total_pnl_pct": round(self.total_pnl_pct, 4),
            "max_drawdown_r": round(self.max_drawdown_r, 4),
            "num_trades": len(self.trades),
        }


def _parse_entry_zone(zone_str: str) -> Tuple[float, float]:
    """Parse 'low-high' entry zone string from AlgoPlan."""
    parts = zone_str.split("-")
    if len(parts) == 2:
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            pass
    # Fallback: sometimes scientific notation uses '-' for negative exponents
    # Try to find the separator dash that isn't part of a number
    for i in range(1, len(zone_str)):
        if zone_str[i] == '-' and zone_str[i - 1] not in ('e', 'E', '+', '-'):
            try:
                lo = float(zone_str[:i])
                hi = float(zone_str[i + 1:])
                return lo, hi
            except ValueError:
                continue
    return 0.0, 0.0


def run_backtest(
    client: BinanceFuturesClient,
    symbol: str,
    side_mode: str,
    interval: str = "4h",
    lookback_bars: int = 500,
    signal_lookback: int = 100,
    signal_cooldown: int = 12,
    fill_window: int = 12,
    trade_timeout: int = 48,
    atr_period: int = 14,
    adx_period: int = 14,
    range_lookback: int = 48,
    late_ok_atr: float = 0.5,
    late_watch_atr: float = 1.5,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    quiet: bool = False,
) -> BacktestResult:
    """Run a walk-forward backtest of the WhaleTraderBot algo on historical klines.

    How it works:
    1. Download historical klines for the symbol
    2. Detect BTC regime from BTC 1D data (used as context throughout)
    3. Walk forward bar-by-bar starting from bar `signal_lookback`
    4. At each bar, run build_algo_plan() using the last `signal_lookback` bars
    5. If a signal is generated and no trade is active, look for entry fill
       in the next `fill_window` bars (price must touch the entry zone)
    6. Once filled, monitor for SL hit, TP hit, or timeout
    7. Collect all trades and compute statistics

    Args:
        client: BinanceFuturesClient instance
        symbol: e.g. "BTCUSDT"
        side_mode: "LONG" or "SHORT"
        interval: kline interval (default "4h")
        lookback_bars: total bars to download (default 500, ignored if start_ms/end_ms set)
        signal_lookback: bars of history needed before generating first signal (default 100)
        signal_cooldown: minimum bars between signals (default 12 = ~2 days on 4h)
        fill_window: bars to wait for price to reach entry zone (default 12)
        trade_timeout: max bars to hold a trade before forced exit (default 48)
        atr_period: ATR period for algo
        adx_period: ADX period for algo
        range_lookback: range lookback for algo
        late_ok_atr: late OK threshold in ATR units
        late_watch_atr: late WATCH threshold in ATR units
        start_ms: Start time in UTC milliseconds (optional, overrides lookback_bars)
        end_ms: End time in UTC milliseconds (optional, defaults to now)
        quiet: Suppress per-symbol output (useful for multi-symbol runs)
    """
    if not quiet:
        console.print(f"[bold]Backtest: {symbol} {side_mode} ({interval})[/bold]")

    # Download klines for the target symbol
    if start_ms is not None:
        # Date-range mode: download between start_ms and end_ms
        if end_ms is None:
            end_ms = int(time.time() * 1000)
        if not quiet:
            console.print(f"Downloading {interval} klines from {ms_to_date_str(start_ms)} to {ms_to_date_str(end_ms)}...")
        klines_raw = _download_klines_range(client, symbol, interval, start_ms, end_ms, quiet)
    else:
        # Legacy mode: use lookback_bars
        if not quiet:
            console.print(f"Downloading {lookback_bars} bars of {interval} klines...")
        klines_raw = client.klines(symbol, interval, limit=lookback_bars)

    if len(klines_raw) < signal_lookback + 20:
        if not quiet:
            console.print(f"[red]Not enough data: got {len(klines_raw)} bars, need at least {signal_lookback + 20}[/red]")
        return _empty_result(symbol, side_mode, interval)

    open_arr = np.array([float(x[1]) for x in klines_raw], dtype=float)
    high_arr = np.array([float(x[2]) for x in klines_raw], dtype=float)
    low_arr = np.array([float(x[3]) for x in klines_raw], dtype=float)
    close_arr = np.array([float(x[4]) for x in klines_raw], dtype=float)
    volume_arr = np.array([float(x[5]) for x in klines_raw], dtype=float)
    times = [int(x[0]) for x in klines_raw]

    total_bars = len(close_arr)
    if not quiet:
        console.print(f"Loaded {total_bars} bars. Price range: {close_arr.min():.4f} - {close_arr.max():.4f}")

    # BTC regime (use 1D data for context — same as live pipeline)
    if not quiet:
        console.print("Fetching BTC 1D regime context...")
    btc_kl = client.klines("BTCUSDT", interval="1d", limit=250)
    btc_high = np.array([float(x[2]) for x in btc_kl], dtype=float)
    btc_low = np.array([float(x[3]) for x in btc_kl], dtype=float)
    btc_close = np.array([float(x[4]) for x in btc_kl], dtype=float)
    btc_open = np.array([float(x[1]) for x in btc_kl], dtype=float)
    btc_regime = detect_btc_regime(btc_open, btc_high, btc_low, btc_close)
    if not quiet:
        console.print(f"BTC regime: {btc_regime.btc_trend}")

    # Walk-forward simulation
    trades: List[Trade] = []
    signals_generated = 0
    last_signal_bar = -signal_cooldown  # allow first signal immediately
    active_trade: Optional[Dict[str, Any]] = None

    if not quiet:
        console.print(f"Running walk-forward from bar {signal_lookback} to {total_bars - 1}...")

    for bar_i in range(signal_lookback, total_bars):
        # --- Check active trade for exit conditions ---
        if active_trade is not None:
            at = active_trade
            bar_high = high_arr[bar_i]
            bar_low = low_arr[bar_i]
            bar_close = close_arr[bar_i]

            # Check if we're still waiting for fill
            if at["fill_index"] == -1:
                # Check if price enters the entry zone on this bar
                entry_lo = at["entry_low"]
                entry_hi = at["entry_high"]
                touched = False
                if side_mode == "LONG":
                    # For LONG: price must dip into entry zone (low <= entry_high)
                    touched = bar_low <= entry_hi
                else:
                    # For SHORT: price must rise into entry zone (high >= entry_low)
                    touched = bar_high >= entry_lo

                if touched:
                    # Fill at midpoint of entry zone
                    fill_price = (entry_lo + entry_hi) / 2.0
                    at["fill_index"] = bar_i
                    at["fill_price"] = fill_price
                elif bar_i - at["bar_index"] >= fill_window:
                    # Fill window expired — cancel this signal
                    active_trade = None
                continue

            # Trade is filled — check SL/TP
            fill_px = at["fill_price"]
            sl = at["stop_loss"]
            tps = at["take_profits"]
            bars_in_trade = bar_i - at["fill_index"]

            exit_price = None
            exit_reason = None
            tp_hit = 0

            if side_mode == "LONG":
                # Check SL first (worst case)
                if bar_low <= sl:
                    exit_price = sl
                    exit_reason = "SL"
                # Check TPs (best to worst: TP3 > TP2 > TP1)
                elif len(tps) >= 3 and bar_high >= tps[2]:
                    exit_price = tps[2]
                    exit_reason = "TP3"
                    tp_hit = 3
                elif len(tps) >= 2 and bar_high >= tps[1]:
                    exit_price = tps[1]
                    exit_reason = "TP2"
                    tp_hit = 2
                elif len(tps) >= 1 and bar_high >= tps[0]:
                    exit_price = tps[0]
                    exit_reason = "TP1"
                    tp_hit = 1
                elif bars_in_trade >= trade_timeout:
                    exit_price = bar_close
                    exit_reason = "TIMEOUT"
            else:  # SHORT
                if bar_high >= sl:
                    exit_price = sl
                    exit_reason = "SL"
                elif len(tps) >= 3 and bar_low <= tps[2]:
                    exit_price = tps[2]
                    exit_reason = "TP3"
                    tp_hit = 3
                elif len(tps) >= 2 and bar_low <= tps[1]:
                    exit_price = tps[1]
                    exit_reason = "TP2"
                    tp_hit = 2
                elif len(tps) >= 1 and bar_low <= tps[0]:
                    exit_price = tps[0]
                    exit_reason = "TP1"
                    tp_hit = 1
                elif bars_in_trade >= trade_timeout:
                    exit_price = bar_close
                    exit_reason = "TIMEOUT"

            if exit_price is not None:
                # Calculate PnL
                if side_mode == "LONG":
                    pnl_pct = (exit_price - fill_px) / fill_px * 100.0
                else:
                    pnl_pct = (fill_px - exit_price) / fill_px * 100.0

                risk = abs(fill_px - sl)
                r_mult = (exit_price - fill_px) / risk if risk > 0 and side_mode == "LONG" else (fill_px - exit_price) / risk if risk > 0 else 0.0

                trade = Trade(
                    bar_index=at["bar_index"],
                    fill_index=at["fill_index"],
                    exit_index=bar_i,
                    symbol=symbol,
                    side=side_mode,
                    setup_type=at["setup_type"],
                    entry_low=at["entry_low"],
                    entry_high=at["entry_high"],
                    fill_price=fill_px,
                    stop_loss=sl,
                    take_profits=tps,
                    tp_hit=tp_hit,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    pnl_pct=pnl_pct,
                    r_multiple=r_mult,
                    algo_score=at["algo_score"],
                )
                trades.append(trade)
                active_trade = None

        # --- Generate new signal if no active trade and cooldown elapsed ---
        if active_trade is None and (bar_i - last_signal_bar) >= signal_cooldown:
            # Slice history up to current bar (inclusive)
            h_slice = high_arr[:bar_i + 1]
            l_slice = low_arr[:bar_i + 1]
            c_slice = close_arr[:bar_i + 1]
            cur_price = float(c_slice[-1])

            # Use recent volume as proxy for 24h volume
            vol_slice = volume_arr[max(0, bar_i - 5):bar_i + 1]
            vol_proxy = float(np.sum(vol_slice)) * cur_price  # rough USDT estimate

            plan = build_algo_plan(
                symbol=symbol,
                side_mode=side_mode,
                close=c_slice,
                high=h_slice,
                low=l_slice,
                current_price=cur_price,
                bid=cur_price * 0.9999,
                ask=cur_price * 1.0001,
                volume_usdt=vol_proxy,
                btc_trend=btc_regime.btc_trend,
                range_lookback=range_lookback,
                atr_period=atr_period,
                adx_period=adx_period,
                late_ok_atr=late_ok_atr,
                late_watch_atr=late_watch_atr,
            )

            signals_generated += 1
            last_signal_bar = bar_i

            # Only take signals where status is OK (not too late)
            if plan.status_hint == "OK" and plan.score_setup >= 40:
                entry_lo, entry_hi = _parse_entry_zone(plan.entry_zone)
                if entry_lo > 0 and entry_hi > 0:
                    active_trade = {
                        "bar_index": bar_i,
                        "fill_index": -1,
                        "fill_price": 0.0,
                        "entry_low": entry_lo,
                        "entry_high": entry_hi,
                        "stop_loss": plan.stop_loss,
                        "take_profits": plan.take_profits,
                        "setup_type": plan.setup_type,
                        "algo_score": plan.score_setup,
                    }

    # Close any remaining open trade at last bar
    if active_trade is not None and active_trade["fill_index"] != -1:
        at = active_trade
        fill_px = at["fill_price"]
        exit_price = float(close_arr[-1])
        sl = at["stop_loss"]
        if side_mode == "LONG":
            pnl_pct = (exit_price - fill_px) / fill_px * 100.0
        else:
            pnl_pct = (fill_px - exit_price) / fill_px * 100.0
        risk = abs(fill_px - sl)
        r_mult = (exit_price - fill_px) / risk if risk > 0 and side_mode == "LONG" else (fill_px - exit_price) / risk if risk > 0 else 0.0
        trade = Trade(
            bar_index=at["bar_index"],
            fill_index=at["fill_index"],
            exit_index=total_bars - 1,
            symbol=symbol,
            side=side_mode,
            setup_type=at["setup_type"],
            entry_low=at["entry_low"],
            entry_high=at["entry_high"],
            fill_price=fill_px,
            stop_loss=sl,
            take_profits=at["take_profits"],
            tp_hit=0,
            exit_price=exit_price,
            exit_reason="OPEN",
            pnl_pct=pnl_pct,
            r_multiple=r_mult,
            algo_score=at["algo_score"],
        )
        trades.append(trade)

    # Compute stats
    result = _compute_stats(symbol, side_mode, interval, total_bars, signals_generated, trades)
    if not quiet:
        _print_results(result)
    return result


def _empty_result(symbol: str, side: str, interval: str) -> BacktestResult:
    return BacktestResult(
        symbol=symbol, side=side, interval=interval, total_bars=0,
        signals_generated=0, trades_filled=0, trades_won=0, trades_lost=0,
        trades_breakeven=0, trades_timeout=0, win_rate=0.0, avg_r=0.0,
        total_r=0.0, max_r=0.0, min_r=0.0, profit_factor=0.0,
        avg_pnl_pct=0.0, total_pnl_pct=0.0, max_drawdown_r=0.0,
        trades=[], equity_curve=[],
    )


def _compute_stats(
    symbol: str, side: str, interval: str,
    total_bars: int, signals_generated: int,
    trades: List[Trade],
) -> BacktestResult:
    if not trades:
        result = _empty_result(symbol, side, interval)
        result.total_bars = total_bars
        result.signals_generated = signals_generated
        return result

    filled = [t for t in trades if t.fill_price > 0]
    won = [t for t in filled if t.r_multiple > 0.05]
    lost = [t for t in filled if t.r_multiple < -0.05]
    be = [t for t in filled if -0.05 <= t.r_multiple <= 0.05]
    timeouts = [t for t in filled if t.exit_reason == "TIMEOUT"]

    r_values = [t.r_multiple for t in filled]
    pnl_values = [t.pnl_pct for t in filled]

    total_r = sum(r_values)
    avg_r = np.mean(r_values) if r_values else 0.0
    max_r = max(r_values) if r_values else 0.0
    min_r = min(r_values) if r_values else 0.0

    gross_profit = sum(r for r in r_values if r > 0)
    gross_loss = abs(sum(r for r in r_values if r < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

    # Equity curve & drawdown (in R units)
    equity = []
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for r in r_values:
        cumulative += r
        equity.append(cumulative)
        peak = max(peak, cumulative)
        dd = peak - cumulative
        max_dd = max(max_dd, dd)

    win_rate = len(won) / len(filled) if filled else 0.0

    return BacktestResult(
        symbol=symbol,
        side=side,
        interval=interval,
        total_bars=total_bars,
        signals_generated=signals_generated,
        trades_filled=len(filled),
        trades_won=len(won),
        trades_lost=len(lost),
        trades_breakeven=len(be),
        trades_timeout=len(timeouts),
        win_rate=win_rate,
        avg_r=float(avg_r),
        total_r=float(total_r),
        max_r=float(max_r),
        min_r=float(min_r),
        profit_factor=float(profit_factor),
        avg_pnl_pct=float(np.mean(pnl_values)) if pnl_values else 0.0,
        total_pnl_pct=float(sum(pnl_values)),
        max_drawdown_r=float(max_dd),
        trades=trades,
        equity_curve=equity,
    )


def _print_results(result: BacktestResult) -> None:
    console.print()
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold cyan]  BACKTEST RESULTS: {result.symbol} {result.side} ({result.interval})[/bold cyan]")
    console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total Bars", str(result.total_bars))
    table.add_row("Signals Generated", str(result.signals_generated))
    table.add_row("Trades Filled", str(result.trades_filled))
    table.add_row("Trades Won", f"[green]{result.trades_won}[/green]")
    table.add_row("Trades Lost", f"[red]{result.trades_lost}[/red]")
    table.add_row("Trades Breakeven", str(result.trades_breakeven))
    table.add_row("Trades Timeout", str(result.trades_timeout))
    table.add_row("", "")
    table.add_row("Win Rate", f"[bold]{result.win_rate:.1%}[/bold]")
    table.add_row("Average R", f"[bold]{result.avg_r:+.3f}R[/bold]")
    table.add_row("Total R", f"[bold {'green' if result.total_r > 0 else 'red'}]{result.total_r:+.3f}R[/bold {'green' if result.total_r > 0 else 'red'}]")
    table.add_row("Best Trade", f"[green]{result.max_r:+.3f}R[/green]")
    table.add_row("Worst Trade", f"[red]{result.min_r:+.3f}R[/red]")
    table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
    table.add_row("", "")
    table.add_row("Avg PnL %", f"{result.avg_pnl_pct:+.3f}%")
    table.add_row("Total PnL %", f"[bold]{result.total_pnl_pct:+.3f}%[/bold]")
    table.add_row("Max Drawdown", f"[red]{result.max_drawdown_r:.3f}R[/red]")

    console.print(table)
    console.print()

    # Trade log
    if result.trades:
        trade_table = Table(title="Trade Log")
        trade_table.add_column("#", justify="right")
        trade_table.add_column("Setup")
        trade_table.add_column("Fill Price", justify="right")
        trade_table.add_column("Exit Price", justify="right")
        trade_table.add_column("Exit Reason")
        trade_table.add_column("PnL %", justify="right")
        trade_table.add_column("R-Multiple", justify="right")
        trade_table.add_column("Bars Held", justify="right")

        for i, t in enumerate(result.trades, 1):
            bars_held = t.exit_index - t.fill_index if t.fill_index >= 0 else 0
            pnl_style = "green" if t.pnl_pct > 0 else "red" if t.pnl_pct < 0 else "white"
            reason_style = "green" if "TP" in t.exit_reason else "red" if t.exit_reason == "SL" else "yellow"

            trade_table.add_row(
                str(i),
                t.setup_type,
                f"{t.fill_price:.4f}",
                f"{t.exit_price:.4f}",
                f"[{reason_style}]{t.exit_reason}[/{reason_style}]",
                f"[{pnl_style}]{t.pnl_pct:+.3f}%[/{pnl_style}]",
                f"[{pnl_style}]{t.r_multiple:+.3f}R[/{pnl_style}]",
                str(bars_held),
            )

        console.print(trade_table)
        console.print()

    # Setup type breakdown
    if result.trades:
        setup_stats: Dict[str, List[float]] = {}
        for t in result.trades:
            setup_stats.setdefault(t.setup_type, []).append(t.r_multiple)

        breakdown = Table(title="Performance by Setup Type")
        breakdown.add_column("Setup Type")
        breakdown.add_column("Count", justify="right")
        breakdown.add_column("Win Rate", justify="right")
        breakdown.add_column("Avg R", justify="right")
        breakdown.add_column("Total R", justify="right")

        for st, rs in sorted(setup_stats.items()):
            wins = sum(1 for r in rs if r > 0.05)
            wr = wins / len(rs) if rs else 0
            breakdown.add_row(
                st,
                str(len(rs)),
                f"{wr:.1%}",
                f"{np.mean(rs):+.3f}R",
                f"{sum(rs):+.3f}R",
            )
        console.print(breakdown)


def run_multi_backtest(
    client: BinanceFuturesClient,
    symbols: List[str],
    side_mode: str,
    interval: str = "4h",
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
    **kwargs: Any,
) -> List[BacktestResult]:
    """Run backtest across multiple symbols and print a summary.

    Args:
        client: BinanceFuturesClient instance
        symbols: List of symbols to backtest
        side_mode: "LONG" or "SHORT"
        interval: Kline interval (default "4h")
        start_ms: Start time in UTC milliseconds (optional)
        end_ms: End time in UTC milliseconds (optional)
        **kwargs: Additional arguments passed to run_backtest
    """
    results: List[BacktestResult] = []
    errors: List[Tuple[str, str]] = []

    # Show date range info if specified
    if start_ms is not None:
        end_display = ms_to_date_str(end_ms) if end_ms else "now"
        console.print(f"[bold cyan]Multi-Symbol Backtest: {len(symbols)} symbols[/bold cyan]")
        console.print(f"[cyan]Date range: {ms_to_date_str(start_ms)} to {end_display}[/cyan]")
        console.print(f"[cyan]Side: {side_mode}, Interval: {interval}[/cyan]")
        console.print()

    # Use progress bar for multi-symbol runs
    use_quiet = len(symbols) > 3
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
        disable=not use_quiet,
    ) as progress:
        task = progress.add_task(f"Backtesting {len(symbols)} symbols...", total=len(symbols))

        for sym in symbols:
            try:
                result = run_backtest(
                    client, sym, side_mode, interval,
                    start_ms=start_ms, end_ms=end_ms,
                    quiet=use_quiet,
                    **kwargs
                )
                results.append(result)
                if use_quiet:
                    # Show inline progress for large runs
                    style = "green" if result.total_r > 0 else "red" if result.total_r < 0 else "white"
                    progress.console.print(
                        f"  {sym}: {result.trades_filled} trades, [{style}]{result.total_r:+.2f}R[/{style}]"
                    )
            except Exception as e:
                errors.append((sym, str(e)))
                if not use_quiet:
                    console.print(f"[red]Error backtesting {sym}: {e}[/red]")
            progress.advance(task)

    # Report errors
    if errors:
        console.print()
        console.print(f"[yellow]Errors ({len(errors)} symbols):[/yellow]")
        for sym, err in errors[:10]:  # Show first 10 errors
            console.print(f"  [red]{sym}[/red]: {err}")
        if len(errors) > 10:
            console.print(f"  ... and {len(errors) - 10} more")

    if len(results) > 1:
        _print_portfolio_summary(results)
    elif len(results) == 1:
        _print_results(results[0])

    return results


def _print_portfolio_summary(results: List[BacktestResult]) -> None:
    console.print()
    console.print(f"[bold magenta]{'=' * 60}[/bold magenta]")
    console.print(f"[bold magenta]  PORTFOLIO BACKTEST SUMMARY[/bold magenta]")
    console.print(f"[bold magenta]{'=' * 60}[/bold magenta]")

    summary = Table(title="Summary by Symbol")
    summary.add_column("Symbol")
    summary.add_column("Trades", justify="right")
    summary.add_column("Win Rate", justify="right")
    summary.add_column("Total R", justify="right")
    summary.add_column("Avg R", justify="right")
    summary.add_column("PF", justify="right")
    summary.add_column("Max DD (R)", justify="right")

    total_trades = 0
    total_won = 0
    total_r = 0.0
    all_r_values: List[float] = []

    for r in results:
        style = "green" if r.total_r > 0 else "red"
        summary.add_row(
            r.symbol,
            str(r.trades_filled),
            f"{r.win_rate:.1%}",
            f"[{style}]{r.total_r:+.2f}R[/{style}]",
            f"{r.avg_r:+.3f}R",
            f"{r.profit_factor:.2f}",
            f"{r.max_drawdown_r:.2f}R",
        )
        total_trades += r.trades_filled
        total_won += r.trades_won
        total_r += r.total_r
        all_r_values.extend(t.r_multiple for t in r.trades)

    console.print(summary)

    if total_trades > 0:
        console.print()
        console.print(f"[bold]Portfolio Totals:[/bold]")
        console.print(f"  Total Trades: {total_trades}")
        console.print(f"  Overall Win Rate: {total_won / total_trades:.1%}")
        r_style = "green" if total_r > 0 else "red"
        console.print(f"  Total R: [{r_style}]{total_r:+.2f}R[/{r_style}]")
        console.print(f"  Avg R per Trade: {np.mean(all_r_values):+.3f}R")

        gross_p = sum(r for r in all_r_values if r > 0)
        gross_l = abs(sum(r for r in all_r_values if r < 0))
        pf = gross_p / gross_l if gross_l > 0 else float('inf') if gross_p > 0 else 0.0
        console.print(f"  Portfolio Profit Factor: {pf:.2f}")
    console.print()


# ---------------------------------------------------------------------------
# Backtest output (JSON, text, ChatGPT prompt)
# ---------------------------------------------------------------------------

def build_backtest_payload(
    results: List[BacktestResult],
    side_mode: str,
    interval: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Build structured JSON payload from backtest results."""
    # Aggregate stats
    total_trades = sum(r.trades_filled for r in results)
    total_won = sum(r.trades_won for r in results)
    total_r = sum(r.total_r for r in results)
    all_r_values = [t.r_multiple for r in results for t in r.trades]

    gross_profit = sum(r for r in all_r_values if r > 0)
    gross_loss = abs(sum(r for r in all_r_values if r < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

    # Setup type breakdown
    setup_stats: Dict[str, Dict[str, Any]] = {}
    for result in results:
        for trade in result.trades:
            st = trade.setup_type
            if st not in setup_stats:
                setup_stats[st] = {"trades": 0, "wins": 0, "total_r": 0.0, "r_values": []}
            setup_stats[st]["trades"] += 1
            setup_stats[st]["r_values"].append(trade.r_multiple)
            setup_stats[st]["total_r"] += trade.r_multiple
            if trade.r_multiple > 0.05:
                setup_stats[st]["wins"] += 1

    setup_breakdown = []
    for st, data in sorted(setup_stats.items()):
        setup_breakdown.append({
            "setup_type": st,
            "trades": data["trades"],
            "win_rate": data["wins"] / data["trades"] if data["trades"] > 0 else 0,
            "avg_r": np.mean(data["r_values"]) if data["r_values"] else 0,
            "total_r": data["total_r"],
        })

    # Per-symbol results
    symbol_results = []
    for r in results:
        symbol_results.append({
            "symbol": r.symbol,
            "trades_filled": r.trades_filled,
            "trades_won": r.trades_won,
            "trades_lost": r.trades_lost,
            "win_rate": r.win_rate,
            "avg_r": r.avg_r,
            "total_r": r.total_r,
            "profit_factor": r.profit_factor if r.profit_factor != float('inf') else 999.99,
            "max_drawdown_r": r.max_drawdown_r,
            "trade_log": [
                {
                    "setup_type": t.setup_type,
                    "entry_zone": f"{t.entry_low:.6g}-{t.entry_high:.6g}",
                    "fill_price": t.fill_price,
                    "stop_loss": t.stop_loss,
                    "take_profits": t.take_profits,
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason,
                    "pnl_pct": round(t.pnl_pct, 3),
                    "r_multiple": round(t.r_multiple, 3),
                    "bars_held": t.exit_index - t.fill_index if t.fill_index >= 0 else 0,
                }
                for t in r.trades
            ]
        })

    return {
        "mode": "BACKTEST",
        "timestamp_utc": utc_now_iso(),
        "parameters": {
            "side_mode": side_mode,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "symbols_count": len(results),
        },
        "portfolio_summary": {
            "total_trades": total_trades,
            "total_won": total_won,
            "total_lost": total_trades - total_won,
            "overall_win_rate": total_won / total_trades if total_trades > 0 else 0,
            "total_r": round(total_r, 3),
            "avg_r_per_trade": round(np.mean(all_r_values), 3) if all_r_values else 0,
            "profit_factor": round(pf, 2) if pf != float('inf') else 999.99,
        },
        "setup_breakdown": setup_breakdown,
        "symbol_results": symbol_results,
    }


def build_backtest_text(
    results: List[BacktestResult],
    side_mode: str,
    interval: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """Build human-readable text summary of backtest results."""
    lines = []
    lines.append("=" * 70)
    lines.append("WHALETRADERBOT BACKTEST RESULTS")
    lines.append("=" * 70)
    lines.append("")

    # Parameters
    date_range = f"{start_date or 'recent'} to {end_date or 'now'}"
    lines.append(f"Side: {side_mode} | Interval: {interval} | Date Range: {date_range}")
    lines.append(f"Symbols Tested: {len(results)}")
    lines.append("")

    # Portfolio summary
    total_trades = sum(r.trades_filled for r in results)
    total_won = sum(r.trades_won for r in results)
    total_r = sum(r.total_r for r in results)
    all_r_values = [t.r_multiple for r in results for t in r.trades]

    gross_profit = sum(r for r in all_r_values if r > 0)
    gross_loss = abs(sum(r for r in all_r_values if r < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

    lines.append("-" * 70)
    lines.append("PORTFOLIO SUMMARY")
    lines.append("-" * 70)
    lines.append(f"Total Trades: {total_trades}")
    lines.append(f"Win/Loss: {total_won}/{total_trades - total_won}")
    lines.append(f"Win Rate: {total_won / total_trades:.1%}" if total_trades > 0 else "Win Rate: N/A")
    lines.append(f"Total R: {total_r:+.2f}R")
    lines.append(f"Avg R per Trade: {np.mean(all_r_values):+.3f}R" if all_r_values else "Avg R: N/A")
    lines.append(f"Profit Factor: {pf:.2f}" if pf != float('inf') else "Profit Factor: ∞")
    lines.append("")

    # Per-symbol results
    lines.append("-" * 70)
    lines.append("RESULTS BY SYMBOL")
    lines.append("-" * 70)

    for r in sorted(results, key=lambda x: x.total_r, reverse=True):
        status = "✓" if r.total_r > 0 else "✗"
        lines.append(f"{status} {r.symbol}: {r.trades_filled} trades | "
                    f"WR {r.win_rate:.0%} | {r.total_r:+.2f}R | PF {r.profit_factor:.2f}")

    lines.append("")

    # Setup type breakdown
    setup_stats: Dict[str, List[float]] = {}
    for result in results:
        for trade in result.trades:
            setup_stats.setdefault(trade.setup_type, []).append(trade.r_multiple)

    lines.append("-" * 70)
    lines.append("PERFORMANCE BY SETUP TYPE")
    lines.append("-" * 70)

    for st, rs in sorted(setup_stats.items(), key=lambda x: sum(x[1]), reverse=True):
        wins = sum(1 for r in rs if r > 0.05)
        wr = wins / len(rs) if rs else 0
        lines.append(f"{st}: {len(rs)} trades | WR {wr:.0%} | Avg {np.mean(rs):+.3f}R | Total {sum(rs):+.2f}R")

    lines.append("")

    # Top 5 best/worst trades
    all_trades = [(r.symbol, t) for r in results for t in r.trades]
    all_trades_sorted = sorted(all_trades, key=lambda x: x[1].r_multiple, reverse=True)

    lines.append("-" * 70)
    lines.append("TOP 5 BEST TRADES")
    lines.append("-" * 70)
    for sym, t in all_trades_sorted[:5]:
        lines.append(f"{sym} {t.setup_type}: {t.r_multiple:+.2f}R ({t.exit_reason})")

    lines.append("")
    lines.append("-" * 70)
    lines.append("TOP 5 WORST TRADES")
    lines.append("-" * 70)
    for sym, t in all_trades_sorted[-5:]:
        lines.append(f"{sym} {t.setup_type}: {t.r_multiple:+.2f}R ({t.exit_reason})")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def build_backtest_chatgpt_prompt(
    results: List[BacktestResult],
    side_mode: str,
    interval: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """Build ChatGPT prompt for backtest analysis."""
    lines = []
    lines.append("=" * 70)
    lines.append("WHALETRADERBOT BACKTEST ANALYSIS REQUEST")
    lines.append("=" * 70)
    lines.append("")
    lines.append("You are a quantitative trading analyst reviewing backtest results from WhaleTraderBot,")
    lines.append("a deterministic crypto futures signal generator. Analyze the following backtest data")
    lines.append("and provide actionable insights.")
    lines.append("")

    # Parameters
    date_range = f"{start_date or 'recent bars'} to {end_date or 'now'}"
    lines.append("## BACKTEST PARAMETERS")
    lines.append(f"- Side: {side_mode}")
    lines.append(f"- Interval: {interval}")
    lines.append(f"- Date Range: {date_range}")
    lines.append(f"- Symbols Tested: {len(results)}")
    lines.append("")

    # Portfolio summary
    total_trades = sum(r.trades_filled for r in results)
    total_won = sum(r.trades_won for r in results)
    total_r = sum(r.total_r for r in results)
    all_r_values = [t.r_multiple for r in results for t in r.trades]

    gross_profit = sum(r for r in all_r_values if r > 0)
    gross_loss = abs(sum(r for r in all_r_values if r < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

    lines.append("## PORTFOLIO SUMMARY")
    lines.append(f"- Total Trades: {total_trades}")
    lines.append(f"- Wins/Losses: {total_won}/{total_trades - total_won}")
    lines.append(f"- Win Rate: {total_won / total_trades:.1%}" if total_trades > 0 else "- Win Rate: N/A")
    lines.append(f"- Total R-Multiple: {total_r:+.2f}R")
    lines.append(f"- Average R per Trade: {np.mean(all_r_values):+.3f}R" if all_r_values else "- Avg R: N/A")
    lines.append(f"- Profit Factor: {pf:.2f}" if pf != float('inf') else "- Profit Factor: ∞")
    lines.append("")

    # Per-symbol breakdown
    lines.append("## RESULTS BY SYMBOL")
    lines.append("")
    lines.append("| Symbol | Trades | Win Rate | Total R | Avg R | PF | Max DD |")
    lines.append("|--------|--------|----------|---------|-------|-----|--------|")

    for r in sorted(results, key=lambda x: x.total_r, reverse=True):
        pf_display = f"{r.profit_factor:.2f}" if r.profit_factor != float('inf') else "∞"
        lines.append(f"| {r.symbol} | {r.trades_filled} | {r.win_rate:.0%} | "
                    f"{r.total_r:+.2f}R | {r.avg_r:+.3f}R | {pf_display} | {r.max_drawdown_r:.2f}R |")

    lines.append("")

    # Setup type breakdown
    setup_stats: Dict[str, Dict[str, Any]] = {}
    for result in results:
        for trade in result.trades:
            st = trade.setup_type
            if st not in setup_stats:
                setup_stats[st] = {"trades": 0, "wins": 0, "total_r": 0.0, "r_values": []}
            setup_stats[st]["trades"] += 1
            setup_stats[st]["r_values"].append(trade.r_multiple)
            setup_stats[st]["total_r"] += trade.r_multiple
            if trade.r_multiple > 0.05:
                setup_stats[st]["wins"] += 1

    lines.append("## PERFORMANCE BY SETUP TYPE")
    lines.append("")
    lines.append("| Setup Type | Trades | Win Rate | Avg R | Total R |")
    lines.append("|------------|--------|----------|-------|---------|")

    for st, data in sorted(setup_stats.items(), key=lambda x: x[1]["total_r"], reverse=True):
        wr = data["wins"] / data["trades"] if data["trades"] > 0 else 0
        avg_r = np.mean(data["r_values"]) if data["r_values"] else 0
        lines.append(f"| {st} | {data['trades']} | {wr:.0%} | {avg_r:+.3f}R | {data['total_r']:+.2f}R |")

    lines.append("")

    # Detailed trade log (top 10 best, top 10 worst)
    all_trades = [(r.symbol, t) for r in results for t in r.trades]
    all_trades_sorted = sorted(all_trades, key=lambda x: x[1].r_multiple, reverse=True)

    lines.append("## NOTABLE TRADES")
    lines.append("")
    lines.append("### Best 10 Trades")
    lines.append("| Symbol | Setup | Entry | Exit | Reason | R-Multiple |")
    lines.append("|--------|-------|-------|------|--------|------------|")

    for sym, t in all_trades_sorted[:10]:
        lines.append(f"| {sym} | {t.setup_type} | {t.fill_price:.6g} | {t.exit_price:.6g} | "
                    f"{t.exit_reason} | {t.r_multiple:+.2f}R |")

    lines.append("")
    lines.append("### Worst 10 Trades")
    lines.append("| Symbol | Setup | Entry | Exit | Reason | R-Multiple |")
    lines.append("|--------|-------|-------|------|--------|------------|")

    for sym, t in all_trades_sorted[-10:]:
        lines.append(f"| {sym} | {t.setup_type} | {t.fill_price:.6g} | {t.exit_price:.6g} | "
                    f"{t.exit_reason} | {t.r_multiple:+.2f}R |")

    lines.append("")

    # Analysis request
    lines.append("=" * 70)
    lines.append("## ANALYSIS REQUEST")
    lines.append("")
    lines.append("Please analyze this backtest data and provide:")
    lines.append("")
    lines.append("1. **Overall Assessment**: Is this strategy profitable? What is the risk-adjusted return?")
    lines.append("")
    lines.append("2. **Setup Type Analysis**: Which setup types perform best/worst? Should any be disabled?")
    lines.append("")
    lines.append("3. **Symbol Analysis**: Which symbols show consistent profitability? Which should be avoided?")
    lines.append("")
    lines.append("4. **Risk Analysis**: Comment on the max drawdown, win rate, and profit factor.")
    lines.append("")
    lines.append("5. **Recommendations**: Provide 3-5 specific, actionable recommendations to improve performance.")
    lines.append("")
    lines.append("6. **Red Flags**: Identify any concerning patterns (e.g., curve fitting, overly optimistic results).")
    lines.append("")
    lines.append("=" * 70)
    lines.append("NOTE: All entry/exit prices and setup classifications are deterministic (rule-based).")
    lines.append("The backtest does NOT account for slippage, fees, or liquidity gaps.")
    lines.append("=" * 70)

    return "\n".join(lines)


def save_backtest_outputs(
    results: List[BacktestResult],
    side_mode: str,
    interval: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    base_dir: str = "outputs",
) -> Dict[str, pathlib.Path]:
    """Save backtest outputs to files and return paths."""
    # Ensure output directories exist
    latest_dir = pathlib.Path(base_dir) / "latest"
    ensure_dir(latest_dir)

    # Build outputs
    payload = build_backtest_payload(results, side_mode, interval, start_date, end_date)
    text_summary = build_backtest_text(results, side_mode, interval, start_date, end_date)
    chatgpt_prompt = build_backtest_chatgpt_prompt(results, side_mode, interval, start_date, end_date)

    # Write files
    payload_path = latest_dir / "backtest_payload.json"
    text_path = latest_dir / "backtest_summary.txt"
    prompt_path = latest_dir / "backtest_chatgpt_prompt.txt"

    write_text(payload_path, json_dumps(payload))
    write_text(text_path, text_summary)
    write_text(prompt_path, chatgpt_prompt)

    console.print(f"[green]Backtest outputs saved to {latest_dir}/[/green]")
    console.print(f"  - backtest_payload.json")
    console.print(f"  - backtest_summary.txt")
    console.print(f"  - backtest_chatgpt_prompt.txt")

    return {
        "payload": payload_path,
        "summary": text_path,
        "prompt": prompt_path,
    }
