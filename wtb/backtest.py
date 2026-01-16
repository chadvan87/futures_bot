from __future__ import annotations

import pathlib
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .binance import BinanceFuturesClient


def cache_klines(client: BinanceFuturesClient, symbol: str, interval: str, start_ms: int, end_ms: int, out_path: pathlib.Path) -> pathlib.Path:
    """Download klines into a Parquet cache (offline backtest support)."""
    rows: List[List] = []
    cursor = start_ms
    step = 1500  # max limit
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
        "open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_base","taker_quote","ignore"
    ])
    if df.empty:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        return out_path
    for c in ["open","high","low","close","volume","quote_volume","taker_base","taker_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


# NOTE: A full backtest engine is intentionally kept as an add-on.
# The cached Parquet gives you reproducible historical data, and you can plug it into
# Backtrader/VectorBT/your own simulator without any extra API calls.
