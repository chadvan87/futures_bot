from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def ema(values: np.ndarray, period: int) -> np.ndarray:
    if len(values) == 0:
        return values
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(values, dtype=float)
    out[0] = float(values[0])
    for i in range(1, len(values)):
        out[i] = alpha * float(values[i]) + (1 - alpha) * out[i - 1]
    return out


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    if len(close) < 2:
        return np.zeros_like(close, dtype=float)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[high[0] - low[0]], tr])
    # Wilder smoothing
    out = np.zeros_like(close, dtype=float)
    out[0] = tr[0]
    for i in range(1, len(close)):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Simplified ADX (Wilder). Returns ADX series."""
    n = len(close)
    if n < period + 2:
        return np.zeros_like(close, dtype=float)

    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))

    # Wilder smoothing
    def wilder_smooth(x: np.ndarray, p: int) -> np.ndarray:
        out = np.zeros(n - 1, dtype=float)
        out[0] = x[0]
        for i in range(1, len(out)):
            out[i] = (out[i - 1] * (p - 1) + x[i]) / p
        return out

    tr_s = wilder_smooth(tr, period)
    plus_s = wilder_smooth(plus_dm, period)
    minus_s = wilder_smooth(minus_dm, period)

    # Avoid div by zero
    tr_s = np.where(tr_s == 0, 1e-12, tr_s)

    plus_di = 100 * (plus_s / tr_s)
    minus_di = 100 * (minus_s / tr_s)

    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-12)

    adx_s = np.zeros_like(dx)
    adx_s[0] = dx[0]
    for i in range(1, len(dx)):
        adx_s[i] = (adx_s[i - 1] * (period - 1) + dx[i]) / period

    # align length to close
    out = np.concatenate([[0.0], adx_s])
    return out


def returns(close: np.ndarray) -> np.ndarray:
    if len(close) < 2:
        return np.zeros_like(close, dtype=float)
    r = np.zeros_like(close, dtype=float)
    r[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-12)
    return r


def rolling_corr(a: np.ndarray, b: np.ndarray, window: int) -> float:
    if len(a) < window or len(b) < window:
        return 0.0
    x = a[-window:]
    y = b[-window:]
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
