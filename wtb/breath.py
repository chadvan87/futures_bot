from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .binance import BinanceFuturesClient


@dataclass
class MarketBreath:
    state: str  # RISK_ON / RISK_OFF
    reasons: List[str]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, object]:
        return {"state": self.state, "reasons": list(self.reasons), "metrics": dict(self.metrics)}


def _rolling_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 10 or len(b) < 10:
        return 0.0
    a = a[-60:]
    b = b[-60:]
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def compute_market_breath(
    client: BinanceFuturesClient,
    universe: List[str],
    btc_symbol: str,
    timeframe: str,
    thresholds: Dict[str, float],
) -> MarketBreath:
    """Free 'market breath' gauge using futures proxies (no news).

    Signals (trigger >=2 => RISK_OFF):
    - aggregate OI drop
    - median corr compression to BTC
    - funding extremes
    - BTC volume share jump (proxy dominance)
    """

    reasons: List[str] = []
    metrics: Dict[str, float] = {}

    # 1) Aggregate OI change (current vs 1h hist)
    oi_now = 0.0
    oi_prev = 0.0
    for sym in universe:
        try:
            oi_now += float(client.open_interest(sym).get("openInterest", 0.0))
            hist = client.open_interest_hist(sym, period="1h", limit=2)
            if len(hist) >= 2:
                oi_prev += float(hist[-2].get("sumOpenInterest", hist[-2].get("openInterest", 0.0)))
        except Exception:
            continue
    if oi_prev > 0:
        oi_change = (oi_now - oi_prev) / oi_prev
    else:
        oi_change = 0.0
    metrics["agg_oi_change"] = float(oi_change)

    # 2) Aggregate funding (median)
    fundings = []
    for sym in universe:
        try:
            prem = client.premium_index(sym)
            fundings.append(float(prem.get("lastFundingRate", 0.0)))
        except Exception:
            continue
    median_funding = float(np.median(fundings)) if fundings else 0.0
    metrics["median_funding"] = median_funding

    # 3) Correlation compression (median corr of 4H returns vs BTC)
    # Use klines on small set to keep light
    try:
        btc_kl = client.klines(btc_symbol, interval=timeframe, limit=120)
        btc_close = np.array([float(x[4]) for x in btc_kl], dtype=float)
        btc_ret = np.diff(np.log(btc_close + 1e-12))
    except Exception:
        btc_ret = np.array([], dtype=float)

    corrs = []
    for sym in universe[: min(20, len(universe))]:
        if sym == btc_symbol:
            continue
        try:
            kl = client.klines(sym, interval=timeframe, limit=120)
            cl = np.array([float(x[4]) for x in kl], dtype=float)
            ret = np.diff(np.log(cl + 1e-12))
            c = _rolling_corr(ret, btc_ret)
            if c != 0.0:
                corrs.append(c)
        except Exception:
            continue
    median_corr = float(np.median(corrs)) if corrs else 0.0
    metrics["median_corr"] = median_corr

    # 4) BTC volume share proxy
    try:
        tickers = client.ticker_24hr()
        vol_map = {t["symbol"]: float(t.get("quoteVolume", 0.0)) for t in tickers if "symbol" in t}
        btc_vol = vol_map.get(btc_symbol, 0.0)
        total_vol = sum(vol_map.get(s, 0.0) for s in universe) or 1.0
        btc_share = btc_vol / total_vol
    except Exception:
        btc_share = 0.0
    metrics["btc_vol_share"] = float(btc_share)

    # Decision
    triggers = 0
    if oi_change <= thresholds.get("oi_drop_pct", -0.04):
        triggers += 1
        reasons.append(f"agg OI drop {oi_change*100:.2f}%")
    if median_corr >= thresholds.get("median_corr", 0.85):
        triggers += 1
        reasons.append(f"corr compression median {median_corr:.2f}")
    if abs(median_funding) >= thresholds.get("funding_abs", 0.0005):
        triggers += 1
        reasons.append(f"funding extreme median {median_funding:.5f}")
    btc_share_thr = float(
        thresholds.get("btc_vol_share", thresholds.get("btc_vol_share_jump", 0.35))
    )
    if btc_share >= btc_share_thr:
        triggers += 1
        reasons.append(f"BTC vol share high {btc_share*100:.1f}%")

    state = "RISK_OFF" if triggers >= 2 else "RISK_ON"
    if not reasons:
        reasons = ["no major stress signals"]

    return MarketBreath(state=state, reasons=reasons, metrics=metrics)
