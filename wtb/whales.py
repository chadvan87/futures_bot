from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .hyperliquid import HyperliquidClient, safe_float


@dataclass
class WhaleContext:
    timestamp_utc: str
    addresses: List[str]
    assets: List[str]
    by_asset: Dict[str, Dict[str, Any]]
    total_net_notional: float
    bullish_score: float
    state: str  # BULLISH|BEARISH|NEUTRAL
    flags: List[str]


def _parse_asset_positions(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract `assetPositions` list from a clearinghouseState response.

    Expected shape (observed in many SDKs):
      {
        "assetPositions": [
          {"position": {"coin": "BTC", "szi": "0.10", "entryPx": "...", "markPx": "...", ...}},
          ...
        ],
        ...
      }

    We parse defensively because fields can change.
    """
    ap = resp.get("assetPositions")
    if isinstance(ap, list):
        return ap
    # some SDKs return nested
    ap = resp.get("state", {}).get("assetPositions")
    if isinstance(ap, list):
        return ap
    return []


def _extract_position(p: Dict[str, Any]) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """Return (coin, szi, markPx).

    - szi: signed size (positive long, negative short)
    - markPx: best available mark/entry price for notional
    """
    pos = p.get("position") if isinstance(p.get("position"), dict) else p
    if not isinstance(pos, dict):
        return None, None, None

    coin = pos.get("coin") or pos.get("asset")
    szi = safe_float(pos.get("szi") or pos.get("size") or pos.get("position") or pos.get("sz"))

    mark = safe_float(pos.get("markPx"))
    if mark is None:
        mark = safe_float(pos.get("entryPx"))

    if isinstance(coin, str):
        coin = coin.upper()
    else:
        coin = None
    return coin, szi, mark


def _bullish_score_from_net_notional(net_notional: float, scale: float = 50_000_000.0) -> float:
    """Map net notional to a 0-100 bullishness score.

    +scale USD net long => ~90
    -scale USD net short => ~10
    """
    if scale <= 0:
        scale = 50_000_000.0
    x = max(-1.0, min(1.0, net_notional / scale))
    return 50.0 + 40.0 * x


def build_whale_context(
    cfg: Dict[str, Any],
    *,
    now_utc: str,
    client: Optional[HyperliquidClient] = None,
    _cache: Dict[str, Any] | None = None,
) -> WhaleContext:
    """Fetch Hyperliquid whales and compute a BTC/ETH context snapshot.

    This is intentionally *lite*: whales are used as a **pressure gauge**,
    not a primary entry signal.

    cfg example:
      {
        "enabled": true,
        "base_url": "https://api.hyperliquid.xyz",
        "timeout_s": 12,
        "assets": ["BTC","ETH"],
        "addresses": [...],
        "cache_ttl_s": 120,
        "net_scale_usd": 50000000
      }
    """
    if _cache is None:
        _cache = {}

    enabled = bool(cfg.get("enabled", True))
    if not enabled:
        return WhaleContext(
            timestamp_utc=now_utc,
            addresses=[],
            assets=["BTC", "ETH"],
            by_asset={},
            total_net_notional=0.0,
            bullish_score=50.0,
            state="NEUTRAL",
            flags=["WHALES_DISABLED"],
        )

    cache_ttl = int(cfg.get("cache_ttl_s", 120))
    cached = _cache.get("value")
    cached_ts = _cache.get("ts", 0)
    if cached is not None and (time.time() - cached_ts) < cache_ttl:
        return cached

    addresses = [a.strip() for a in cfg.get("addresses", []) if isinstance(a, str) and a.strip()]
    assets = [a.upper() for a in cfg.get("assets", ["BTC", "ETH"]) if isinstance(a, str) and a.strip()]

    base_url = str(cfg.get("base_url", "https://api.hyperliquid.xyz"))
    timeout_s = int(cfg.get("timeout_s", 12))
    net_scale = float(cfg.get("net_scale_usd", 50_000_000.0))

    if client is None:
        client = HyperliquidClient(base_url=base_url, timeout_s=timeout_s)

    by_asset: Dict[str, Dict[str, Any]] = {a: {"net_notional": 0.0, "gross_notional": 0.0, "n_pos": 0} for a in assets}
    flags: List[str] = []

    ok_count = 0
    err_count = 0

    for addr in addresses:
        try:
            resp = client.clearinghouse_state(addr)
            ap = _parse_asset_positions(resp)
            for item in ap:
                coin, szi, mark = _extract_position(item)
                if coin is None or szi is None or mark is None:
                    continue
                if coin not in by_asset:
                    continue
                notional = abs(szi) * mark
                by_asset[coin]["gross_notional"] += notional
                by_asset[coin]["net_notional"] += (1.0 if szi > 0 else -1.0) * notional
                by_asset[coin]["n_pos"] += 1
            ok_count += 1
        except Exception:
            err_count += 1

    total_net = sum(by_asset[a]["net_notional"] for a in assets)
    bull = _bullish_score_from_net_notional(total_net, scale=net_scale)

    if bull >= 60.0:
        state = "BULLISH"
    elif bull <= 40.0:
        state = "BEARISH"
    else:
        state = "NEUTRAL"

    if err_count > 0:
        flags.append("WHALES_PARTIAL")
    if ok_count == 0:
        flags.append("WHALES_NO_DATA")

    ctx = WhaleContext(
        timestamp_utc=now_utc,
        addresses=addresses,
        assets=assets,
        by_asset=by_asset,
        total_net_notional=float(total_net),
        bullish_score=float(bull),
        state=state,
        flags=flags,
    )

    _cache["value"] = ctx
    _cache["ts"] = time.time()
    return ctx


def whales_component_score(ctx: WhaleContext, *, side_mode: str, symbol: str) -> float:
    """Map WhaleContext to a per-trade 0-100 component score.

    For BTC/ETH symbols, we use that asset's net notional.
    For all alts, we use the combined BTC+ETH bullishness.

    LONG: higher bullish_score is better
    SHORT: lower bullish_score is better (mirror)
    """
    side = (side_mode or "").upper()
    sym = (symbol or "").upper()

    bull = ctx.bullish_score
    if sym.startswith("BTC") and "BTC" in ctx.by_asset:
        bull = _bullish_score_from_net_notional(float(ctx.by_asset["BTC"]["net_notional"]), scale=50_000_000.0)
    elif sym.startswith("ETH") and "ETH" in ctx.by_asset:
        bull = _bullish_score_from_net_notional(float(ctx.by_asset["ETH"]["net_notional"]), scale=50_000_000.0)

    if side == "SHORT":
        return float(100.0 - bull)
    return float(bull)
