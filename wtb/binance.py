from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import requests


# Cache for exchange info to avoid repeated API calls
_exchange_info_cache: Dict[str, Any] = {}
_exchange_info_cache_time: float = 0.0
_EXCHANGE_INFO_CACHE_TTL: float = 300.0  # 5 minutes


class BinanceFuturesClient:
    """Lightweight Binance USDT-M futures REST client (public endpoints only).

    Uses requests and supports insecure SSL when user's network injects a self-signed cert.
    """

    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        timeout_sec: int = 15,
        insecure_ssl: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.insecure_ssl = insecure_ssl
        self.session = requests.Session()

        if insecure_ssl:
            # Silence warnings for verify=False
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except Exception:
                pass

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, timeout=self.timeout_sec, verify=(not self.insecure_ssl))
        resp.raise_for_status()
        return resp.json()

    def ping(self) -> bool:
        try:
            self._get("/fapi/v1/ping")
            return True
        except Exception:
            return False

    def exchange_info(self) -> Dict[str, Any]:
        return self._get("/fapi/v1/exchangeInfo")

    def ticker_24hr(self) -> List[Dict[str, Any]]:
        return self._get("/fapi/v1/ticker/24hr")

    def book_ticker(self) -> List[Dict[str, Any]]:
        return self._get("/fapi/v1/ticker/bookTicker")

    def klines(self, symbol: str, interval: str, limit: int = 500, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> List[List[Any]]:
        params: Dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)
        return self._get("/fapi/v1/klines", params)

    def premium_index(self, symbol: Optional[str] = None) -> Any:
        params = {"symbol": symbol} if symbol else None
        return self._get("/fapi/v1/premiumIndex", params)

    def funding_rate(self, symbol: str, limit: int = 24) -> List[Dict[str, Any]]:
        return self._get("/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit})

    def open_interest(self, symbol: str) -> Dict[str, Any]:
        return self._get("/fapi/v1/openInterest", {"symbol": symbol})

    def open_interest_hist(self, symbol: str, period: str = "1h", limit: int = 6) -> List[Dict[str, Any]]:
        # /futures/data/openInterestHist uses same base domain
        return self._get("/futures/data/openInterestHist", {"symbol": symbol, "period": period, "limit": limit})

    def agg_trades(
        self,
        symbol: str,
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        # most recent aggTrades; optional time window in ms
        params: Dict[str, Any] = {"symbol": symbol, "limit": limit}
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        return self._get("/fapi/v1/aggTrades", params)

    def get_usdt_perpetual_symbols(self) -> Dict[str, Dict[str, Any]]:
        """Get all valid USDT-M perpetual symbols with metadata.

        Returns dict mapping symbol -> {symbol, baseAsset, quoteAsset, status, contractType}
        Only includes: contractType=PERPETUAL, quoteAsset=USDT, status=TRADING
        """
        global _exchange_info_cache, _exchange_info_cache_time

        now = time.time()
        if _exchange_info_cache and (now - _exchange_info_cache_time) < _EXCHANGE_INFO_CACHE_TTL:
            return _exchange_info_cache

        info = self.exchange_info()
        result: Dict[str, Dict[str, Any]] = {}

        for sym_info in info.get("symbols", []):
            if (
                sym_info.get("contractType") == "PERPETUAL"
                and sym_info.get("quoteAsset") == "USDT"
                and sym_info.get("status") == "TRADING"
            ):
                symbol = sym_info["symbol"]
                result[symbol] = {
                    "symbol": symbol,
                    "baseAsset": sym_info.get("baseAsset", ""),
                    "quoteAsset": sym_info.get("quoteAsset", ""),
                    "status": sym_info.get("status", ""),
                    "contractType": sym_info.get("contractType", ""),
                }

        _exchange_info_cache = result
        _exchange_info_cache_time = now
        return result


def normalize_symbol(raw: str) -> str:
    """Normalize user input to Binance symbol format.

    - Uppercase
    - Remove common separators (/, -, _)
    - Append USDT if missing

    Examples:
        "eth" -> "ETHUSDT"
        "btc/usdt" -> "BTCUSDT"
        "ETHUSDT" -> "ETHUSDT"
        "pepe" -> "PEPEUSDT"
    """
    s = raw.strip().upper()
    # Remove common separators
    for sep in ["/", "-", "_", " "]:
        s = s.replace(sep, "")
    # Append USDT if not present
    if not s.endswith("USDT"):
        s = s + "USDT"
    return s


def resolve_symbol(
    normalized: str,
    valid_symbols: Dict[str, Dict[str, Any]],
    volumes: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve a normalized symbol to an actual Binance symbol.

    Handles cases like:
    - PEPEUSDT vs 1000PEPEUSDT (prefers exact match, otherwise 1000x variant)
    - Direct matches

    Resolution is strict - only exact matches or 1000x variants are accepted.
    Partial string matches are NOT allowed to avoid false positives.

    Args:
        normalized: Normalized symbol (e.g., "PEPEUSDT")
        valid_symbols: Dict from get_usdt_perpetual_symbols()
        volumes: Optional dict mapping symbol -> 24h volume for disambiguation

    Returns:
        (resolved_symbol, warning_message) - warning is None if exact match
    """
    # Direct match
    if normalized in valid_symbols:
        return normalized, None

    # Try with 1000 prefix (common for meme coins like PEPE, SHIB, BONK, FLOKI)
    with_1000 = "1000" + normalized
    if with_1000 in valid_symbols:
        return with_1000, f"Resolved {normalized} -> {with_1000} (1000x variant)"

    # Try removing 1000 prefix if user included it but it doesn't exist
    if normalized.startswith("1000"):
        without_1000 = normalized[4:]
        if without_1000 in valid_symbols:
            return without_1000, f"Resolved {normalized} -> {without_1000}"

    # Extract base asset from normalized (remove USDT suffix)
    base = normalized[:-4] if normalized.endswith("USDT") else normalized

    # Only look for EXACT baseAsset matches (no partial string matching)
    candidates = []
    for sym, info in valid_symbols.items():
        sym_base = info.get("baseAsset", "")
        # Strict match: base asset must exactly match user input
        if sym_base == base:
            candidates.append(sym)

    if not candidates:
        return None, None

    if len(candidates) == 1:
        return candidates[0], f"Resolved {normalized} -> {candidates[0]}"

    # Multiple matches with same base asset - prefer by volume if available
    if volumes:
        candidates.sort(key=lambda x: volumes.get(x, 0.0), reverse=True)
        return candidates[0], f"Resolved {normalized} -> {candidates[0]} (highest volume among {candidates})"

    # Without volume data, prefer shorter symbol (usually the main one)
    candidates.sort(key=len)
    return candidates[0], f"Resolved {normalized} -> {candidates[0]} (multiple matches: {candidates})"


def validate_symbols(
    client: "BinanceFuturesClient",
    raw_symbols: List[str],
) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
    """Validate and resolve a list of user-provided symbols.

    Args:
        client: BinanceFuturesClient instance
        raw_symbols: List of raw user input (e.g., ["eth", "btc", "pepe"])

    Returns:
        (valid_symbols, warnings, errors)
        - valid_symbols: List of resolved Binance symbols
        - warnings: List of (original, warning_message) tuples
        - errors: List of symbols that couldn't be resolved
    """
    valid_perps = client.get_usdt_perpetual_symbols()

    # Get volumes for disambiguation
    volumes: Dict[str, float] = {}
    try:
        tickers = client.ticker_24hr()
        for t in tickers:
            sym = t.get("symbol", "")
            vol = float(t.get("quoteVolume", 0.0))
            volumes[sym] = vol
    except Exception:
        pass  # Volumes are optional for disambiguation

    valid: List[str] = []
    warnings: List[Tuple[str, str]] = []
    errors: List[str] = []

    seen: set = set()

    for raw in raw_symbols:
        if not raw.strip():
            continue

        normalized = normalize_symbol(raw)

        # Skip duplicates
        if normalized in seen:
            continue

        resolved, warning = resolve_symbol(normalized, valid_perps, volumes)

        if resolved:
            if resolved not in seen:
                valid.append(resolved)
                seen.add(resolved)
                seen.add(normalized)
                if warning:
                    warnings.append((raw, warning))
        else:
            errors.append(raw)

    return valid, warnings, errors


def parse_symbols_input(symbols_str: str) -> List[str]:
    """Parse comma-separated symbols string.

    Args:
        symbols_str: e.g., "ETH, BTC, PEPE" or "ETHUSDT,BTCUSDT"

    Returns:
        List of raw symbol strings
    """
    return [s.strip() for s in symbols_str.split(",") if s.strip()]


def parse_symbols_file(file_path: str) -> List[str]:
    """Parse symbols from a file (one per line, # for comments).

    Args:
        file_path: Path to symbols file

    Returns:
        List of raw symbol strings
    """
    import pathlib

    p = pathlib.Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Symbols file not found: {file_path}")

    symbols = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue
        # Handle inline comments
        if "#" in line:
            line = line.split("#")[0].strip()
        if line:
            symbols.append(line)

    return symbols
