from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests


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

    def klines(self, symbol: str, interval: str, limit: int = 500) -> List[List[Any]]:
        return self._get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit})

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
