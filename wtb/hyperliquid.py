from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


@dataclass
class HyperliquidClient:
    """Minimal Hyperliquid client for public `/info` endpoint.

    We only need `clearinghouseState` for a given address.

    Request (as commonly used in SDKs):
      POST {base_url}/info
      JSON: {"type": "clearinghouseState", "user": "0x..."}

    Note: Hyperliquid may change fields. We intentionally return raw JSON
    and parse defensively in `wtb.whales`.
    """

    base_url: str = "https://api.hyperliquid.xyz"
    timeout_s: int = 12

    def info(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self.base_url.rstrip("/") + "/info"
        r = requests.post(url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected Hyperliquid response type: {type(data)}")
        return data

    def clearinghouse_state(self, user: str) -> Dict[str, Any]:
        return self.info({"type": "clearinghouseState", "user": user})

    def user_state(self, user: str) -> Dict[str, Any]:
        # Some SDKs expose both. Kept for flexibility.
        return self.info({"type": "userState", "user": user})


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None
