from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from .utils import extract_json_from_text


@dataclass
class PlutusMeta:
    ok: bool
    error: Optional[str]
    attempts: int
    model_used: Optional[str]
    models_tried: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "ok": self.ok,
            "error": self.error,
            "attempts": self.attempts,
            "model_used": self.model_used,
            "models_tried": list(self.models_tried),
        }


class OllamaClient:
    """Ollama API client for JSON generation.

    FIXED: Now accepts and uses a requests.Session for consistent HTTP handling.
    """
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout_sec: int = 120,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec
        self.session = session if session is not None else requests.Session()

    def generate_json(self, model: str, prompt: str, temperature: float = 0.2) -> Tuple[Dict[str, Any], str]:
        """Generate JSON response from Ollama.

        FIXED: Robust JSON parsing pipeline:
          1. Try json.loads(raw_text) first (Ollama format='json' often returns valid JSON)
          2. Fallback to extract_json_from_text if that fails
          3. Raise ValueError with diagnostic if both fail
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature
            }
        }
        # FIXED: Use self.session.post instead of requests.post directly
        r = self.session.post(url, json=payload, timeout=self.timeout_sec)
        r.raise_for_status()
        data = r.json()
        raw_text = data.get("response", "")

        # FIXED: Robust parsing pipeline
        parsed: Optional[Dict[str, Any]] = None

        # Try 1: Direct JSON parse (Ollama format='json' usually returns valid JSON)
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                return parsed, raw_text
        except (json.JSONDecodeError, ValueError):
            pass

        # Try 2: Fallback to extract_json_from_text (handles markdown, etc.)
        try:
            parsed = extract_json_from_text(raw_text)
            if isinstance(parsed, dict):
                return parsed, raw_text
        except Exception:
            pass

        # Both failed - raise with diagnostic
        preview = raw_text[:200] if raw_text else "(empty response)"
        raise ValueError(f"Failed to parse JSON from Ollama response. Preview: {preview}")


def _compact_candidate(c: Dict[str, Any]) -> Dict[str, Any]:
    """Extract minimal fields from candidate for Ollama prompt to reduce payload size."""
    return {
        "symbol": c.get("symbol"),
        "side": c.get("side"),
        "setup_type": c.get("setup_type"),
        "entry_zone": c.get("entry_zone"),
        "stop_loss": c.get("stop_loss"),
        "take_profits": c.get("take_profits", [])[:2],  # Only first 2 TPs
        "flags": c.get("flags", [])[:5],  # Limit flags
        "score_tradeability": c.get("score_tradeability"),
        "score_setup": c.get("score_setup"),
        "late_status": c.get("late_status"),
    }


def plutus_overlay_prompt(side_mode: str, market: Dict[str, Any], candidates: List[Dict[str, Any]]) -> str:
    """Generate Ollama prompt for psychology overlay.

    FIXED: Aligned with algo-only philosophy:
      - Clearly states: DO NOT change algo numbers (entry/SL/TP)
      - Removed setup_label (already provided) and overlay_score_adjust (conflicts with algo-only)
      - Compact payload (only essential fields)
    """
    # Compact market context
    market_compact = {
        "btc_trend": market.get("market_regime", {}).get("btc_trend"),
        "btc_close": market.get("market_regime", {}).get("close"),
        "breath_state": market.get("breath", {}).get("state") if market.get("breath") else None,
    }

    # Compact candidates
    candidates_compact = [_compact_candidate(c) for c in candidates]

    return (
        "You are PLUTUS, elite crypto futures trader and market psychologist.\n"
        "You MUST output STRICT JSON ONLY. No markdown. No commentary.\n\n"
        "CRITICAL: You are adding PSYCHOLOGY OVERLAY to precomputed ALGO plans.\n"
        "DO NOT change entry_zone, stop_loss, or take_profits - these are FINAL from algo.\n"
        "Your job: add psych_score (0-100), identify biases/manipulation, create confirm_checklist.\n\n"
        "RULES:\n"
        "1) For EACH input candidate, return ONE object.\n"
        "2) confirm_checklist: 2-4 concise items (what to watch before entry).\n"
        "3) psych_score: 0-100 (0=extreme fear/FOMO, 50=neutral, 100=high conviction setup).\n"
        "4) biases: array of psychological biases detected (e.g., \"FOMO\", \"anchoring\", \"recency_bias\").\n"
        "5) manipulation_flags: array of market manipulation patterns (e.g., \"stop_hunt\", \"fake_breakout\").\n"
        "6) notes: brief 1-2 sentence commentary.\n\n"
        "OUTPUT JSON schema (EXACTLY this structure):\n"
        "{\n"
        '  "items": [\n'
        '    {\n'
        '      "symbol": "BTCUSDT",\n'
        '      "psych_score": 75,\n'
        '      "biases": ["recency_bias"],\n'
        '      "manipulation_flags": [],\n'
        '      "confirm_checklist": ["Wait for 4H close above entry zone", "Check funding not extreme"],\n'
        '      "notes": "Clean pullback setup with low manipulation risk."\n'
        '    }\n'
        "  ]\n"
        "}\n\n"
        f"SIDE_MODE={side_mode}\n"
        f"MARKET={json.dumps(market_compact, ensure_ascii=False)}\n"
        f"CANDIDATES={json.dumps(candidates_compact, ensure_ascii=False)}\n"
    )


def _validate_overlay_item(item: Dict[str, Any]) -> bool:
    """Validate a single overlay item meets schema requirements.

    FIXED: Strict validation of overlay output.
    """
    # Required fields
    if "symbol" not in item:
        return False

    # psych_score must be 0-100
    psych_score = item.get("psych_score")
    if not isinstance(psych_score, (int, float)) or not (0 <= psych_score <= 100):
        return False

    # confirm_checklist must be list of 2-4 strings
    checklist = item.get("confirm_checklist")
    if not isinstance(checklist, list) or not (2 <= len(checklist) <= 4):
        return False
    if not all(isinstance(s, str) and len(s) > 0 for s in checklist):
        return False

    # biases and manipulation_flags must be lists (can be empty)
    if not isinstance(item.get("biases", []), list):
        return False
    if not isinstance(item.get("manipulation_flags", []), list):
        return False

    return True


def run_plutus_batch(
    cfg: Dict[str, Any],
    side_mode: str,
    market: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    insecure_ssl: bool = False,
) -> Tuple[Dict[str, Any], PlutusMeta]:
    """Run Ollama overlay on candidate trades.

    FIXED:
      - enabled defaults to False (was True)
      - When disabled: returns ok=False, error="disabled" (consistent failure state)
      - Session is created and passed to OllamaClient
      - require_nonempty config: fails if candidates non-empty but items empty
      - On total failure: meta.ok=False, meta.error set (was ok=True!)
      - Strict validation of each item
    """
    # FIXED: Default to disabled (was True)
    if not cfg.get("enabled", False):
        return {"items": []}, PlutusMeta(
            ok=False,
            error="disabled",
            attempts=0,
            model_used=None,
            models_tried=[]
        )

    base_url = cfg.get("base_url", "http://localhost:11434")
    timeout = int(cfg.get("timeout_sec", 120))
    models = list(cfg.get("models", []))
    temperature = float(cfg.get("temperature", 0.2))
    max_retries = int(cfg.get("max_retries", 2))
    require_nonempty = bool(cfg.get("require_nonempty", True))

    if not models:
        return {"items": []}, PlutusMeta(
            ok=False,
            error="no_models_configured",
            attempts=0,
            model_used=None,
            models_tried=[]
        )

    # FIXED: Create session and configure SSL (even for HTTP, for consistency)
    sess = requests.Session()
    sess.verify = not insecure_ssl

    # FIXED: Pass session to client
    client = OllamaClient(base_url=base_url, timeout_sec=timeout, session=sess)
    meta = PlutusMeta(ok=False, error=None, attempts=0, model_used=None, models_tried=[])

    prompt = plutus_overlay_prompt(side_mode, market, candidates)

    last_error: Optional[str] = None
    for model in models:
        meta.models_tried.append(model)
        for attempt in range(max_retries):
            meta.attempts += 1
            try:
                parsed, raw = client.generate_json(model=model, prompt=prompt, temperature=temperature)

                # Validate minimal structure
                items = parsed.get("items")
                if not isinstance(items, list):
                    raise ValueError("missing 'items' array in response")

                # FIXED: require_nonempty check
                if require_nonempty and len(candidates) > 0 and len(items) == 0:
                    raise ValueError(f"candidates non-empty ({len(candidates)}) but model returned 0 items")

                # FIXED: Strict validation of each item
                valid_items = []
                for i, item in enumerate(items):
                    if not isinstance(item, dict):
                        raise ValueError(f"item[{i}] is not a dict")
                    if not _validate_overlay_item(item):
                        raise ValueError(f"item[{i}] failed validation (symbol={item.get('symbol')})")
                    valid_items.append(item)

                # Success!
                meta.ok = True
                meta.error = None
                meta.model_used = model

                # Attach raw text if requested
                result = {"items": valid_items}
                if cfg.get("save_raw", False):
                    result["_raw"] = raw

                return result, meta

            except requests.exceptions.RequestException as e:
                last_error = f"HTTP error: {e}"
                # Connection errors: no point retrying same model, try next
                break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                # Other errors: retry same model
                continue

    # FIXED: Total failure - meta.ok must be False (was True!)
    meta.ok = False
    meta.error = f"all_models_failed: {last_error}" if last_error else "all_models_failed"
    meta.model_used = None
    return {"items": []}, meta
