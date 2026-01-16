from __future__ import annotations

from typing import Any, Dict, List

from .utils import json_dumps


def build_plutus_overlay_prompt(side_mode: str, market_regime: Dict[str, Any], items: List[Dict[str, Any]]) -> str:
    """Prompt for local model to add psych + confirm checklist.

    IMPORTANT: We do NOT ask the model to invent entries/SL/TP.
    """
    return (
        "You are PLUTUS — a ruthless professional crypto futures trader and tape-reader.\n"
        "ROLE: Overlay ONLY. Do NOT change price levels from input.\n\n"
        "You receive:\n"
        "- BTC regime summary\n"
        "- A list of candidate ideas where ALGO has already computed entry_zone, stop_loss and take_profits.\n"
        "Your job: For EACH input item, output overlay fields ONLY:\n"
        "- rating_adjustment: integer in [-10, +10]\n"
        "- psych_score: 0-100\n"
        "- manipulation_flags: array of strings (e.g., stop_hunt, spoofing, squeeze_risk, none)\n"
        "- confirm_checklist: array of 2-4 short items (what MUST be seen before entry)\n"
        "- one_liner: <= 16 words\n\n"
        "STRICT RULES:\n"
        "1) OUTPUT JSON ONLY.\n"
        "2) Output must be an object: {\"overlays\": [...], \"notes\": [...]}\n"
        "3) overlays length MUST equal input length.\n"
        "4) Do not invent prices.\n\n"
        f"SIDE_MODE: {side_mode}\n"
        f"BTC_REGIME: {market_regime}\n"
        f"INPUT_ITEMS: {items}\n"
    )


def build_chatgpt_teamlead_prompt(payload: Dict[str, Any]) -> str:
    return (
        "You are GPT-5.2 (Team Lead) — senior crypto futures trader + risk manager.\n"
        "Goal: Choose 0–3 EXECUTE trades from WATCHLIST. Refine Entry/SL/TP and output Cornix-ready messages for EXECUTE.\n"
        "Rules: NO hallucination. If data is missing, mark WATCH/SKIP.\n\n"
        "IMPORTANT: ALGO provided deterministic Entry/SL/TP (always present).\n"
        "Local LLM overlay (if present) is advisory only (psych + confirm checklist).\n\n"
        "=== INPUT JSON (copy) ===\n"
        f"{json_dumps(payload, pretty=True)}\n"
    )
