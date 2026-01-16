#!/usr/bin/env python3
"""Test Ollama overlay integration.

This script tests the Ollama psychology overlay with a fake candidate.
It verifies that:
1. Ollama is running and accessible
2. The model is available
3. JSON parsing works correctly
4. Validation passes

Usage:
    python3 scripts/test_ollama_overlay.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wtb.config import load_config
from wtb.plutus import run_plutus_batch


def test_ollama_overlay():
    """Test Ollama overlay with a fake candidate."""
    print("=" * 60)
    print("Ollama Overlay Test")
    print("=" * 60)
    print()

    # Load config
    try:
        cfg = load_config("config.json")
    except FileNotFoundError:
        print("ERROR: config.json not found. Copy from config.example.json")
        return 1

    plutus_cfg = cfg.get("plutus", {})
    if not plutus_cfg.get("enabled", False):
        print("INFO: Ollama overlay is disabled in config.json")
        print("To enable, set plutus.enabled = true")
        print()
        print("Current config:")
        print(json.dumps(plutus_cfg, indent=2))
        return 0

    print("Ollama Config:")
    print(f"  Base URL: {plutus_cfg.get('base_url', 'http://localhost:11434')}")
    print(f"  Models: {plutus_cfg.get('models', [])}")
    print(f"  Temperature: {plutus_cfg.get('temperature', 0.2)}")
    print(f"  Max Retries: {plutus_cfg.get('max_retries', 2)}")
    print()

    # Create fake market context
    market = {
        "market_regime": {
            "btc_trend": "BULLISH",
            "close": 95000.0,
        },
        "breath": {
            "state": "RISK_ON"
        }
    }

    # Create fake candidate
    candidates = [
        {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "setup_type": "TREND_PULLBACK",
            "entry_zone": "94500-94800",
            "stop_loss": 94000.0,
            "take_profits": [95500.0, 96000.0, 96500.0],
            "flags": ["TRENDING", "GOOD_ATR"],
            "score_tradeability": 75.0,
            "score_setup": 68.0,
            "late_status": "OK",
        }
    ]

    print("Testing with fake candidate:")
    print(json.dumps(candidates[0], indent=2))
    print()

    print("Calling Ollama...")
    try:
        result, meta = run_plutus_batch(
            cfg=plutus_cfg,
            side_mode="LONG",
            market=market,
            candidates=candidates,
            insecure_ssl=False,
        )

        print()
        print("=" * 60)
        print("RESULT")
        print("=" * 60)
        print()

        print("Meta:")
        print(f"  OK: {meta.ok}")
        print(f"  Error: {meta.error}")
        print(f"  Attempts: {meta.attempts}")
        print(f"  Model Used: {meta.model_used}")
        print(f"  Models Tried: {meta.models_tried}")
        print()

        if meta.ok:
            print("✓ Overlay succeeded!")
            print()
            print("Items:")
            for i, item in enumerate(result.get("items", []), 1):
                print(f"\n[{i}] {item.get('symbol')}")
                print(f"  Psych Score: {item.get('psych_score')}")
                print(f"  Biases: {item.get('biases', [])}")
                print(f"  Manipulation Flags: {item.get('manipulation_flags', [])}")
                print(f"  Confirm Checklist:")
                for check in item.get("confirm_checklist", []):
                    print(f"    - {check}")
                print(f"  Notes: {item.get('notes', '(none)')}")

            print()
            print("=" * 60)
            print("TEST PASSED ✓")
            print("=" * 60)
            return 0
        else:
            print(f"✗ Overlay failed: {meta.error}")
            print()
            if "all_models_failed" in (meta.error or ""):
                print("TROUBLESHOOTING:")
                print("1. Check if Ollama is running:")
                print("   curl http://localhost:11434/api/tags")
                print()
                print("2. Check if model is available:")
                print("   ollama list")
                print()
                print("3. Pull model if needed:")
                print("   ollama pull qwen2.5:14b")
                print()
                print("4. Test generation:")
                base_url = plutus_cfg.get("base_url", "http://localhost:11434")
                model = plutus_cfg.get("models", ["qwen2.5:14b"])[0] if plutus_cfg.get("models") else "qwen2.5:14b"
                print(f'   curl {base_url}/api/generate -d \'{{"model":"{model}","prompt":"test","stream":false}}\'')

            print()
            print("=" * 60)
            print("TEST FAILED ✗")
            print("=" * 60)
            return 1

    except Exception as e:
        print()
        print("=" * 60)
        print(f"ERROR: {type(e).__name__}")
        print("=" * 60)
        print(str(e))
        print()

        print("TROUBLESHOOTING:")
        print("1. Check if Ollama is running:")
        print("   curl http://localhost:11434/api/tags")
        print()
        print("2. Install Ollama:")
        print("   https://ollama.ai/download")
        print()
        print("3. Pull a model:")
        print("   ollama pull qwen2.5:14b")

        print()
        print("=" * 60)
        print("TEST FAILED ✗")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(test_ollama_overlay())
