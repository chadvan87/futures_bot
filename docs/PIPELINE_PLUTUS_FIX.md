# Pipeline Plutus Integration Fix

**Date**: 2026-01-11
**Issue**: Ollama overlay never called even when `plutus.enabled=true`

---

## Problem

User enabled Ollama overlay in config.json:
```json
"plutus": {
  "enabled": true,
  ...
}
```

But running `./run scan --side LONG` always produced "ALGO_ONLY" output with no psychology overlay.

---

## Root Cause

In [wtb/pipeline.py](wtb/pipeline.py):

1. **Lines 296-305** hardcoded ALGO_ONLY behavior:
   ```python
   # Algo-only mode: no local LLM overlays
   for p in plans:
       p["psych_notes"] = {"psych_score": None, "biases": [], "manipulation_flags": [], "comment": "ALGO_ONLY"}
       p["confirm"] = _default_confirm(p)
   ```

2. **Line 97** had unused `enable_plutus: bool = False` parameter that was never read from config

3. **No import** of `run_plutus_batch` from plutus module

4. **No code path** to actually call the Ollama overlay when enabled

---

## Fix

### 1. Added import
```python
from .plutus import run_plutus_batch
```

### 2. Replaced hardcoded ALGO_ONLY with conditional logic

**Before** (lines 296-305):
```python
# Algo-only mode: no local LLM overlays
for p in plans:
    p["psych_notes"] = {"psych_score": None, "biases": [], "manipulation_flags": [], "comment": "ALGO_ONLY"}
    p["confirm"] = _default_confirm(p)
    p["overlay_flags"] = []
    p["overlay_score"] = 0.0
    final_score, score_detail = _score_breakdown(p, btc_regime, breath, cfg)
    p["final_score"] = float(final_score)
    p["score_detail"] = score_detail
```

**After** (lines 297-361):
```python
# Optional: Plutus (Ollama) psychology overlay
plutus_enabled = bool(cfg.get("plutus", {}).get("enabled", False))
if plutus_enabled:
    console.print("Running Ollama psychology overlay (Plutus)...")
    try:
        market_ctx = {
            "btc_regime": btc_regime.to_dict() if btc_regime else {},
            "breath": breath.to_dict() if breath else {},
            "whale_ctx": whale_ctx.to_dict() if whale_ctx else {},
        }
        overlay_result, overlay_meta = run_plutus_batch(
            cfg.get("plutus", {}),
            side_mode,
            market_ctx,
            candidates=plans,
            insecure_ssl=bool(cfg.get("binance", {}).get("insecure_ssl", False)),
        )

        if overlay_meta.ok:
            console.print(f"Ollama overlay succeeded: {len(overlay_result.get('items', []))} items")
            # Merge overlay results into plans
            overlay_by_symbol = {item["symbol"]: item for item in overlay_result.get("items", [])}
            for p in plans:
                overlay = overlay_by_symbol.get(p["symbol"])
                if overlay:
                    p["psych_notes"] = {
                        "psych_score": overlay.get("psych_score"),
                        "biases": overlay.get("biases", []),
                        "manipulation_flags": overlay.get("manipulation_flags", []),
                        "comment": overlay.get("notes", ""),
                    }
                    p["confirm"] = overlay.get("confirm_checklist", _default_confirm(p))
                else:
                    p["psych_notes"] = {"psych_score": None, "biases": [], "manipulation_flags": [], "comment": "NO_OVERLAY"}
                    p["confirm"] = _default_confirm(p)
                p["overlay_flags"] = []
                p["overlay_score"] = 0.0
        else:
            console.print(f"Ollama overlay failed: {overlay_meta.error} (tried {overlay_meta.models_tried})")
            # Fall back to ALGO_ONLY
            for p in plans:
                p["psych_notes"] = {"psych_score": None, "biases": [], "manipulation_flags": [], "comment": f"OVERLAY_FAILED: {overlay_meta.error}"}
                p["confirm"] = _default_confirm(p)
                p["overlay_flags"] = []
                p["overlay_score"] = 0.0
    except Exception as e:
        console.print(f"Ollama overlay error: {e}")
        for p in plans:
            p["psych_notes"] = {"psych_score": None, "biases": [], "manipulation_flags": [], "comment": f"OVERLAY_ERROR: {str(e)}"}
            p["confirm"] = _default_confirm(p)
            p["overlay_flags"] = []
            p["overlay_score"] = 0.0
else:
    # Algo-only mode: no local LLM overlays
    for p in plans:
        p["psych_notes"] = {"psych_score": None, "biases": [], "manipulation_flags": [], "comment": "ALGO_ONLY"}
        p["confirm"] = _default_confirm(p)
        p["overlay_flags"] = []
        p["overlay_score"] = 0.0

# Compute final scores (same for both algo-only and overlay modes)
for p in plans:
    final_score, score_detail = _score_breakdown(p, btc_regime, breath, cfg)
    p["final_score"] = float(final_score)
    p["score_detail"] = score_detail
```

---

## What Changed

### When `plutus.enabled = true`:
1. **Console message**: "Running Ollama psychology overlay (Plutus)..."
2. **Calls `run_plutus_batch()`** with market context (BTC regime, breath, whales)
3. **Merges overlay results** into plans:
   - `psych_score` (0-100)
   - `biases` (list of psychological biases)
   - `manipulation_flags` (list of manipulation patterns)
   - `confirm_checklist` (2-4 items to confirm before entry)
   - `notes` (LLM commentary)
4. **On success**: Shows count of overlay items
5. **On failure**: Falls back to ALGO_ONLY with clear error message

### When `plutus.enabled = false` (default):
- Same behavior as before: "ALGO_ONLY"
- No Ollama calls, no LLM overhead

---

## Testing

### 1. Verify Ollama is running
```bash
curl http://localhost:11434/api/tags
```

### 2. Run scan with overlay enabled
```bash
./run scan --side LONG
```

Expected console output:
```
Running Ollama psychology overlay (Plutus)...
Ollama overlay succeeded: 5 items
```

### 3. Check output files
```bash
cat outputs/latest/payload.json | jq '.watchlist[0].psych_notes'
```

Expected output (when overlay succeeds):
```json
{
  "psych_score": 75,
  "biases": ["recency_bias"],
  "manipulation_flags": [],
  "comment": "Clean pullback setup with low manipulation risk."
}
```

Expected output (when overlay disabled):
```json
{
  "psych_score": null,
  "biases": [],
  "manipulation_flags": [],
  "comment": "ALGO_ONLY"
}
```

---

## Impact

**CRITICAL** - This was a complete show-stopper for the Ollama overlay feature. Even with perfect configuration, the overlay was never called. Users who wanted psychology analysis couldn't use it at all.

Now fixed: users can enable/disable Ollama overlay via config, and it works as designed.

---

## Related Files

- [CHANGELOG.md](CHANGELOG.md) - Bug #14
- [PLUTUS_FIXES.md](PLUTUS_FIXES.md) - Complete Ollama overlay fixes
- [README.md](README.md) - Ollama setup documentation
- [scripts/test_ollama_overlay.py](scripts/test_ollama_overlay.py) - Test script
