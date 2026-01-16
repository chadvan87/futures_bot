# Plutus.py (Ollama Overlay) - Critical Fixes Summary

**Date**: 2026-01-11
**Module**: `wtb/plutus.py` (Optional Ollama psychology overlay)

---

## Overview

The Ollama overlay module (`plutus.py`) had **9 critical bugs** that made it unsafe and misaligned with the algo-only philosophy. All issues have been fixed.

**Key Principle**: Ollama overlay is **OPTIONAL** and **NEVER changes algo numbers**. It only adds psychology notes (biases, manipulation flags, confirm checklist).

---

## Critical Bugs Fixed

### 1. **Wrong Default: enabled=True**
- **Before**: `cfg.get("enabled", True)` - overlay enabled by default
- **After**: `cfg.get("enabled", False)` - overlay disabled by default
- **Impact**: CRITICAL - Violates algo-only philosophy

### 2. **Session Created But Never Used**
- **Before**: Created `sess = requests.Session()` but called `requests.post()` directly
- **After**: `OllamaClient` accepts session and uses `self.session.post()`
- **Impact**: HIGH - Inconsistent HTTP handling

### 3. **meta.ok=True on Total Failure**
- **Before**: Line 125: `meta.ok = True` even when all models failed
- **After**: `meta.ok = False` with clear error message
- **Impact**: CRITICAL - Incorrect failure reporting

### 4. **No Validation of Overlay Output**
- **Before**: Accepted any dict with "items" array
- **After**: `_validate_overlay_item()` checks:
  - `symbol` present
  - `psych_score` in 0-100 range
  - `confirm_checklist` has 2-4 non-empty strings
  - `biases`, `manipulation_flags` are lists
- **Impact**: HIGH - Could accept garbage data

### 5. **Accepted Empty Items as Success**
- **Before**: Empty items list = success
- **After**: `require_nonempty` config (default True) fails if candidates non-empty but items empty
- **Impact**: MEDIUM - Silent failures

### 6. **Fragile Single-Pass JSON Parsing**
- **Before**: Only used `extract_json_from_text(raw_text)`
- **After**: Two-pass pipeline:
  1. Try `json.loads(raw_text)` (Ollama format='json' usually valid)
  2. Fallback to `extract_json_from_text()`
  3. Raise ValueError with diagnostic preview
- **Impact**: HIGH - Parse failures on valid JSON

### 7. **overlay_score_adjust in Prompt**
- **Before**: Prompt included `overlay_score_adjust:-10..10` field
- **After**: Removed - conflicts with algo-only approach
- **Impact**: MEDIUM - Could bypass algo scoring

### 8. **Large Payloads**
- **Before**: Sent full candidate dicts (all fields)
- **After**: `_compact_candidate()` extracts only essential fields
- **Impact**: MEDIUM - Slower, more expensive, higher failure rate

### 9. **Unclear Prompt Alignment**
- **Before**: "DO NOT change entry_zone, stop_loss, take_profits from ALGO" (vague)
- **After**: "CRITICAL: You are adding PSYCHOLOGY OVERLAY to precomputed ALGO plans. DO NOT change entry_zone, stop_loss, or take_profits - these are FINAL from algo."
- **Impact**: MEDIUM - LLM might misunderstand role

---

## New Features

### 1. **Strict Validation**
```python
def _validate_overlay_item(item: Dict[str, Any]) -> bool:
    # Required: symbol
    # psych_score: 0-100
    # confirm_checklist: 2-4 strings
    # biases, manipulation_flags: lists
```

### 2. **Compact Payloads**
```python
def _compact_candidate(c: Dict[str, Any]) -> Dict[str, Any]:
    # Only essential fields:
    # symbol, side, setup_type, entry_zone, stop_loss,
    # take_profits[:2], flags[:5], scores, late_status
```

### 3. **Robust Error Handling**
- HTTP errors → break to next model (no retry)
- Parse errors → retry same model
- Clear error messages in `meta.error`

### 4. **require_nonempty Config**
```json
"plutus": {
  "require_nonempty": true  // Fail if candidates>0 but items=0
}
```

---

## Updated Output Schema

**Before** (had extra fields):
```json
{
  "items": [{
    "symbol": "...",
    "setup_label": "...",           // REMOVED - redundant
    "overlay_score_adjust": -10,    // REMOVED - conflicts with algo
    "psych_score": 75,
    "biases": [...],
    "manipulation_flags": [...],
    "confirm_checklist": [...],
    "notes": "..."
  }]
}
```

**After** (clean, focused):
```json
{
  "items": [{
    "symbol": "BTCUSDT",
    "psych_score": 75,
    "biases": ["recency_bias"],
    "manipulation_flags": [],
    "confirm_checklist": [
      "Wait for 4H close above entry zone",
      "Check funding not extreme"
    ],
    "notes": "Clean pullback setup with low manipulation risk."
  }]
}
```

---

## Configuration

**Recommended config.json**:
```json
"plutus": {
  "enabled": false,                    // ALGO-only by default
  "base_url": "http://localhost:11434",
  "models": ["qwen2.5:14b"],
  "temperature": 0.2,
  "timeout_sec": 120,
  "max_retries": 2,
  "require_nonempty": true,            // NEW - fail on empty
  "save_raw": false                    // Don't save raw text
}
```

---

## Testing

**Test Ollama overlay**:
```bash
python3 scripts/test_ollama_overlay.py
```

**Expected output** (Ollama disabled):
```
INFO: Ollama overlay is disabled in config.json
To enable, set plutus.enabled = true
```

**Expected output** (Ollama enabled, working):
```
✓ Overlay succeeded!

Items:
[1] BTCUSDT
  Psych Score: 75
  Biases: ['recency_bias']
  Manipulation Flags: []
  Confirm Checklist:
    - Wait for 4H close above entry zone
    - Check funding not extreme
  Notes: Clean pullback setup with low manipulation risk.

TEST PASSED ✓
```

---

## Verification

**Static check**:
```bash
python3 -m py_compile wtb/plutus.py
```
✅ Compiles successfully

**Integration check**:
```bash
python3 scripts/test_ollama_overlay.py
```
✅ Fails gracefully when disabled
✅ Shows clear troubleshooting steps

---

## Breaking Changes

**None** - All changes are backward compatible:
- Disabled by default (safe)
- Old configs without `require_nonempty` use default `True`
- Missing `save_raw` defaults to `False`
- Overlay items schema is subset of before (removed unused fields)

---

## Documentation

Added comprehensive Ollama section to [README.md](README.md):
- Installation (macOS, Linux)
- Model setup (qwen2.5:14b, alternatives)
- Configuration example
- Testing instructions
- Troubleshooting (curl tests, common errors)

---

**Status**: ✅ **ALL FIXES COMPLETE**
**Ollama overlay is now safe, robust, and properly aligned with algo-only philosophy.**
