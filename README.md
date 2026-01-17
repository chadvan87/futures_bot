# WhaleTraderBot v2.3 (ALGO-only)

WhaleTraderBot v2.3 is a **deterministic** crypto-perps signal generator (Binance Futures USDT-M).

Core idea: **trade market inefficiency / imbalance** (sweeps, forced moves, late traders) with strict invalidation + risk control.

This version is **ALGO-only** (no Plutus/Qwen/DeepSeek). It still generates a ready-to-paste **ChatGPT advisory prompt** if you want a final human/AI check.

## Features

- Multi-layer gating:
  - **Structure** (range edges, sweep+reclaim, breakout+retest, trend pullback)
  - **Volatility** (ATR% + late-at-entry in ATR units)
  - **Derivatives** (funding + OI)
  - **Orderflow** (delta / CVD slope)
  - **Breath / Regime** (BTC context)
  - **Whales (light)**: Hyperliquid whale addresses for **BTC/ETH context only** (scoring weight=5)
- Outputs:
  - Ranked WATCHLIST (0–10 best)
  - Cornix-ready messages for **EXECUTE** trades
  - `chatgpt_prompt.txt` to audit trades

## Quick start

### 1) Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure

Copy sample config and edit if needed:

```bash
cp config.example.json config.json
```

Your Hyperliquid whale addresses are already included in `config.example.json`.

### 3) Run a scan

```bash
./run scan --side LONG
```

Or:

```bash
python whaletraderbot.py scan --side LONG
```

Shortcut commands:

```bash
./run long    # Same as: scan --side LONG
./run short   # Same as: scan --side SHORT
```

## Manual Mode

Manual mode lets you analyze specific symbols instead of scanning the entire market. It runs the **exact same pipeline** as scan mode (BTC regime, breath, derivatives, orderflow, scoring) but on your chosen symbols.

### Usage

**Analyze specific symbols:**

```bash
# Using comma-separated list
python whaletraderbot.py manual --symbols "ETH,BTC,PEPE"

# Or with full symbol names
python whaletraderbot.py manual -s "ETHUSDT,BTCUSDT,1000PEPEUSDT"

# With side override
python whaletraderbot.py manual --symbols "SOL,DOGE" --side SHORT
```

**Using a symbols file:**

```bash
python whaletraderbot.py manual --symbols-file watchlist.txt
```

Example `watchlist.txt`:
```
# My watchlist
ETH
BTC
SOL
PEPE  # meme coin
DOGE
```

### Symbol Normalization

Manual mode automatically normalizes your input:

- **Uppercase**: `eth` → `ETHUSDT`
- **Add USDT suffix**: `BTC` → `BTCUSDT`
- **Remove separators**: `ETH/USDT` → `ETHUSDT`
- **1000x variants**: `PEPE` → `1000PEPEUSDT` (auto-resolved via Binance API)

### Validation

- Symbols are validated against Binance USDT-M perpetual contracts
- Invalid symbols are skipped with an error message
- Warnings are shown when symbols are auto-resolved (e.g., PEPE → 1000PEPEUSDT)

### Output

Manual mode generates the same outputs as scan mode in `outputs/latest/`:

- `payload.json` - Full analysis data (includes `mode: "manual"`)
- `watchlist.txt` - Human-readable watchlist
- `chatgpt_prompt.txt` - Ready for ChatGPT review

### Config

You can set default symbols in `config.json`:

```json
{
  "manual": {
    "enabled": true,
    "default_symbols": ["BTCUSDT", "ETHUSDT"]
  }
}
```

If no `--symbols` or `--symbols-file` is provided, these defaults are used.

## Outputs

After a scan, you get (default output dir: `outputs/latest/`):

- `outputs/latest/payload.json` – full JSON output
- `outputs/latest/chatgpt_prompt.txt` – paste into ChatGPT for final review
- `outputs/latest/watchlist.txt` – human-readable watchlist

Output directory can be configured in `config.json` under `outputs.base_dir`.

## Optional: Ollama Psychology Overlay

The bot is **fully functional without Ollama** (ALGO-only mode). Ollama adds an optional psychology overlay that:
- Identifies psychological biases (FOMO, recency bias, etc.)
- Detects manipulation patterns (stop hunts, fake breakouts)
- Adds a psychology score (0-100)
- Creates a confirm checklist (2-4 items to watch before entry)

**IMPORTANT**: Ollama does NOT change algo numbers (entry/SL/TP are final from algo).

### Setup Ollama

1. **Install Ollama**:
   - Download from https://ollama.ai/download
   - Or via package manager: `brew install ollama` (macOS), `apt install ollama` (Linux)

2. **Start Ollama** (if not auto-started):
   ```bash
   ollama serve
   ```

3. **Pull a model** (one-time):
   ```bash
   ollama pull qwen2.5:14b
   ```

   Alternative models:
   - `deepseek-r1:14b` - DeepSeek reasoning model
   - `llama3.1:8b` - Faster, lighter option

4. **Enable in config.json**:
   ```json
   "plutus": {
     "enabled": true,
     "base_url": "http://localhost:11434",
     "models": ["qwen2.5:14b"],
     "temperature": 0.2,
     "timeout_sec": 300,
     "max_retries": 2,
     "max_candidates": 10,
     "require_nonempty": true,
     "save_raw": false
   }
   ```

   **Config options**:
   - `timeout_sec`: Max time for LLM response (default: 300s / 5min)
   - `max_candidates`: Only analyze top N candidates after algo scoring (default: 10)
   - This prevents timeouts when scanning many symbols

5. **Test the overlay**:
   ```bash
   python3 scripts/test_ollama_overlay.py
   ```

### Expected Output

With Ollama enabled, each candidate will include `psych_notes`:

```json
{
  "symbol": "BTCUSDT",
  "psych_notes": {
    "psych_score": 75,
    "biases": ["recency_bias"],
    "manipulation_flags": [],
    "comment": "Clean pullback setup with low manipulation risk."
  },
  "confirm": [
    "Wait for 4H close above entry zone",
    "Check funding not extreme"
  ]
}
```

### Troubleshooting

**Test Ollama connection**:
```bash
curl http://localhost:11434/api/tags
```

**Check available models**:
```bash
ollama list
```

**Test generation**:
```bash
curl http://localhost:11434/api/generate -d '{"model":"qwen2.5:14b","prompt":"test","stream":false}'
```

**Common issues**:
- `Connection refused`: Ollama not running (`ollama serve`)
- `model not found`: Pull model first (`ollama pull qwen2.5:14b`)
- Empty items returned: Model may not understand prompt (try different model or increase temperature)

## Understanding Entry Prices

### Why Entries May Look "Far" from Current Price

The bot uses a **pullback entry strategy** for most setups. This means:

- **For SHORT setups:** Entry zone is positioned BELOW current price (waiting for price to rise into the zone)
- **For LONG setups:** Entry zone is positioned ABOVE current price (waiting for price to drop into the zone)

This is **intentional behavior**, not a bug!

### Price Reference

The bot uses the **last completed 4H candle close** as the "current price" reference:
- Appropriate for structure-based 4H execution timeframe
- Aligns with technical indicators (EMA, ATR, ADX) calculated on 4H data
- May differ from live ticker price by up to 4 hours

### Reading the Output

Starting from v2.3.1, each watchlist entry includes distance metrics:

```json
{
  "symbol": "XMRUSDT",
  "current_price": 477.73,
  "entry_zone": "474.2-488.3",
  "distance_to_entry": {
    "percent": 14.4,
    "atr_units": 5.3,
    "direction": "ABOVE"
  }
}
```

**Interpreting `direction`:**
- `ABOVE`: Current price is above entry zone (waiting for pullback DOWN into zone for SHORT)
- `BELOW`: Current price is below entry zone (waiting for pullback UP into zone for LONG)
- `INSIDE`: Current price is inside entry zone (ready to enter)

### Late Detection

The `late_atr` field measures if price has already moved PAST the entry:

- `late_atr = 0.0`: Price has NOT reached entry yet ✓
- `0.0 < late_atr <= 0.15`: Slightly late, still acceptable (status: `OK`)
- `0.15 < late_atr <= 0.25`: Late entry (status: `WATCH_LATE`)
- `late_atr > 0.25`: Very late, prefer waiting for pullback (status: `WATCH_PULLBACK`)

### Setup Types

**TREND_PULLBACK** (most common):
- Entry anchored around EMA(50) on 4H timeframe
- For SHORT: Entry below current price (wait for bounce to EMA50)
- For LONG: Entry above current price (wait for dip to EMA50)

**BREAKOUT_RETEST**:
- Entry near range edge after breakout
- Wait for price to retest the breakout level

**RANGE_SWEEP_RECLAIM**:
- Entry after wick sweeps range edge and reclaims
- Positioned at range boundary

**VOLATILITY_FADE**:
- Counter-trend entry during extreme volatility
- Entry near range extremes

For more details, see [INVESTIGATION_REPORT.md](INVESTIGATION_REPORT.md).

## Claude Code audit

There is a ready prompt you can paste into Claude Code to review the codebase:

- `CLAUDE_CODE_AUDIT_PROMPT.txt`

