# WhaleTraderBot v2.3

**Deterministic crypto-perps signal generator for Binance Futures (USDT-M perpetual contracts).**

WhaleTraderBot generates ranked trading watchlists using multi-layer algorithmic gating with optional AI psychology overlays. Every entry, stop-loss, and take-profit is a reproducible mathematical formula — no black box, no randomness.

**Core thesis:** Trade market inefficiency and imbalance — sweeps, forced liquidations, late-chaser traps — with strict invalidation + risk control measured in R-multiples.

---

## Table of Contents

- [Architecture](#architecture)
- [Scoring System](#scoring-system)
- [Setup Types](#setup-types)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Commands](#commands)
  - [Scan Mode](#scan-mode)
  - [Manual Mode](#manual-mode)
  - [Backtest](#backtest)
  - [Cache Klines](#cache-klines)
  - [DCA Discovery](#dca-discovery)
- [Configuration](#configuration)
- [Output Format](#output-format)
- [Understanding Entry Prices](#understanding-entry-prices)
- [Optional: Ollama Psychology Overlay](#optional-ollama-psychology-overlay)
- [Project Structure](#project-structure)
- [Risk Disclaimer](#risk-disclaimer)

---

## Architecture

```
                    Binance Futures API (public, no key required)
                              |
              +---------------+---------------+
              |               |               |
         Klines (4H)    Book Ticker     24H Tickers
              |               |               |
              v               v               v
     +--------+--------+  Spread %     Volume Filter
     |  Indicators      |
     |  EMA(50/200)     |
     |  ATR(14)         +-------> Prescore (0-100)
     |  ADX(14)         |              |
     +--------+---------+              v
              |                   Shortlist (top 40)
              v                        |
     +--------+---------+             |
     | Structure Layer   |             |
     | Range detection   |             v
     | Pivot detection   +-------> build_algo_plan()
     +-------------------+         |
                                   |  Entry Zone / SL / TP1-3
                                   |  Setup Type classification
                                   |  Late detection (ATR units)
                                   v
                          +--------+---------+
                          | Overlay Layers    |
                          | Derivatives       | Funding + OI scoring
                          | Orderflow         | Aggtrade delta
                          | Whales (HyperLiq) | BTC/ETH whale context
                          | Market Breath     | Corr + regime gauge
                          +--------+---------+
                                   |
                                   v
                          Final Score (0-100)
                          Weighted sum + penalties
                                   |
                                   v
                          Ranked Watchlist (top K)
                                   |
                        +----------+----------+
                        |          |          |
                  payload.json  watchlist.txt  chatgpt_prompt.txt
```

## Scoring System

Final score is a weighted sum of 6 component scores (each 0-100), plus penalties:

| Component       | Weight | Source                                      |
|-----------------|--------|---------------------------------------------|
| **Tradeability**| 35%    | Volume, spread, ATR%, ADX                   |
| **Setup Quality**| 25%   | Entry/SL/TP geometry, structure alignment    |
| **Derivatives** | 20%    | Funding rate, open interest changes          |
| **Orderflow**   | 10%    | Buy/sell aggressor delta (15m + 1h windows)  |
| **Context**     | 5%     | BTC regime + market breath alignment         |
| **Whales**      | 5%     | Hyperliquid whale net exposure (BTC/ETH)     |

**Penalties** (capped at -10 total):
- Late entry: -2R (price already past the entry zone)
- BTC headwind: -6 (LONG in BEARISH BTC or SHORT in BULLISH BTC)
- Whale conflict: -4 (trading against strong whale positioning)

## Setup Types

| Setup                 | Trigger                                              | Entry Logic                         |
|-----------------------|------------------------------------------------------|-------------------------------------|
| **TREND_PULLBACK**    | ADX >= 25, price trending                            | EMA(50) +/- 0.6 ATR               |
| **BREAKOUT_RETEST**   | Price breaks tight range, retests edge               | Range edge +/- 0.4 ATR            |
| **RANGE_SWEEP_RECLAIM**| Wick sweeps range edge, body closes back inside     | Range boundary + 0.25 * range_h   |
| **VOLATILITY_FADE**   | ATR% >= 8% or candle >= 2.5x ATR                   | Range extreme + 0.15 * range_h    |

All entries use **limit-order zones** (not market). The bot waits for price to come to the zone — no chasing.

---

## Installation

### Requirements

- Python 3.10+
- Internet connection (Binance public API, no API key needed)

### Setup

```bash
# Clone and enter directory
cd futures_bot

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configure

```bash
cp config.example.json config.json
```

Edit `config.json` to customize scan parameters, scoring weights, and whale addresses.

---

## Quick Start

```bash
# Scan for LONG setups (top 120 symbols by volume)
python whaletraderbot.py scan --side LONG

# Scan for SHORT setups
python whaletraderbot.py scan --side SHORT

# Shortcuts
./run long
./run short

# Analyze specific symbols
python whaletraderbot.py manual --symbols "ETH,BTC,SOL"

# Run backtest on BTCUSDT
python whaletraderbot.py backtest --symbols BTC --side LONG

# Multi-symbol backtest
python whaletraderbot.py backtest --symbols "BTC,ETH,SOL" --side LONG

# Backtest top 50 coins by volume
python whaletraderbot.py backtest --top 50 --side LONG

# Backtest with specific date range
python whaletraderbot.py backtest --symbols BTC --start-date 2024-01-01 --end-date 2024-12-31

# Save backtest results for ChatGPT analysis
python whaletraderbot.py backtest --top 20 --save-outputs --print-prompt

# DCA Discovery - find symbols suitable for DCA bot operation
./run dca --side LONG
./run dca --side SHORT
./run dca --side BOTH
```

---

## Commands

### Scan Mode

Full market scan: fetch top symbols by volume, filter, score, and rank.

```bash
python whaletraderbot.py scan --side LONG [--top 120] [--watchlist-k 10] [--min-score 60]
```

| Flag            | Default | Description                                  |
|-----------------|---------|----------------------------------------------|
| `--side`        | LONG    | Trade direction: LONG or SHORT               |
| `--top`         | 120     | Number of top-volume symbols to scan         |
| `--watchlist-k` | 10      | Max symbols in final watchlist               |
| `--min-score`   | 60      | Minimum score to qualify for watchlist        |
| `--print-prompt`| false   | Print ChatGPT audit prompt to stdout         |

### Manual Mode

Analyze specific symbols through the same full pipeline.

```bash
# Comma-separated symbols (auto-normalized: eth -> ETHUSDT, PEPE -> 1000PEPEUSDT)
python whaletraderbot.py manual --symbols "ETH,BTC,PEPE,SOL"

# From file (one symbol per line, # for comments)
python whaletraderbot.py manual --symbols-file watchlist.txt

# Override side
python whaletraderbot.py manual --symbols "SOL,DOGE" --side SHORT
```

**Symbol normalization:**
- `eth` -> `ETHUSDT`
- `BTC/USDT` -> `BTCUSDT`
- `PEPE` -> `1000PEPEUSDT` (auto-resolved via Binance API)

### Backtest

Walk-forward backtest that replays the bot's algo signals on historical klines.

```bash
# Single symbol (default: 500 bars of 4h = ~83 days)
python whaletraderbot.py backtest --symbols BTC --side LONG

# Multi-symbol portfolio backtest
python whaletraderbot.py backtest --symbols "BTC,ETH,SOL" --side LONG

# Backtest top N symbols by 24h volume
python whaletraderbot.py backtest --top 50 --side LONG
python whaletraderbot.py backtest --top 100 --side LONG --interval 1h

# Date range backtest (specify start and/or end dates)
python whaletraderbot.py backtest --symbols BTC --start-date 2024-01-01 --end-date 2024-12-31
python whaletraderbot.py backtest --top 20 --start-date "2024-06-01" --side LONG

# Custom parameters
python whaletraderbot.py backtest --symbols BTC --side LONG \
    --interval 4h \
    --bars 1000 \
    --cooldown 12 \
    --fill-window 12 \
    --timeout 48
```

| Flag            | Default | Description                                          |
|-----------------|---------|------------------------------------------------------|
| `--symbols`     | BTCUSDT | Comma-separated symbols to backtest                  |
| `--top`         | —       | Backtest top N symbols by 24h volume (e.g., --top 50)|
| `--side`        | LONG    | Trade direction                                      |
| `--interval`    | 4h      | Kline interval (1d, 4h, 1h, 15m)                    |
| `--start-date`  | —       | Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM)         |
| `--end-date`    | now     | End date (YYYY-MM-DD or YYYY-MM-DD HH:MM)           |
| `--bars`        | 500     | Total bars to download (if no date range specified)  |
| `--cooldown`    | 12      | Min bars between signals (12 bars on 4h = ~2 days)   |
| `--fill-window` | 12      | Bars to wait for price to reach entry zone           |
| `--timeout`     | 48      | Max bars to hold a trade before forced exit          |
| `--print-prompt`| false   | Print ChatGPT analysis prompt to stdout              |
| `--save-outputs`| false   | Save results to outputs/latest/                      |

**Symbol selection:**
- `--symbols "BTC,ETH,SOL"` — Specific symbols (comma-separated)
- `--top 100` — Auto-select top 100 symbols by 24h trading volume
- If neither specified, defaults to BTCUSDT

**Date range mode:**
- Use `--start-date` and `--end-date` to test on specific historical periods
- Dates can be YYYY-MM-DD or YYYY-MM-DD HH:MM (UTC)
- If only `--start-date` is given, backtest runs to current time
- If neither is given, uses `--bars` (default 500) most recent bars

**How the backtest works:**

1. Downloads historical klines from Binance (or paginated download for date range)
2. Detects BTC regime from 1D data (same as live pipeline)
3. Walks forward bar-by-bar, running `build_algo_plan()` at each signal point
4. Only takes signals with `status_hint == "OK"` and `score_setup >= 40`
5. Waits for price to enter the entry zone within `fill_window` bars
6. Monitors for SL hit, TP1/TP2/TP3 hit, or timeout
7. Reports: win rate, R-multiples, profit factor, equity curve, drawdown, trade log, and performance breakdown by setup type

**Output includes:**
- Per-trade log (setup type, fill/exit prices, PnL%, R-multiple, bars held)
- Aggregate stats (win rate, avg R, total R, profit factor, max drawdown)
- Breakdown by setup type
- Portfolio summary (when backtesting multiple symbols)

**Saving results for ChatGPT analysis:**

```bash
# Save outputs to files
python whaletraderbot.py backtest --top 20 --save-outputs

# Print ChatGPT prompt to stdout
python whaletraderbot.py backtest --symbols "BTC,ETH" --print-prompt

# Both save and print
python whaletraderbot.py backtest --top 50 --save-outputs --print-prompt
```

When `--save-outputs` is used, three files are created in `outputs/latest/`:
- `backtest_payload.json` — Full structured data (symbol results, trade logs, setup breakdown)
- `backtest_summary.txt` — Human-readable summary
- `backtest_chatgpt_prompt.txt` — Pre-formatted prompt for ChatGPT analysis

**Sample ChatGPT prompt includes:**
- Portfolio summary (total trades, win rate, profit factor)
- Per-symbol performance table
- Setup type breakdown
- Best/worst 10 trades
- Specific analysis questions for ChatGPT to answer

### Cache Klines

Download historical klines to Parquet for offline analysis.

```bash
python whaletraderbot.py cache BTCUSDT 4h 1609459200000 1640995200000 data/BTCUSDT_4h.parquet
```

Arguments: `symbol interval start_ms end_ms output_path`

### DCA Discovery

**New in v2.3:** Separate scanner mode focused on finding symbols suitable for operating a DCA (Dollar-Cost Averaging) futures bot.

This mode is designed for grid/DCA bot operators who want to:
- Identify coins with good mean-reversion characteristics
- Control risk with tiered liquidity filtering
- Explore smaller caps (EXPLORE tier) with strict guardrails

```bash
# Find LONG DCA candidates
./run dca --side LONG

# Find SHORT DCA candidates
./run dca --side SHORT

# Evaluate both sides, return best per symbol
./run dca --side BOTH

# With overrides
./run dca --side LONG --top 200 --watchlist-k 20 --min-score 50
```

| Flag            | Default | Description                                          |
|-----------------|---------|------------------------------------------------------|
| `--side`        | LONG    | Side mode: LONG, SHORT, or BOTH                      |
| `--top`         | 180     | Number of top-volume symbols to evaluate             |
| `--watchlist-k` | 15      | Max symbols in final watchlist                       |
| `--min-score`   | 55      | Minimum DCA score to qualify                         |
| `--print-prompt`| false   | Print ChatGPT audit prompt to stdout                 |

**DCA Scoring Model (5 components, 0-100 each):**

| Component            | Weight | What It Measures                                    |
|----------------------|--------|-----------------------------------------------------|
| **Microstructure**   | 30%    | Spread quality, volume persistence                  |
| **Mean Reversion**   | 25%    | Wick reclaim tendency, EMA bounce frequency         |
| **Volatility Fit**   | 20%    | ATR% suitability (2-6% ideal for grids)            |
| **Derivatives Health**| 15%   | Funding neutrality, OI stability                    |
| **Context**          | 10%    | BTC regime compatibility                            |

**Penalties** (capped at -15 total):
- BTC headwind: -6 (LONG in BEARISH or SHORT in BULLISH)
- Extreme funding: -4 (|funding| > 0.1%)
- Liquidity stress: -4 (spread exceeds tier threshold)
- Trend runaway: -5 (ADX > 35 + ATR% > 6%)

**Tier System:**

| Tier      | Min Volume    | Max Spread | Description                          |
|-----------|---------------|------------|--------------------------------------|
| **CORE**  | $500M         | 0.08%      | Blue chips (BTC, ETH, SOL, etc.)    |
| **MID**   | $100M         | 0.15%      | Established alts                     |
| **EXPLORE**| $25M         | 0.30%      | Smaller caps (capped by quota)       |

The `explore_quota` config (default: 20%) caps how many EXPLORE tier symbols can appear in the final watchlist.

**Status Classification:**
- `RUN`: Score >= threshold, no hard blockers — safe to operate DCA bot
- `WATCH`: Borderline score or moderate risk flags — manual review needed
- `SKIP`: Fails hard gate or has critical risk factors

**Output Files (outputs/latest/):**
- `dca_payload.json` — Full structured analysis
- `dca_watchlist.txt` — Human-readable candidate list
- `dca_chatgpt_prompt.txt` — Audit prompt for second opinion

**Sample dca_payload.json structure:**

```json
{
  "mode": "DCA_DISCOVERY",
  "side_mode": "LONG",
  "candidates": [
    {
      "symbol": "ETHUSDT",
      "recommended_side": "LONG",
      "tier": "CORE",
      "dca_score": 72.4,
      "score_components": {
        "microstructure": 85.0,
        "mean_reversion": 68.5,
        "volatility_fit": 70.0,
        "derivatives_health": 65.0,
        "context": 75.0
      },
      "suggested_dca_profile": {
        "profile": "BALANCED",
        "grid_step_pct_hint": 1.2,
        "max_layers_hint": 10,
        "size_multiplier_hint": 1.5
      },
      "kill_switch_conditions": [
        "STOP if price < 2800 (current - 4*ATR)",
        "STOP if |funding| > 0.1% for 8+ hours"
      ],
      "status": "RUN"
    }
  ]
}
```

---

## Configuration

Configuration lives in `config.json` (falls back to built-in defaults if missing).

### Key sections:

```json
{
  "scan": {
    "side_default": "LONG",
    "scan_top": 120,
    "min_volume_usdt": 50000000,
    "max_spread_pct": 0.2,
    "watchlist_k": 10,
    "min_watch_score": 60
  },
  "indicators": {
    "ema_fast": 50,
    "ema_slow": 200,
    "atr_period": 14,
    "adx_period": 14,
    "range_lookback_bars": 48
  },
  "scoring": {
    "weights": {
      "tradeability": 35,
      "setup_quality": 25,
      "derivatives": 20,
      "orderflow": 10,
      "context": 5,
      "whales": 5
    },
    "late": {
      "ok_atr": 0.5,
      "watch_atr": 1.5
    }
  },
  "whales": {
    "enabled": true,
    "assets": ["BTC", "ETH"],
    "addresses": ["0x..."]
  },
  "plutus": {
    "enabled": false,
    "models": ["qwen2.5:14b"],
    "timeout_sec": 300
  },
  "dca": {
    "side_default": "LONG",
    "scan_top": 180,
    "watchlist_k": 15,
    "min_dca_score": 55,
    "explore_quota": 0.2,
    "tiers": {
      "core": { "min_volume_usdt": 500000000, "max_spread_pct": 0.08 },
      "mid": { "min_volume_usdt": 100000000, "max_spread_pct": 0.15 },
      "explore": { "min_volume_usdt": 25000000, "max_spread_pct": 0.30 }
    },
    "weights": {
      "microstructure": 30,
      "mean_reversion": 25,
      "volatility_fit": 20,
      "derivatives_health": 15,
      "context": 10
    },
    "penalties": {
      "btc_headwind": 6,
      "extreme_funding": 4,
      "liquidity_stress": 4,
      "trend_runaway": 5,
      "max_total": 15
    }
  }
}
```

---

## Output Format

All outputs are written to `outputs/latest/`:

### payload.json

Full structured analysis data:

```json
{
  "analysis_timestamp_utc": "2026-01-31 12:34:56",
  "side_mode": "LONG",
  "market_regime": {
    "btc_trend": "BULLISH",
    "close": 105000.0,
    "ema50": 102000.0,
    "ema200": 95000.0
  },
  "breath": { "state": "RISK_ON" },
  "watchlist": [
    {
      "symbol": "ETHUSDT",
      "side": "LONG",
      "setup_type": "TREND_PULLBACK",
      "entry_zone": "3200-3280",
      "stop_loss": 3120,
      "take_profits": [3350, 3450, 3600],
      "expected_r": 2.5,
      "final_score": 72.3,
      "late_status": "OK",
      "derivatives": { "score": 55, "funding_now": 0.00015 },
      "orderflow": { "score": 60, "delta_15m": 1000000 }
    }
  ]
}
```

### watchlist.txt

Human-readable watchlist for quick reference.

### chatgpt_prompt.txt

Pre-formatted prompt containing the full analysis — paste into ChatGPT for a second opinion / audit.

---

## Understanding Entry Prices

### Pullback Entry Strategy

The bot uses **limit-order entry zones**, not market orders. Entries are positioned to catch pullbacks:

- **LONG:** Entry zone is typically BELOW or at current price (buy the dip)
- **SHORT:** Entry zone is typically ABOVE or at current price (sell the bounce)

This is intentional — the bot does not chase. If price has already moved past the zone, it flags the signal as `WATCH_LATE` or `WATCH_PULLBACK`.

### Late Detection

| `late_atr` value | Status          | Meaning                                    |
|------------------|-----------------|--------------------------------------------|
| `0.0`            | OK              | Price hasn't reached entry yet             |
| `0.0 - 0.5`     | OK              | Slightly late, still acceptable            |
| `0.5 - 1.5`     | WATCH_LATE      | Late entry, consider reduced size          |
| `> 1.5`          | WATCH_PULLBACK  | Very late, wait for pullback to zone       |

### Distance Metrics

Each entry includes distance-to-entry info:

```json
{
  "distance_to_entry": {
    "percent": 2.1,
    "atr_units": 0.8,
    "direction": "BELOW"
  }
}
```

- `BELOW`: Price is below zone (typical for LONG — waiting for dip)
- `ABOVE`: Price is above zone (typical for SHORT — waiting for bounce)
- `INSIDE`: Price is in the zone (ready to enter)

---

## Optional: Ollama Psychology Overlay

The bot is **fully functional without Ollama**. The overlay adds:
- Psychological bias detection (FOMO, recency bias, etc.)
- Manipulation pattern flags (stop hunts, fake breakouts)
- Psychology score (0-100)
- Confirm checklist (2-4 items to verify before entry)

Ollama does **NOT** change algo numbers — entry/SL/TP are final from the algo.

### Setup

```bash
# Install Ollama
brew install ollama   # macOS
# or: https://ollama.ai/download

# Start server
ollama serve

# Pull model
ollama pull qwen2.5:14b
```

Enable in `config.json`:

```json
{
  "plutus": {
    "enabled": true,
    "base_url": "http://localhost:11434",
    "models": ["qwen2.5:14b"],
    "temperature": 0.2,
    "timeout_sec": 300,
    "max_candidates": 10
  }
}
```

---

## Project Structure

```
futures_bot/
├── whaletraderbot.py        # Entry point
├── config.json              # Active config
├── config.example.json      # Template config
├── requirements.txt         # Python dependencies
├── run                      # Bash shortcut (./run long, ./run short)
│
├── wtb/                     # Core package
│   ├── cli.py               # CLI argument parsing & command routing
│   ├── pipeline.py          # Scan mode pipeline (prescore -> algo -> overlays -> rank)
│   ├── manual.py            # Manual mode (user-specified symbols, same pipeline)
│   ├── dca.py               # DCA Discovery pipeline
│   ├── dca_scoring.py       # DCA scoring components and penalties
│   ├── algo.py              # Algo plan generation (entry zones, SL, TP1-3, R-multiples)
│   ├── backtest.py          # Walk-forward backtest engine + Parquet caching
│   ├── indicators.py        # EMA, ATR, ADX, correlation
│   ├── structure.py         # Range detection, pivot detection
│   ├── regime.py            # BTC regime detection (BULLISH / BEARISH / RANGE)
│   ├── breath.py            # Market breath gauge (correlation, OI, funding)
│   ├── derivatives.py       # Funding rate + open interest scoring
│   ├── orderflow.py         # Aggressor trade delta analysis
│   ├── whales.py            # Hyperliquid whale context (BTC/ETH)
│   ├── hyperliquid.py       # Hyperliquid API client
│   ├── plutus.py            # Ollama LLM psychology overlay
│   ├── prompts.py           # ChatGPT audit prompt builder
│   ├── binance.py           # Binance Futures REST client (public endpoints)
│   ├── config.py            # Config loading & defaults
│   └── utils.py             # JSON/text utilities
│
├── outputs/latest/          # Latest run outputs
├── docs/                    # Audit logs & investigation reports
└── prompts/                 # Pre-written LLM prompts
```

---

## Risk Disclaimer

**This software is for educational and research purposes only.** It does not constitute financial advice. Cryptocurrency futures trading involves substantial risk of loss. You could lose more than your initial investment.

- Past backtest performance does not guarantee future results
- The bot generates signals, not execution — always review before trading
- Use proper position sizing (risk 1-2% per trade max)
- Never trade with money you cannot afford to lose
- Backtest results do not account for slippage, fees, or liquidity gaps

**Use at your own risk.** The authors are not responsible for any financial losses incurred through use of this software.
