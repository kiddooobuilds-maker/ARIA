# 🤖 AI Hedge Fund — Indian Equity Markets

A full-stack, AI-powered trading system for Indian equities (NSE/BSE). It combines an **LSTM neural network** for price prediction, **FinBERT/VADER** for news sentiment analysis, and a **Flask + React dashboard** with an AI chatbot (ARIA) for signal interpretation.

> Built as a major project. Not intended for live trading with real money.

---

## 📁 Project Structure

```
major/
├── ai_hedge_fund.py      # Core ML pipeline (LSTM + Sentiment + Signal + Backtester)
├── app.py                # Flask REST API server
├── index.html            # Frontend dashboard (React, single-file)
├── base.py               # Baseline runner — runs one ticker and prints metrics
├── run_test.py           # Quick smoke-test (2 epochs, fast validation)
├── sentiment_backtester.py # Standalone backtester for sentiment signals only
├── requirements.txt      # Python dependencies
└── cache/                # Auto-generated — saved models, sentiment CSVs, plots
```

---

## ✅ What's Working

| Feature | Status | Notes |
|---|---|---|
| LSTM price prediction | ✅ Working | Bidirectional, 3-layer, predicts daily returns |
| FinBERT sentiment | ✅ Working | Loads `ProsusAI/finbert` via HuggingFace |
| VADER fallback | ✅ Working | Used if FinBERT fails or is disabled |
| DuckDuckGo news | ✅ Working | Free, no API key needed |
| yfinance news | ✅ Working | Used as a secondary source with date filtering |
| RSS feeds (ET, Moneycontrol, etc.) | ✅ Working | Applied only to latest/today's date |
| Signal fusion (LSTM + Sentiment + Momentum) | ✅ Working | 50/30/20 weight split |
| Backtesting engine | ✅ Working | Stop-loss, trailing stop, take-profit, Sharpe, alpha vs Nifty 50 |
| Flask API server | ✅ Working | SSE progress streaming, job queue |
| Web dashboard | ✅ Working | Open `http://localhost:5000` after starting `app.py` |
| ARIA chatbot (rule-based) | ✅ Working | Answers signal/performance/risk questions without Ollama |
| ARIA chatbot (Ollama/LLaMA 3) | ⚠️ Optional | Only works if Ollama is running locally |
| Sentiment cache | ✅ Working | Incremental — only scores new trading days |
| Model cache | ✅ Working | Skips retraining if a saved model exists |
| Nifty 50 benchmark | ✅ Working | Alpha is calculated vs `^NSEI` |

---

## ⚠️ Known Limitations / What Isn't Perfect

- **Historical sentiment is set to neutral (0.0)** — news APIs don't provide archives older than ~30 days, so all historical trading days get a sentiment score of 0. Only the most recent ~30 days get real sentiment scores. This is a data availability limitation, not a bug.
- **DuckDuckGo search rate-limits** — if you run the pipeline many times in quick succession it may silently return 0 headlines for some days. The model degrades gracefully (falls back to neutral sentiment).
- **No live trading** — this is a backtesting/research system only. There is no broker integration.
- **Ollama is optional** — ARIA works without it via rule-based answers, but deep Q&A requires Ollama with LLaMA 3 running locally.
- **First run is slow** — FinBERT model (~500 MB) needs to download from HuggingFace on first use.
- **GPU optional** — everything runs on CPU. A GPU makes FinBERT and LSTM training faster.

---

## 🚀 Setup & Running

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Python 3.9+ recommended. Use a virtual environment (`.venv` is already in the folder).

### 2. Start the server

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

### 3. Quick smoke test (no UI)

```bash
python run_test.py
```

Runs RELIANCE.NS with only 2 training epochs — good for checking everything is wired up.

### 4. Baseline metrics (no UI)

```bash
python base.py
```

Prints Total Return, Sharpe, Alpha, Max Drawdown, Win Rate for RELIANCE.NS.

---

## 🧠 How the ML Pipeline Works

```
yfinance OHLCV
      │
      ▼
DataIngestor  →  15 features (OHLCV + SMA/EMA + RSI + MACD + BB + ATR + OBV)
      │
      ▼
SentimentAnalyser  →  DuckDuckGo + yfinance news → FinBERT/VADER → daily score
      │
      ▼
FeaturePipeline  →  Merge price + sentiment → MinMax scale → sliding windows (seq=90)
      │
      ▼
AIHedgeFundLSTM  →  Bidirectional LSTM (3 layers, 128 hidden) → predict daily return
      │
      ▼
SignalGenerator  →  Fuse LSTM (50%) + Sentiment (30%) + Momentum RSI/MACD (20%)
                    → STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL
      │
      ▼
Backtester  →  Simulate trades with stop-loss / trailing stop / take-profit
               → Sharpe, Alpha vs Nifty 50, Win Rate, Max Drawdown
```

---

## 📈 Supported Tickers

Any NSE ticker with `.NS` suffix (e.g. `RELIANCE.NS`, `INFY.NS`, `TCS.NS`). The project originally targeted a basket of **10 major Indian stocks**, and the curated keyword list was expanded to 20 tickers over the course of development.

**Original 10:**
`RELIANCE.NS` · `INFY.NS` · `TCS.NS` · `HDFCBANK.NS` · `ICICIBANK.NS` · `WIPRO.NS` · `SBIN.NS` · `ADANIENT.NS` · `TATAMOTORS.NS` · `BAJFINANCE.NS`

**Added later (now in `TICKER_MAP`):**
`HINDUNILVR.NS` · `ITC.NS` · `KOTAKBANK.NS` · `LT.NS` · `MARUTI.NS` · `BHARTIARTL.NS` · `AXISBANK.NS` · `NESTLEIND.NS` · `SUNPHARMA.NS` · `DRREDDY.NS`

---

## 🖥️ Dashboard Features

- **Run Analysis** — Enter any NSE ticker, watch real-time SSE progress logs
- **Charts** — Price vs predicted, training loss curve, backtest equity curve
- **Trade Log** — Every simulated buy/sell with P&L
- **Metrics Panel** — Total return, Sharpe, alpha vs Nifty, win rate, drawdown
- **ARIA Chatbot** — Ask questions like:
  - *"Why was this BUY signal given?"*
  - *"What's the Sharpe ratio?"*
  - *"What was the best trade?"*
  - *"Explain the sentiment score"*
  - *"What are the risks?"*

---

## ⚙️ Key Config Options (`Config` in `ai_hedge_fund.py`)

| Parameter | Default | Description |
|---|---|---|
| `ticker` | `RELIANCE.NS` | NSE ticker symbol |
| `start_date` | `2020-01-01` | Training data start |
| `seq_len` | `90` | LSTM lookback window (days) |
| `use_finbert` | `True` | Use FinBERT (disable to use VADER, much faster) |
| `epochs` | `100` | Max training epochs (early stopping at 15 patience) |
| `price_weight` | `0.50` | LSTM signal weight in fusion |
| `sentiment_weight` | `0.30` | Sentiment signal weight |
| `momentum_weight` | `0.20` | RSI+MACD signal weight |
| `stop_loss_pct` | `0.07` | Exit if down 7% from entry |
| `trailing_stop_pct` | `0.05` | Exit if down 5% from peak |
| `take_profit_pct` | `0.15` | Exit if up 15% from entry |
| `min_return_threshold` | `0.007` | Min 0.7% predicted return to call BUY |
| `force_retrain` | `False` | Set True to ignore cached model |

---

## 🗂️ Cache Behaviour

The `cache/` folder is auto-managed:

- `data_<TICKER>.csv` — full OHLCV + all 15 indicators, exported every run (ready to open in Excel)
- `model_<TICKER>.pt` — saved LSTM weights, reloaded automatically on next run
- `sentiment_<TICKER>.csv` — incremental sentiment cache, max 2000 rows per ticker
- `loss_<TICKER>.png` — training loss curve image
- `backtest_<TICKER>.png` — backtest equity curve image

To force a full fresh run: delete the cache folder or set `force_retrain=True`.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `torch` | LSTM training |
| `transformers` | FinBERT (HuggingFace) |
| `yfinance` | Stock price data + news |
| `duckduckgo-search` | Free news search |
| `vaderSentiment` | Lightweight sentiment fallback |
| `scikit-learn` | MinMaxScaler |
| `flask` + `flask-cors` | REST API server |
| `pandas` / `numpy` / `matplotlib` | Data processing and plotting |

---

## 💬 ARIA Chatbot — Advanced Mode (Ollama)

For open-ended questions beyond the rule-based engine, install [Ollama](https://ollama.com) and pull LLaMA 3:

```bash
ollama pull llama3
ollama serve
```

With Ollama running, ARIA can answer arbitrary finance questions grounded in the current analysis context.
