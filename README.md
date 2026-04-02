# AI-Powered Quant Portfolio Management System

An engineering major project for Indian equity markets. This project combines an LSTM-led forecasting pipeline, technical indicators, news sentiment, backtesting, and a web dashboard for analysing NSE stocks and comparing decisions against real market benchmarks such as the Nifty 50.

This is a research and demonstration system, not a live-trading product.

## Project Overview

The system is designed to answer one practical question:

"Can we combine historical market sequences, technical indicators, and news sentiment to generate better trading decisions than naive investing baselines?"

The project includes:

- a core ML pipeline for a single stock
- an LSTM-led forecasting engine
- sentiment analysis using FinBERT or VADER
- signal generation using model forecast, sentiment, and momentum
- realistic backtesting with whole-share execution
- a Flask API and React-based dashboard
- a multi-stock portfolio manager

## Current Positioning

The project should be presented as an:

- `AI-Powered Quant Portfolio Management System`
- `LSTM-Based Stock Signal and Portfolio Intelligence System`

This is a better fit for Indian markets than calling it a literal hedge fund.

## Repository Structure

```text
major/
├── ai_hedge_fund.py         # Core pipeline: data, features, LSTM, signals, backtesting
├── app.py                   # Flask API server for the dashboard
├── index.html               # Frontend dashboard (single-file React UI)
├── portfolio_manager.py     # Multi-stock portfolio backtester using real Nifty benchmark
├── base.py                  # Runs one baseline analysis for RELIANCE.NS
├── run_test.py              # Quick smoke test
├── sentiment_backtester.py  # Sentiment-focused experimental backtester
├── requirements.txt         # Python dependencies
└── cache/                   # Generated models, CSVs, charts, and cached sentiment
```

## What the System Does

For a selected stock, the system:

1. downloads OHLCV market data
2. computes technical indicators
3. builds a 90-day rolling sequence dataset
4. runs an LSTM-led return forecast
5. incorporates supporting models for forecast stabilisation
6. scores news sentiment
7. combines forecast, sentiment, and momentum into a final signal
8. simulates trades using realistic whole-share execution
9. compares performance against:
   - asset buy-and-hold
   - real Nifty 50 benchmark (`^NSEI`)

## Core Architecture

```text
Yahoo Finance OHLCV
        │
        ▼
DataIngestor
        │
        ▼
Technical Features
(OHLCV + SMA/EMA + RSI + MACD + BB Width + ATR + OBV + Volume)
        │
        ▼
SentimentAnalyser
(FinBERT / VADER + cached news sentiment)
        │
        ▼
FeaturePipeline
(scaling + sliding windows, seq_len = 90)
        │
        ▼
LSTM-Led Forecasting
(with support models for stabilisation)
        │
        ▼
SignalGenerator
(price forecast + sentiment + momentum)
        │
        ▼
Backtester
(whole shares, stop-loss, trailing stop, take-profit)
        │
        ▼
Metrics + Charts + Dashboard + Portfolio Manager
```

## Model and Features

### Forecasting model

The main forecasting model is an LSTM. This is the core model used in the project because stock market data is sequential and the LSTM is designed to learn temporal dependencies over time.

The project now uses an LSTM-led hybrid forecast internally:

- `LSTM` remains the primary sequence model
- `Ridge` adds a stable linear correction
- `RandomForestRegressor` adds non-linear tabular support

This keeps the LSTM as the main predictive engine while making the return forecast more stable in practice.

### Input features

The model uses:

- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `SMA_20`
- `SMA_50`
- `EMA_20`
- `RSI_14`
- `MACD`
- `MACD_Signal`
- `MACD_Hist`
- `BB_Width`
- `ATR`
- `OBV`

When sentiment data is available, the pipeline also includes:

- `sentiment_score`
- `sentiment_magnitude`
- `headline_count`

### Important note on volume

Yes, the project includes volume-based information. Raw `Volume` and `OBV` are part of the input features, so the model does use participation and flow-related signals. However, that does not mean crash prediction is guaranteed; it only means the model can learn patterns related to volatility and market stress.

## Signal Logic

The dashboard and backtester do not directly trade on raw predicted price. They trade on a final signal score built from three components:

- model-led price score
- sentiment score
- momentum score

Outputs are:

- `STRONG_BUY`
- `BUY`
- `HOLD`
- `SELL`
- `STRONG_SELL`

### Composite score

The `Composite` line shown in the dashboard graphs is the final confidence score after combining forecast, sentiment, and momentum. It is not the stock price. It is the model's overall bullish/bearish conviction for each day.

## Backtesting Logic

The single-stock backtester and comparison helpers now use whole-share execution only. Fractional shares are not used, which makes the simulation more realistic for NSE/BSE cash equity trading.

The backtester includes:

- entry on `BUY` or `STRONG_BUY`
- stop-loss
- trailing stop
- take-profit
- exit on weak or bearish signal

Reported metrics include:

- total return
- benchmark return
- alpha vs Nifty 50
- Sharpe ratio
- max drawdown
- win rate
- trade count

## Benchmarks

There are two important baselines in this project:

- `Asset Buy & Hold`: buying the analysed stock and holding it throughout the period
- `Nifty 50`: real benchmark fetched using `^NSEI`

This distinction matters:

- buy-and-hold can outperform active systems during strong one-direction bull runs
- alpha is measured against Nifty 50, not against buy-and-hold

## Portfolio Manager

`portfolio_manager.py` runs the system across a basket of major Indian stocks and constructs a dynamic portfolio.

Current portfolio characteristics:

- rank-based selection
- conviction-weighted allocation
- trend filter using SMA20, SMA50, and RSI
- market-regime-aware gross exposure
- real Nifty 50 benchmark in the multi-period portfolio backtest

The default portfolio universe is:

- `RELIANCE.NS`
- `TCS.NS`
- `HDFCBANK.NS`
- `INFY.NS`
- `ICICIBANK.NS`
- `ITC.NS`
- `LT.NS`
- `SBIN.NS`
- `BAJFINANCE.NS`
- `BHARTIARTL.NS`

## Dashboard Features

The web dashboard provides:

- stock-wise analysis
- price and prediction chart
- sentiment timeline
- equity curve
- metrics panel
- trade log
- chatbot support for explanation

Displayed benchmark and equity metrics now reflect:

- real Nifty 50 series
- realistic backtesting rules
- whole-share trades

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.9+ is recommended.

### 2. Start the app

```bash
python app.py
```

Then open:

```text
http://localhost:5000
```

### 3. Quick smoke test

```bash
python run_test.py
```

### 4. Baseline single-stock run

```bash
python base.py
```

### 5. Portfolio backtest

```bash
python portfolio_manager.py
```

## Important Configuration

The main settings live in `Config` inside [ai_hedge_fund.py](C:\Users\vinit\OneDrive\Desktop\major\ai_hedge_fund.py).

Key parameters:

- `seq_len = 90`
- `epochs = 100`
- `hidden_size = 128`
- `num_layers = 3`
- `price_weight = 0.70`
- `sentiment_weight = 0.10`
- `momentum_weight = 0.20`
- `strong_threshold = 0.68`
- `signal_threshold = 0.56`
- `stop_loss_pct = 0.07`
- `trailing_stop_pct = 0.05`
- `take_profit_pct = 0.15`
- `benchmark_ticker = "^NSEI"`

## Cache Behaviour

The `cache/` directory stores:

- `data_<TICKER>.csv`
- `sentiment_<TICKER>.csv`
- `model_<TICKER>.pt`
- `loss_<TICKER>.png`
- `backtest_<TICKER>.png`

The pipeline:

- reuses cached market data when valid
- reuses model checkpoints unless retraining is required
- incrementally updates sentiment cache
- saves model checkpoints atomically for stability

## Current Limitations

These are important to state honestly in evaluation:

- free historical news availability is limited
- sentiment quality is weaker for old historical periods than for recent periods
- buy-and-hold can still outperform in long uninterrupted rallies
- the model is regime-sensitive
- transaction costs, brokerage, taxes, and slippage are not fully modeled
- this is a research prototype, not a live trading platform

## Why Buy-and-Hold Can Beat the Model

Buy-and-hold is not a "bad" baseline. In strongly trending markets, it can outperform active models because it never exits.

An active AI system may underperform buy-and-hold when:

- it exits too early
- it stays in cash during part of a rally
- the forecasting signal is not strong enough to time re-entry correctly

So trailing buy-and-hold does not automatically mean the pipeline is broken. It usually means the market favored continuous exposure more than active timing during that period.

## Recommended Viva Positioning

The most defensible way to present the project is:

"We built an LSTM-led AI-based quantitative portfolio management system for Indian equities that combines market data, technical indicators, and news sentiment to generate actionable trading signals and benchmark-aware backtests."

Avoid saying:

- "we always beat the market"
- "the model predicts crashes perfectly"
- "buy-and-hold is bad"
- "this is production-ready"

Instead say:

- "the system is benchmark-aware and research-oriented"
- "it aims to improve structured decision-making"
- "performance varies by market regime"
- "LSTM is the core forecasting engine, with sentiment and momentum as supporting decision layers"

## Dependencies

Main packages used:

- `torch`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `yfinance`
- `transformers`
- `duckduckgo-search`
- `vaderSentiment`
- `flask`
- `flask-cors`
- `requests`


