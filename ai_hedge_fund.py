"""
=============================================================================
AI HEDGE FUND — CORE ML PIPELINE
=============================================================================
File        : ai_hedge_fund.py
Description : End-to-end pipeline for Indian equity markets.
=============================================================================
"""
import os, math, warnings, logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# ── Optional heavy deps — degrade gracefully ──────────────────────────────
try:
    from transformers import pipeline as hf_pipeline
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Cache settings ─────────────────────────────────────────────────────────
CACHE_DIR        = "cache"                # folder next to app.py
SENTIMENT_CEILING = 2000                  # max rows to keep per ticker
RETRAIN_EVERY    = 30                     # retrain model after this many new days
os.makedirs(CACHE_DIR, exist_ok=True)


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class Config:
    """All tuneable hyperparameters in one place."""
    ticker:               str   = "RELIANCE.NS"
    start_date:           str   = "2020-01-01"
    end_date:             str   = datetime.today().strftime("%Y-%m-%d")
    seq_len:              int   = 90        # lookback window for LSTM

    # ── Sentiment ──
    newsapi_key:          str   = ""
    use_finbert:          bool  = True
    sentiment_lookback:   int   = 3         # days of news per trading day

    # ── LSTM architecture ──
    hidden_size:          int   = 128
    num_layers:           int   = 3         # deeper model for returns
    dropout_p:            float = 0.3

    # ── Training ──
    batch_size:           int   = 32
    epochs:               int   = 100
    learning_rate:        float = 1e-3
    patience:             int   = 15
    train_frac:           float = 0.70
    val_frac:             float = 0.15

    # ── Signal fusion weights ──
    price_weight:         float = 0.50      # LSTM prediction weight
    sentiment_weight:     float = 0.30      # FinBERT/VADER weight
    momentum_weight:      float = 0.20      # RSI + MACD weight
    strong_threshold:     float = 0.70
    signal_threshold:     float = 0.55

    # ── Portfolio ──
    initial_capital:      float = 100_000.0
    strong_buy_size:      float = 1.00      # % of capital for STRONG_BUY
    buy_size:             float = 0.60      # % of capital for BUY

    # ── Risk management ──
    stop_loss_pct:        float = 0.07      # relaxed stop — exit if drops 7% from entry
    trailing_stop_pct:    float = 0.05      # trailing stop — exit if drops 5% from peak
    take_profit_pct:      float = 0.15      # take profit — exit if gains 15% from entry

    # ── Alpha Overhaul ──
    min_return_threshold: float = 0.007     # 0.7% predicted move required to call BUY
    benchmark_ticker:     str   = "^NSEI"   # Nifty 50 for benchmarking

    # ── Output files — all set dynamically per ticker ──
    model_path:           str   = ""
    loss_plot:            str   = ""
    backtest_plot:        str   = ""

    # ── Cache control ──
    force_retrain:        bool  = False        # set True to force a fresh train

    # ── FinBERT toggle ──
    use_finbert:          bool  = True      

    def __post_init__(self):
        if not self.newsapi_key:
            self.newsapi_key = os.environ.get("NEWSAPI_KEY", "")
        # Build ticker-specific filenames so tickers never overwrite each other
        safe = self.ticker.replace(".", "_").replace("/", "_")
        if not self.model_path:
            self.model_path    = os.path.join(CACHE_DIR, f"model_{safe}.pt")
        if not self.loss_plot:
            self.loss_plot     = os.path.join(CACHE_DIR, f"loss_{safe}.png")
        if not self.backtest_plot:
            self.backtest_plot = os.path.join(CACHE_DIR, f"backtest_{safe}.png")


# =============================================================================
# PART 1A — DATA INGESTION & TECHNICAL INDICATORS
# =============================================================================

class DataIngestor:
    """
    Downloads OHLCV from yfinance and computes 13 technical features.

    Feature set:
        OHLCV      : Open, High, Low, Close, Volume
        Trend      : SMA_20, SMA_50, EMA_20
        Momentum   : RSI_14
        MACD       : MACD, MACD_Signal, MACD_Hist
        Volatility : BB_Width
    """

    FEATURE_COLS = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_20", "SMA_50", "EMA_20",
        "RSI_14",
        "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Width",
        "ATR", "OBV"
    ]

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def fetch(self) -> pd.DataFrame:
        log.info(f"Fetching {self.cfg.ticker}  {self.cfg.start_date} → {self.cfg.end_date}")
        safe = self.cfg.ticker.replace(".", "_").replace("/", "_")
        csv_path = os.path.join(CACHE_DIR, f"data_{safe}.csv")

        # ── 1. Check local cache first ─────────────────────────────────────
        if os.path.exists(csv_path) and not self.cfg.force_retrain:
            try:
                # Check when the cache was last modified
                mod_time = datetime.fromtimestamp(os.path.getmtime(csv_path)).date()
                if mod_time == datetime.today().date():
                    log.info(f"Loading stock data from today's cache → {csv_path}")
                    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    
                    # Filter down to the exactly requested date range
                    start_req = pd.to_datetime(self.cfg.start_date)
                    end_req   = pd.to_datetime(self.cfg.end_date)
                    
                    df_filtered = df[(df.index >= start_req) & (df.index <= end_req)]
                    if not df_filtered.empty:
                        return df_filtered
            except Exception as e:
                log.warning(f"Cache read failed, downloading fresh data ({e})")

        # ── 2. Download from yfinance ──────────────────────────────────────
        raw = yf.download(
            self.cfg.ticker,
            start=self.cfg.start_date,
            end=self.cfg.end_date,
            progress=False,
            auto_adjust=True,
        )
        if raw.empty:
            raise ValueError(
                f"No data returned for ticker '{self.cfg.ticker}'. "
                "Use suffix .NS for NSE (e.g. RELIANCE.NS)."
            )

        # Flatten MultiIndex columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        # Enforce unique index to prevent multi-row loc[] which crashes the signal generator
        df = df[~df.index.duplicated(keep='first')]
        df = self._add_indicators(df)
        df.dropna(inplace=True)
        log.info(f"Dataset ready: {len(df)} rows × {len(df.columns)} features")

        # ── 3. Export to CSV for caching ───────────────────────────────────
        try:
            df.to_csv(csv_path)
            log.info(f"Stock data saved to cache → {csv_path}")
        except Exception as e:
            log.warning(f"Could not save stock CSV: {e}")

        return df

    def fetch_benchmark(self) -> pd.Series:
        """Fetch Nifty 50 (^NSEI) for the same period."""
        log.info(f"Fetching benchmark {self.cfg.benchmark_ticker}...")
        try:
            bench = yf.download(self.cfg.benchmark_ticker, 
                                start=self.cfg.start_date, 
                                end=self.cfg.end_date, 
                                progress=False, 
                                auto_adjust=True)
            if isinstance(bench.columns, pd.MultiIndex):
                bench.columns = bench.columns.get_level_values(0)
            return bench["Close"]
        except Exception as e:
            log.warning(f"Could not fetch benchmark: {e}")
            return pd.Series()

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute 8 technical indicators on top of OHLCV."""
        c = df["Close"]

        # ── Trend ──
        df["SMA_20"] = c.rolling(20).mean()
        df["SMA_50"] = c.rolling(50).mean()
        df["EMA_20"] = c.ewm(span=20, adjust=False).mean()

        # ── Momentum: RSI-14 ──
        delta = c.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        df["RSI_14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))

        # ── MACD (12, 26, 9) ──
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        df["MACD"]        = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

        # ── Volatility: Bollinger Band Width ──
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        df["BB_Width"] = (4 * std20) / sma20  # (upper - lower) / middle

        # ── Volatility: ATR (Average True Range) ──
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - c.shift())
        low_close = np.abs(df["Low"] - c.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df["ATR"] = true_range.rolling(14).mean()

        # ── Volume: OBV (On-Balance Volume) ──
        obv = (np.sign(delta) * df["Volume"]).fillna(0).cumsum()
        df["OBV"] = obv

        return df


# =============================================================================
# PART 1B — SENTIMENT PIPELINE
# =============================================================================

class SentimentAnalyser:
    """
    Scores daily market sentiment for a given ticker using:
      1. FinBERT  (ProsusAI/finbert)  — finance-domain BERT, best quality
      2. VADER    — lightweight fallback, no GPU needed
      3. Neutral  — silent fallback when neither is available

    News sources (in priority order per day):
      NewsAPI → yfinance .news → Moneycontrol/ET RSS feeds
    """

    def __init__(self, cfg: Config):
        self.cfg       = cfg
        self._finbert  = None
        self._vader    = None
        self._newsapi  = None
        self._init_models()

    def _init_models(self):
        # Try FinBERT first (best quality)
        if self.cfg.use_finbert and FINBERT_AVAILABLE:
            try:
                log.info("Loading FinBERT (ProsusAI/finbert)...")
                cuda_device = 0 if torch.cuda.is_available() else -1
                log.info(f"FinBERT device: {'GPU (cuda:0)' if cuda_device == 0 else 'CPU'}")
                self._finbert = hf_pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    top_k=None,
                    device=cuda_device,
                    batch_size=32,        # process 32 headlines per forward pass
                )
                log.info("✓ FinBERT loaded successfully")
            except Exception as e:
                log.warning(f"FinBERT failed to load ({e}). Falling back to VADER.")

        # VADER fallback
        if self._finbert is None and VADER_AVAILABLE:
            self._vader = SentimentIntensityAnalyzer()
            log.info("✓ VADER sentiment analyser initialised")

        # DuckDuckGo Search
        if DDGS_AVAILABLE:
            log.info("✓ DuckDuckGo Search available")

    def build_daily_sentiment(self, date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Score sentiment for every trading day — with rolling cache.

        Cache behaviour:
          * On first run: scores all days, saves to cache/sentiment_TICKER.csv
          * On subsequent runs: loads cache, scores only NEW days since last run,
            appends them, enforces the SENTIMENT_CEILING rolling window.
        """
        safe       = self.cfg.ticker.replace(".", "_").replace("/", "_")
        cache_path = os.path.join(CACHE_DIR, f"sentiment_{safe}.csv")
        company    = self._ticker_to_name(self.cfg.ticker)

        # Load existing cache if present
        cached_df = None
        if os.path.exists(cache_path):
            try:
                cached_df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
                log.info(f"Sentiment cache loaded: {len(cached_df)} rows")
            except Exception as e:
                log.warning(f"Cache read failed ({e}), rebuilding from scratch.")
                cached_df = None

        # Determine which dates still need scoring
        if cached_df is not None and not cached_df.empty:
            already_scored = set(cached_df.index.normalize())
            new_dates = [d for d in date_index if d.normalize() not in already_scored]
        else:
            new_dates = list(date_index)

        # Score only the new dates
        if new_dates:
            # Split into historical (no news available) and recent (fetch live)
            cutoff     = datetime.today() - timedelta(days=self.RSS_LOOKBACK_DAYS)
            old_dates  = [d for d in new_dates if d.to_pydatetime().replace(tzinfo=None) < cutoff]
            live_dates = [d for d in new_dates if d.to_pydatetime().replace(tzinfo=None) >= cutoff]

            rows = []

            # Historical dates — write zeros instantly, no network calls
            if old_dates:
                log.info(f"Writing {len(old_dates)} historical dates as neutral (no news archive)...")
                for date in old_dates:
                    rows.append({
                        "Date":                date,
                        "sentiment_score":     0.0,
                        "sentiment_magnitude": 0.0,
                        "headline_count":      0,
                    })

            # Recent dates — fetch and score properly
            if live_dates:
                log.info(f"Scoring {len(live_dates)} recent trading day(s) with FinBERT/VADER...")
                try:
                    from tqdm import tqdm
                    date_iter = tqdm(live_dates, desc=f"Sentiment [{self.cfg.ticker}]",
                                     unit="day", colour="green")
                except ImportError:
                    date_iter = live_dates
                today = datetime.today().date()
                for date in date_iter:
                    is_today = date.date() == today
                    headlines         = self._collect_headlines(company, date, is_today=is_today)
                    score, mag, count = self._aggregate_scores(headlines)
                    rows.append({
                        "Date":                date,
                        "sentiment_score":     score,
                        "sentiment_magnitude": mag,
                        "headline_count":      count,
                    })
            new_df = pd.DataFrame(rows).set_index("Date")

            # Merge with cache
            if cached_df is not None and not cached_df.empty:
                combined = pd.concat([cached_df, new_df]).sort_index()
            else:
                combined = new_df.sort_index()

            # Enforce rolling window — drop exactly as many rows as we added
            # so the size stays stable once it hits the ceiling
            if len(combined) > SENTIMENT_CEILING:
                drop = len(new_df)  # drop exactly as many as we just added
                combined = combined.iloc[drop:]
                log.info(f"Rolling window: dropped {drop} oldest rows, "
                         f"added {len(new_df)} new rows, "
                         f"total: {len(combined)} rows")

            # Save back to disk
            try:
                combined.to_csv(cache_path)
                log.info(f"Sentiment cache saved -> {cache_path} ({len(combined)} rows)")
            except Exception as e:
                log.warning(f"Cache save failed: {e}")

            result_df = combined
        else:
            log.info("Sentiment cache is up to date - no new days to score.")
            result_df = cached_df

        avg = result_df["headline_count"].mean()
        log.info(f"Sentiment ready - avg {avg:.1f} headlines/day | "
                 f"Engine: {'FinBERT' if self._finbert else 'VADER' if self._vader else 'Neutral'}")

        # Return both the dataframe and the exact count of newly scored days
        # This is counted BEFORE trimming so the ceiling never distorts it
        result_df._new_days_scored = len(new_dates)
        return result_df

    # ── Ticker → (company name, search keywords) mapping ────────────────
    # Keywords are lowercase. ANY match = headline is relevant.
    TICKER_MAP = {
        "RELIANCE.NS":   ("Reliance Industries",   ["reliance", "ril", "mukesh ambani", "jio"]),
        "INFY.NS":       ("Infosys",                ["infosys", "infy"]),
        "TCS.NS":        ("TCS",                    ["tcs", "tata consultancy"]),
        "HDFCBANK.NS":   ("HDFC Bank",              ["hdfc bank", "hdfcbank", "hdfc"]),
        "ICICIBANK.NS":  ("ICICI Bank",             ["icici bank", "icicibank", "icici"]),
        "WIPRO.NS":      ("Wipro",                  ["wipro"]),
        "SBIN.NS":       ("State Bank of India",    ["sbi", "state bank", "sbin"]),
        "ADANIENT.NS":   ("Adani Enterprises",      ["adani", "gautam adani"]),
        "TATAMOTORS.NS": ("Tata Motors",            ["tata motors", "tatamotors", "jlr"]),
        "BAJFINANCE.NS": ("Bajaj Finance",          ["bajaj finance", "bajfinance"]),
        "HINDUNILVR.NS": ("Hindustan Unilever",     ["hindustan unilever", "hul"]),
        "ITC.NS":        ("ITC",                    ["itc ltd", "itc limited", " itc "]),
        "KOTAKBANK.NS":  ("Kotak Mahindra Bank",    ["kotak", "kotak mahindra"]),
        "LT.NS":         ("Larsen & Toubro",        ["larsen", "l&t", "lnt"]),
        "MARUTI.NS":     ("Maruti Suzuki",          ["maruti", "msil"]),
        "BHARTIARTL.NS": ("Bharti Airtel",          ["airtel", "bharti airtel", "bharti"]),
        "AXISBANK.NS":   ("Axis Bank",              ["axis bank", "axisbank"]),
        "NESTLEIND.NS":  ("Nestle India",           ["nestle india", "nestle"]),
        "SUNPHARMA.NS":  ("Sun Pharma",             ["sun pharma", "sun pharmaceutical"]),
        "DRREDDY.NS":    ("Dr Reddy's",             ["dr reddy", "drreddys"]),
    }

    # Free Indian financial RSS feeds — no API key needed
    RSS_FEEDS = [
        "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://www.moneycontrol.com/rss/marketreports.xml",
        "https://www.moneycontrol.com/rss/latestnews.xml",
        "https://feeds.feedburner.com/ndtvprofit-latest",
        "https://www.business-standard.com/rss/markets-106.rss",
        "https://www.livemint.com/rss/markets",
    ]

    def _ticker_to_name(self, ticker: str) -> str:
        entry = self.TICKER_MAP.get(ticker)
        if entry:
            return entry[0]
        return ticker.replace(".NS", "").replace(".BO", "")

    def _ticker_keywords(self, ticker: str) -> List[str]:
        entry = self.TICKER_MAP.get(ticker)
        if entry:
            return entry[1]
        base = ticker.replace(".NS", "").replace(".BO", "").lower()
        return [base]

    # ── Headline collection from multiple sources ─────────────────────────
    # How many days back RSS feeds are useful (they only carry recent articles)
    RSS_LOOKBACK_DAYS = 30
    _rss_cache: dict = {}  # in-memory cache — fetch RSS once per run, not per date

    def _fetch_rss_once(self) -> List[str]:
        """Fetch all RSS feeds once and store in memory. Reused for every date."""
        import urllib.request
        import xml.etree.ElementTree as ET

        key = self.cfg.ticker
        if key in SentimentAnalyser._rss_cache:
            return SentimentAnalyser._rss_cache[key]

        keywords = self._ticker_keywords(self.cfg.ticker)
        headlines = []

        def matches(text: str) -> bool:
            return any(kw in text.lower() for kw in keywords)

        for url in self.RSS_FEEDS:
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0 (compatible; ARIA/1.0)"},
                )
                with urllib.request.urlopen(req, timeout=5) as r:
                    raw = r.read()
                try:
                    root = ET.fromstring(raw)
                    for item in root.iter("item"):
                        title = (item.findtext("title") or "").strip()
                        desc  = (item.findtext("description") or "").strip()
                        if title and matches(f"{title} {desc}"):
                            headlines.append(title + (". " + desc if desc else ""))
                except Exception as e:
                    log.warning(f"Failed to parse RSS feed {url}: {e}")
                    continue
            except Exception:
                continue

        seen, unique = set(), []
        for h in headlines:
            if h[:80].lower() not in seen:
                seen.add(h[:80].lower())
                unique.append(h)

        SentimentAnalyser._rss_cache[key] = unique
        log.info(f"RSS fetched once: {len(unique)} headlines for {key}")
        return unique

    def _collect_headlines(self, company: str, date: pd.Timestamp,
                           is_today: bool = False) -> List[str]:
        headlines = []
        keywords  = self._ticker_keywords(self.cfg.ticker)
        ds = date.strftime("%Y-%m-%d")
        ps = (date - timedelta(days=self.cfg.sentiment_lookback)).strftime("%Y-%m-%d")

        # Source 1: DuckDuckGo Search (DDGS) - 100% Free
        if DDGS_AVAILABLE:
            try:
                results = DDGS().news(f'"{company}" stock India', max_results=20)
                for a in results:
                    t = a.get("title", "")
                    d = a.get("body", "")
                    if t:
                        headlines.append(t + ". " + d)
            except Exception:
                pass

        # Source 2: yfinance .news — has timestamps, filter correctly per date
        try:
            tk  = yf.Ticker(self.cfg.ticker)
            cut = int(date.timestamp())
            sta = int((date - timedelta(days=self.cfg.sentiment_lookback)).timestamp())
            for n in (tk.news or []):
                c   = n.get("content", {})
                t   = c.get("title", "")
                pub = c.get("pubDate", {})
                ts  = pub.get("raw", 0) if isinstance(pub, dict) else 0
                if t and sta <= ts <= cut:
                    headlines.append(t)
        except Exception:
            pass

        # Source 3: RSS — fetched ONCE, only applied to today
        # RSS has no date filter so applying it to every date gives identical scores
        if is_today:
            headlines += self._fetch_rss_once()

        # Deduplicate while preserving order
        seen, unique = set(), []
        for h in headlines:
            key = h[:80].lower()
            if key not in seen:
                seen.add(key)
                unique.append(h)
        return unique

    # ── Score aggregation ─────────────────────────────────────────────────
    def _aggregate_scores(self, headlines: List[str]) -> Tuple[float, float, int]:
        if not headlines:
            return 0.0, 0.0, 0
        scores = []
        for h in headlines:
            if not h.strip():
                continue
            if self._finbert:
                scores.append(self._finbert_score(h))
            elif self._vader:
                scores.append(self._vader_score(h))
            else:
                scores.append((0.0, 0.0))
        if not scores:
            return 0.0, 0.0, 0
        return (
            float(np.mean([s for s, _ in scores])),
            float(np.mean([m for _, m in scores])),
            len(scores),
        )

    def _finbert_score(self, text: str) -> Tuple[float, float]:
        try:
            res   = self._finbert(text[:512])[0]
            lmap  = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
            best  = max(res, key=lambda x: x["score"])
            score = lmap.get(best["label"].lower(), 0.0) * best["score"]
            return score, best["score"]
        except Exception:
            return 0.0, 0.0

    def _vader_score(self, text: str) -> Tuple[float, float]:
        try:
            c = self._vader.polarity_scores(text)["compound"]
            return c, abs(c)
        except Exception:
            return 0.0, 0.0


# =============================================================================
# PART 1C — FEATURE FUSION & SEQUENCE GENERATION
# =============================================================================

class FeaturePipeline:
    """
    Merges price+indicators with sentiment, MinMax scales (train-only fit),
    and generates (N, seq_len, features) sliding windows for the LSTM.
    """

    def __init__(self, cfg: Config):
        self.cfg          = cfg
        self.scaler       = MinMaxScaler()
        self.feature_cols: List[str] = []

    def build(self, price_df: pd.DataFrame,
              sentiment_df: Optional[pd.DataFrame]
              ) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:

        df = price_df.copy()

        # Merge sentiment features if available
        if sentiment_df is not None and not sentiment_df.empty:
            df = df.join(sentiment_df, how="left")
            for col in ["sentiment_score", "sentiment_magnitude", "headline_count"]:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            self.feature_cols = DataIngestor.FEATURE_COLS + [
                "sentiment_score", "sentiment_magnitude", "headline_count"
            ]
        else:
            self.feature_cols = DataIngestor.FEATURE_COLS

        df = df[self.feature_cols].dropna()

        n         = len(df)
        train_end = int(n * self.cfg.train_frac)

        # Fit scaler on TRAINING data only — prevents data leakage
        self.scaler.fit(df.iloc[:train_end].values)
        scaled = self.scaler.transform(df.values)

        close_idx = self.feature_cols.index("Close")
        seq       = self.cfg.seq_len
        X, y, dates = [], [], []

        for i in range(seq, len(scaled)):
            X.append(scaled[i - seq : i])
            # TARGET: Percentage return for day i (Close_i / Close_i-1 - 1)
            # We use the raw values before scaling for easier return calculation
            curr_c = df["Close"].iloc[i]
            prev_c = df["Close"].iloc[i-1]
            ret    = (curr_c / prev_c) - 1
            y.append(ret)
            dates.append(df.index[i])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        log.info(f"Sequences: {X.shape}  |  Targets: Returns  |  {len(self.feature_cols)} features")
        return X, y, pd.DatetimeIndex(dates)

    def split(self, X, y, dates) -> Dict:
        """Split into train / validation / test sets."""
        n  = len(X)
        tr = int(n * self.cfg.train_frac)
        va = int(n * (self.cfg.train_frac + self.cfg.val_frac))
        return {
            "X_train": X[:tr],   "y_train": y[:tr],
            "X_val":   X[tr:va], "y_val":   y[tr:va],
            "X_test":  X[va:],   "y_test":  y[va:],
            "dates_test": dates[va:],
        }

    def inverse_close(self, scaled: np.ndarray) -> np.ndarray:
        """Convert scaled Close values back to original price scale."""
        ci    = self.feature_cols.index("Close")
        nf    = len(self.feature_cols)
        dummy = np.zeros((len(scaled), nf), dtype=np.float32)
        dummy[:, ci] = scaled
        return self.scaler.inverse_transform(dummy)[:, ci]


# =============================================================================
# PART 2 — LSTM MODEL
# =============================================================================

class AIHedgeFundLSTM(nn.Module):
    """
    Stacked 2-layer LSTM for multimodal stock prediction.

    Forward path:
        (B, S, F) → LSTM ×2 → last timestep (B, H)
                  → Dropout → FC1+ReLU (B, H//2)
                  → Dropout → FC2 (B, 1)
    """

    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 2, dropout_p: float = 0.3,
                 output_size: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout_p)
        self.fc1     = nn.Linear(hidden_size * 2, hidden_size)
        self.relu    = nn.ReLU()
        self.fc2     = nn.Linear(hidden_size, output_size)
        self._init_weights()

    def _init_weights(self):
        """Xavier for input weights, orthogonal for hidden, forget gate bias = 1."""
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
                n = p.size(0)
                p.data[n // 4 : n // 2].fill_(1.0)  # forget gate bias = 1
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def init_hidden(self, B: int, device: torch.device):
        num_directions = 2
        h = torch.zeros(self.num_layers * num_directions, B, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers * num_directions, B, self.hidden_size, device=device)
        return h, c

    def forward(self, x: torch.Tensor, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)
        out, _ = self.lstm(x, hidden)
        out = out[:, -1, :]              # last timestep: (B, H*2)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))   # (B, H)
        out = self.dropout(out)
        return self.fc2(out)             # (B, 1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# PART 3 — TRAINING PIPELINE
# =============================================================================

class Trainer:
    """AdamW + CosineAnnealingLR + early stopping + gradient clipping."""

    def __init__(self, model: AIHedgeFundLSTM, cfg: Config):
        self.model  = model
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        log.info(f"Device: {self.device}  |  Params: {model.count_parameters():,}")

    def train(self, splits: Dict, progress_callback: Optional[Callable] = None) -> Dict:
        tr_loader = self._loader(splits["X_train"], splits["y_train"], shuffle=True)
        va_loader = self._loader(splits["X_val"],   splits["y_val"],   shuffle=False)

        criterion = nn.HuberLoss()
        opt       = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate, weight_decay=1e-4,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.cfg.epochs, eta_min=1e-6,
        )

        best_val    = float("inf")
        patience    = 0
        tr_losses   = []
        va_losses   = []

        for epoch in range(1, self.cfg.epochs + 1):
            tr = self._epoch(tr_loader, criterion, opt, train=True)
            va = self._epoch(va_loader, criterion, opt, train=False)
            sched.step()
            tr_losses.append(tr)
            va_losses.append(va)

            if va < best_val:
                best_val = va
                patience = 0
                torch.save(self.model.state_dict(), self.cfg.model_path)
            else:
                patience += 1

            if epoch % 10 == 0:
                msg = (f"Epoch {epoch:4d} | Train {tr:.6f} | "
                       f"Val {va:.6f} | Best {best_val:.6f} | "
                       f"LR {sched.get_last_lr()[0]:.1e}")
                log.info(msg)
                if progress_callback:
                    progress_callback(f"Training epoch {epoch}/{self.cfg.epochs} — val_loss: {va:.6f}")

            if patience >= self.cfg.patience:
                log.info(f"🛑 Early stop at epoch {epoch}")
                break

        best_epoch = len(tr_losses) - patience
        self._save_loss_plot(tr_losses, va_losses, best_epoch)

        return {
            "train_losses": tr_losses,
            "val_losses":   va_losses,
            "best_epoch":   best_epoch,
            "best_val_loss": best_val,
        }

    def _epoch(self, loader, criterion, opt, train: bool) -> float:
        self.model.train(train)
        total = 0.0
        with torch.set_grad_enabled(train):
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X).squeeze(-1)
                loss = criterion(pred, y)
                if train:
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    opt.step()
                total += loss.item() * len(X)
        return total / len(loader.dataset)

    def _loader(self, X, y, shuffle) -> DataLoader:
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=self.cfg.batch_size,
                          shuffle=shuffle, drop_last=False)

    def _save_loss_plot(self, tr, va, best):
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        ax.plot(tr, color="#00ff88", lw=1.5, label="Train Loss")
        ax.plot(va, color="#ff6b35", lw=1.5, label="Val Loss")
        ax.axvline(best, color="#ffffff", ls="--", lw=0.8,
                   label=f"Best Epoch {best}")
        ax.set_xlabel("Epoch", color="#aaa")
        ax.set_ylabel("MSE Loss", color="#aaa")
        ax.set_title("LSTM Training Loss Curve", color="#fff", fontsize=14)
        ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#fff")
        ax.tick_params(colors="#aaa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")
        plt.tight_layout()
        plt.savefig(self.cfg.loss_plot, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Loss curve saved → {self.cfg.loss_plot}")


# =============================================================================
# PART 4 — SIGNAL GENERATOR
# =============================================================================

class SignalGenerator:
    """
    Fuses three signals into one actionable call per day:

        LSTM Price  (50%) — predicted direction vs today's close
        Sentiment   (30%) — FinBERT/VADER daily score from news
        Momentum    (20%) — RSI + MACD crossover

    Outputs: STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL

    The final call is ALWAYS based on BOTH the LSTM output AND
    the news sentiment output — never just one alone.
    """

    LABEL_MAP = {
        "STRONG_BUY":  {"label": "STRONG BUY",  "emoji": "🟢", "color": "#00ff88"},
        "BUY":         {"label": "BUY",          "emoji": "🟡", "color": "#ffcc00"},
        "HOLD":        {"label": "HOLD",         "emoji": "⚪", "color": "#aaaaaa"},
        "SELL":        {"label": "SELL",         "emoji": "🔴", "color": "#ff6b6b"},
        "STRONG_SELL": {"label": "STRONG SELL",  "emoji": "🔴", "color": "#cc0000"},
    }

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def generate(self, dates, actual_close, pred_close,
                 sentiment_df, price_df) -> pd.DataFrame:
        rows = []
        for i, date in enumerate(dates):
            # ── Price signal from LSTM: predicted Return ──
            # pred_close[i] is now the predicted return decimal
            pred_ret = pred_close[i]
            thresh   = self.cfg.min_return_threshold
            
            if pred_ret >= thresh:      price_score = 1.0  # Bullish
            elif pred_ret <= -thresh:   price_score = 0.0  # Bearish
            else:                       price_score = 0.5  # Neutral conviction

            # ── Sentiment signal from FinBERT/VADER ──
            # FACT-CHECK: Only trust sentiment if multiple sources/headlines agree
            sent_score = 0.5
            if sentiment_df is not None and date in sentiment_df.index:
                row = sentiment_df.loc[date]
                # If only 1 headline, it's often noise — revert to neutral
                if row.get("headline_count", 0) >= 2:
                    raw = float(row["sentiment_score"])
                    sent_score = (raw + 1) / 2      # [-1,+1] → [0, 1]

            # ── Momentum signal from RSI + MACD ──
            mom_score = self._momentum(price_df, date)

            # ── Composite: weighted fusion of all three ──
            composite = (
                self.cfg.price_weight     * price_score +
                self.cfg.sentiment_weight * sent_score  +
                self.cfg.momentum_weight  * mom_score
            )

            # ── Threshold to call ──
            if   composite >= self.cfg.strong_threshold:             signal = "STRONG_BUY"
            elif composite >= self.cfg.signal_threshold:             signal = "BUY"
            elif composite <= (1 - self.cfg.strong_threshold):       signal = "STRONG_SELL"
            elif composite <= (1 - self.cfg.signal_threshold):       signal = "SELL"
            else:                                                     signal = "HOLD"

            try:
                open_price = float(price_df.loc[date, "Open"])
            except Exception:
                open_price = actual_close[i] # fallback

            rows.append({
                "date":         date,
                "actual_close": actual_close[i],
                "open_price":   open_price,
                "pred_return":  round(float(pred_ret), 6),
                "price_score":  round(price_score, 4),
                "sent_score":   round(sent_score, 4),
                "mom_score":    round(mom_score, 4),
                "composite":    round(composite, 4),
                "signal":       signal,
            })

        return pd.DataFrame(rows).set_index("date")

    def _momentum(self, price_df: pd.DataFrame,
                  date: pd.Timestamp) -> float:
        """RSI normalised to [0,1] + MACD crossover binary → average."""
        try:
            row  = price_df.loc[:date].iloc[-1]
            rsi  = float(row.get("RSI_14", 50))
            macd = float(row.get("MACD", 0))
            sig  = float(row.get("MACD_Signal", 0))
            return 0.5 * (rsi / 100) + 0.5 * (1.0 if macd > sig else 0.0)
        except Exception:
            return 0.5


# =============================================================================
# PART 5 — BACKTESTING ENGINE
# =============================================================================

class Backtester:
    """Long/flat strategy driven by signal DataFrame."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, signals: pd.DataFrame, benchmark_series: pd.Series = None) -> Dict:
        """
        Signals: DataFrame with 'Signal', 'Open', 'pred_return'
        benchmark_series: Nifty 50 close prices for same period
        """
        # Align benchmark return
        bench_ret = 0.0
        if benchmark_series is not None and not benchmark_series.empty:
            common = signals.index.intersection(benchmark_series.index)
            if len(common) > 1:
                b = benchmark_series.loc[common]
                bench_ret = (b.iloc[-1] / b.iloc[0] - 1) * 100
        self.benchmark_return = bench_ret

        cash     = self.cfg.initial_capital
        position = 0.0
        entry    = 0.0
        equity   = [cash]
        trades   = []

        sl_mult  = 1.0 - self.cfg.stop_loss_pct    # hard stop from entry
        tp_mult  = 1.0 + self.cfg.take_profit_pct  # take profit from entry
        trail_pct = self.cfg.trailing_stop_pct      # trailing stop % from peak

        peak_price = 0.0  # tracks highest price since entry

        for date, row in signals.iterrows():
            price  = row["actual_close"]
            open_p = row.get("open_price", price)
            signal = row["signal"]

            if position == 0:
                # Buy at the OPEN of the day to capture the move towards predicted CLOSE
                if signal == "STRONG_BUY":
                    shares = (cash * self.cfg.strong_buy_size) / open_p
                    cost   = shares * open_p
                    cash -= cost
                    position   = shares
                    entry      = open_p
                    peak_price = open_p
                    trades.append({"date": date, "action": "BUY",
                                   "price": open_p, "signal": signal,
                                   "shares": round(position, 4)})
                elif signal == "BUY":
                    shares = (cash * self.cfg.buy_size) / open_p
                    cost   = shares * open_p
                    cash -= cost
                    position   = shares
                    entry      = open_p
                    peak_price = open_p
                    trades.append({"date": date, "action": "BUY",
                                   "price": open_p, "signal": signal,
                                   "shares": round(position, 4)})

            elif position > 0:
                # Update peak price — trails upward as price rises
                if price > peak_price:
                    peak_price = price

                # ── Hard stop: price dropped from entry (protects capital) ──
                hard_stop_hit = price <= entry * sl_mult

                # ── Trailing stop: price dropped from peak (locks in profit) ──
                # Only activates if we've moved up from entry (trailing stop logic)
                trailing_stop_hit = (peak_price > entry) and (price <= peak_price * (1 - trail_pct))

                # ── Take profit: price hit gain threshold ──
                profit_hit = price >= entry * tp_mult

                # ── Normal signal exit ──
                signal_exit = signal in ("SELL", "STRONG_SELL", "HOLD")

                if hard_stop_hit or trailing_stop_hit or profit_hit or signal_exit:
                    proceeds = position * price
                    pnl      = position * (price - entry)
                    cash    += proceeds

                    if hard_stop_hit:
                        action = "STOP_LOSS"
                    elif trailing_stop_hit:
                        action = "TRAIL_STOP"
                    elif profit_hit:
                        action = "TAKE_PROFIT"
                    else:
                        action = "SELL"

                    trades.append({"date": date, "action": action,
                                   "price": price, "signal": signal,
                                   "pnl": round(pnl, 2),
                                   "shares": round(position, 4)})
                    position   = 0.0
                    peak_price = 0.0

            # Total Equity = Cash on Hand + Current Value of Position
            current_val = position * price if position > 0 else 0
            equity.append(cash + current_val)

        # Close any open position at end
        if position > 0:
            lp  = signals["actual_close"].iloc[-1]
            proceeds = position * lp
            pnl = position * (lp - entry)
            cash += proceeds
            trades.append({"date": signals.index[-1], "action": "SELL(EOD)",
                           "price": lp, "signal": "EOD",
                           "pnl": round(pnl, 2),
                           "shares": round(position, 4)})
            equity[-1] = cash # Correct last equity point

        eq = pd.Series(equity)
        m  = self._compute_metrics(eq, trades)
        log.info(f"Backtest: Return {m['total_return_pct']:+.2f}%  "
                 f"Sharpe {m['sharpe']:.2f}  MaxDD {m['max_drawdown_pct']:.2f}%  "
                 f"WinRate {m['win_rate_pct']:.1f}%  "
                 f"HardStop: {m['stop_loss_hits']}  TrailStop: {m['trail_stop_hits']}  TakeProfit: {m['take_profit_hits']}")
        return {"metrics": m, "equity_curve": eq,
                "trades": pd.DataFrame(trades), "signals": signals}

    def _compute_metrics(self, eq: pd.Series, trades: List) -> Dict:
        ret    = eq.pct_change().dropna()
        total  = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
        sharpe = (ret.mean() / (ret.std() + 1e-9)) * math.sqrt(252)
        dd     = ((eq - eq.cummax()) / eq.cummax()).min() * 100
        # Count all closing actions as trades (SELL, STOP_LOSS, TAKE_PROFIT, EOD)
        exit_actions = {"SELL", "STOP_LOSS", "TRAIL_STOP", "TAKE_PROFIT", "SELL(EOD)"}
        sells  = [t for t in trades if t.get("action", "") in exit_actions]
        wins   = [t for t in sells  if t.get("pnl", 0) > 0]
        wr     = (len(wins) / len(sells) * 100) if sells else 0.0
        sl_hits    = sum(1 for t in sells if t.get("action") == "STOP_LOSS")
        trail_hits = sum(1 for t in sells if t.get("action") == "TRAIL_STOP")
        tp_hits    = sum(1 for t in sells if t.get("action") == "TAKE_PROFIT")

        # Alpha (Return vs Nifty)
        bench_ret = getattr(self, "benchmark_return", 0.0)
        alpha     = total - bench_ret

        return {
            "total_return_pct": round(total, 2),
            "benchmark_return_pct": round(bench_ret, 2),
            "alpha_pct":        round(alpha, 2),
            "sharpe":           round(sharpe, 2),
            "max_drawdown_pct": round(dd, 2),
            "win_rate_pct":     round(wr, 1),
            "num_trades":       len(sells),
            "final_capital":    round(eq.iloc[-1], 2),
            "stop_loss_hits":   sl_hits,
            "trail_stop_hits":  trail_hits,
            "take_profit_hits": tp_hits,
        }

    def save_plot(self, result: Dict, buy_hold: np.ndarray):
        """Generate backtest chart with equity curve vs buy-and-hold."""
        eq      = result["equity_curve"]
        signals = result["signals"]
        m       = result["metrics"]

        fig = plt.figure(figsize=(14, 8), facecolor="#0d1117")
        gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

        # ── Top: Equity curve ──
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor("#0d1117")
        ax1.plot(eq.values[1:], color="#00ff88", lw=2, label="AI Portfolio")
        bh = (buy_hold / buy_hold[0]) * self.cfg.initial_capital
        ax1.plot(bh, color="#ffa500", lw=1.5, ls="--",
                 label="Buy & Hold", alpha=0.8)
        ax1.set_title(
            f"{self.cfg.ticker}  |  Return: {m['total_return_pct']:+.1f}%  "
            f"Sharpe: {m['sharpe']:.2f}  MaxDD: {m['max_drawdown_pct']:.1f}%  "
            f"WinRate: {m['win_rate_pct']:.0f}%",
            color="#fff", fontsize=11)
        ax1.set_ylabel("Portfolio Value (₹)", color="#aaa")
        ax1.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#fff")
        ax1.tick_params(colors="#aaa")
        for sp in ax1.spines.values():
            sp.set_edgecolor("#333")

        # ── Bottom: Actual vs Predicted ──
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor("#0d1117")
        ax2.plot(signals["actual_close"].values, color="#fff", lw=1.2, label="Actual")
        if "pred_return" in signals.columns:
            # Shift and secondary axis for returns
            ax2_ret = ax2.twinx()
            ax2_ret.plot(signals["pred_return"].values, color="#00aaff", lw=1.0,
                         ls="--", label="Pred Return", alpha=0.8)
            ax2_ret.set_ylabel("Pred Return", color="#00aaff", fontsize=8)
            ax2_ret.tick_params(colors="#00aaff", labelsize=8)
        elif "pred_close" in signals.columns:
            ax2.plot(signals["pred_close"].values, color="#00aaff", lw=1.0,
                     ls="--", label="Predicted", alpha=0.8)
        ax2.set_ylabel("Close (₹)", color="#aaa")
        ax2.set_xlabel("Test Day", color="#aaa")
        ax2.legend(facecolor="#1a1a2e", edgecolor="#333",
                   labelcolor="#fff", fontsize=8)
        ax2.tick_params(colors="#aaa")
        for sp in ax2.spines.values():
            sp.set_edgecolor("#333")

        plt.savefig(self.cfg.backtest_plot, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Backtest chart saved → {self.cfg.backtest_plot}")


# =============================================================================
# ORCHESTRATOR — FULL PIPELINE
# =============================================================================

def run_full_pipeline(ticker: str = "RELIANCE.NS",
                      cfg: Optional[Config] = None,
                      progress_callback: Optional[Callable] = None) -> Dict:
    """
    Main entry point — called by app.py or directly.

    progress_callback: optional callable(msg: str) for live status updates
                       to the Flask SSE stream.
    """
    def step(msg: str):
        log.info(msg)
        if progress_callback:
            progress_callback(msg)

    if cfg is None:
        cfg = Config(ticker=ticker)
    else:
        cfg.ticker = ticker

    # ── Step 1: Fetch OHLCV + technical indicators ────────────────────────
    step("Fetching OHLCV data from Yahoo Finance...")
    ingestor = DataIngestor(cfg)
    price_df = ingestor.fetch()

    # ── Step 2: Score news sentiment ──────────────────────────────────────
    step("Scoring news sentiment (FinBERT / VADER)...")
    sentiment_analyser = SentimentAnalyser(cfg)
    sentiment_df       = sentiment_analyser.build_daily_sentiment(price_df.index)

    # Track accumulated new days for retrain trigger
    # Uses the exact count from before trimming so the ceiling never distorts it
    safe         = ticker.replace(".", "_").replace("/", "_")
    newdays_path = os.path.join(CACHE_DIR, f"newdays_{safe}.txt")
    new_scored   = getattr(sentiment_df, "_new_days_scored", 0)
    try:
        prev_accumulated = 0
        if os.path.exists(newdays_path):
            prev_accumulated = int(open(newdays_path).read().strip())
        accumulated_new_days = prev_accumulated + new_scored
        open(newdays_path, "w").write(str(accumulated_new_days))
        if new_scored > 0:
            log.info(f"New days this run: {new_scored} | "
                     f"Accumulated since last retrain: {accumulated_new_days}/{RETRAIN_EVERY}")
        else:
            log.info("No new days this run — counter unchanged.")
    except Exception as e:
        log.warning(f"Could not update new-days counter: {e}")
        accumulated_new_days = 0

    # ── Step 3: Build feature sequences ───────────────────────────────────
    step("Building feature sequences for LSTM...")
    pipeline    = FeaturePipeline(cfg)
    X, y, dates = pipeline.build(price_df, sentiment_df)
    splits      = pipeline.split(X, y, dates)

    # ── Step 4: Train LSTM model (skip if cached model is fresh enough) ───
    input_size = X.shape[2]
    model      = AIHedgeFundLSTM(
        input_size, cfg.hidden_size, cfg.num_layers, cfg.dropout_p,
    )

    # accumulated_new_days was computed cleanly in step 2
    model_exists   = os.path.exists(cfg.model_path)
    should_retrain = (
        cfg.force_retrain
        or not model_exists
        or accumulated_new_days >= RETRAIN_EVERY
    )

    if should_retrain:
        if not model_exists:
            step("Training LSTM model (first run for this ticker)...")
        elif cfg.force_retrain:
            step("Training LSTM model (forced retrain)...")
        else:
            step(f"Retraining LSTM model ({accumulated_new_days} new days accumulated)...")

        trainer    = Trainer(model, cfg)
        train_info = trainer.train(splits, progress_callback=progress_callback)

        # Reset the new-days counter after retraining
        try:
            open(newdays_path, "w").write("0")
        except Exception:
            pass
    else:
        step(f"Loading cached LSTM model (skipping training — {accumulated_new_days}/{RETRAIN_EVERY} new days)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(cfg.model_path, map_location=device, weights_only=True))
        log.info(f"Cached model loaded from {cfg.model_path}")
        # Build dummy train_info so the rest of the pipeline still works
        train_info = {
            "train_losses": [0.0],
            "val_losses":   [0.0],
            "best_epoch":   0,
            "best_val_loss": 0.0,
        }

    # ── Step 5: Inference on test set ─────────────────────────────────────
    step("Running inference on test set...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(cfg.model_path, map_location=device, weights_only=True))
    model.eval().to(device)
    with torch.no_grad():
        preds_sc = model(
            torch.tensor(splits["X_test"]).to(device)
        ).squeeze(-1).cpu().numpy()

    # Targets are returns, so no inverse scaling needed for Close price
    pred_return  = preds_sc
    dates_test   = splits["dates_test"]
    actual_close = price_df.loc[dates_test, "Close"].values

    # ── Step 6: Generate trade signals (LSTM + Sentiment fused) ───────────
    step("Generating trade signals (LSTM + Sentiment + Momentum)...")
    sig_gen = SignalGenerator(cfg)
    signals = sig_gen.generate(
        dates_test, actual_close, pred_return, sentiment_df, price_df,
    )
    latest = signals.iloc[-1].to_dict()
    latest["date"]   = str(signals.index[-1].date())
    latest["ticker"] = cfg.ticker

    # ── Step 7: Backtest ──────────────────────────────────────────────────
    step("Running backtest simulation...")
    bench_data = ingestor.fetch_benchmark()
    bt         = Backtester(cfg)
    bt_result  = bt.run(signals, benchmark_series=bench_data)
    buy_hold  = price_df.loc[dates_test, "Close"].values
    bt.save_plot(bt_result, buy_hold)

    # ── Step 8: Build analytics data for extra charts ─────────────────────
    step("Building analytics data...")

    # RSI data for analytics
    rsi_data = []
    for d in dates_test:
        try:
            rsi_val = float(price_df.loc[:d, "RSI_14"].iloc[-1])
            rsi_data.append(round(rsi_val, 2))
        except Exception:
            rsi_data.append(50.0)

    # MACD data for analytics
    macd_data = []
    macd_signal_data = []
    macd_hist_data = []
    for d in dates_test:
        try:
            row = price_df.loc[:d].iloc[-1]
            macd_data.append(round(float(row.get("MACD", 0)), 4))
            macd_signal_data.append(round(float(row.get("MACD_Signal", 0)), 4))
            macd_hist_data.append(round(float(row.get("MACD_Hist", 0)), 4))
        except Exception:
            macd_data.append(0)
            macd_signal_data.append(0)
            macd_hist_data.append(0)

    # Signal distribution
    signal_counts = signals["signal"].value_counts().to_dict()

    # Serialise trades for JSON
    trades_list = []
    if not bt_result["trades"].empty:
        trades_list = bt_result["trades"].fillna(0).to_dict("records")
        for t in trades_list:
            t["date"] = str(t["date"])[:10]
            for k, v in t.items():
                if isinstance(v, (np.floating, np.integer)):
                    t[k] = float(v)

    step("✅ Pipeline complete!")

    return {
        "ticker":        cfg.ticker,
        "latest_signal": latest,
        "metrics":       bt_result["metrics"],
        "chart_data": {
            "dates":        [str(d.date()) for d in signals.index],
            "actual_close": [round(float(v), 2) for v in actual_close],
            "pred_return":  [round(float(v), 6) for v in pred_return],
            "signals":      signals["signal"].tolist(),
            "sentiment":    signals["sent_score"].tolist(),
            "composite":    signals["composite"].tolist(),
            "equity_curve": bt_result["equity_curve"].tolist()[1:],
            "buy_hold":     [round(float(v), 2) for v in buy_hold],
            "benchmark":    [round(float(v), 2) for v in bench_data.reindex(signals.index).fillna(method='ffill').tolist()] if not bench_data.empty else [],
        },
        "analytics": {
            "rsi":          rsi_data,
            "macd":         macd_data,
            "macd_signal":  macd_signal_data,
            "macd_hist":    macd_hist_data,
            "signal_distribution": signal_counts,
            "train_losses": [round(l, 6) for l in train_info["train_losses"]],
            "val_losses":   [round(l, 6) for l in train_info["val_losses"]],
        },
        "trade_log":     trades_list,
        "training_info": {
            "best_epoch":    train_info["best_epoch"],
            "best_val_loss": round(train_info["best_val_loss"], 6),
            "total_epochs":  len(train_info["train_losses"]),
            "features_used": pipeline.feature_cols,
            "input_size":    input_size,
            "parameters":    model.count_parameters(),
        },
        "signal_meta": SignalGenerator.LABEL_MAP,
        "raw_test_data": {
            "dates": dates_test.tolist(),
            "actual": [float(v) for v in actual_close],
            "predictions": pred_return.tolist()
        }
    }




# =============================================================================
# MODEL COMPARISON — LSTM vs Baselines
# =============================================================================

def run_comparison(ticker: str = "RELIANCE.NS",
                   cfg: Optional[Config] = None) -> Dict:
    """
    Runs LSTM alongside 4 baseline models on the same test set and returns
    a comparison table of RMSE, MAE, Total Return, Sharpe, Max Drawdown, Win Rate.

    Baselines:
        1. Moving Average Crossover (SMA20 vs SMA50)
        2. Linear Regression
        3. Random Forest
        4. ARIMA (optional — falls back gracefully if statsmodels not installed)
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    if cfg is None:
        cfg = Config(ticker=ticker)
    else:
        cfg.ticker = ticker

    log.info(f"Running model comparison for {ticker}...")

    # ── Fetch data ────────────────────────────────────────────────────────
    price_df = DataIngestor(cfg).fetch()

    # ── Load sentiment cache if available ─────────────────────────────────
    safe       = ticker.replace(".", "_").replace("/", "_")
    cache_path = os.path.join(CACHE_DIR, f"sentiment_{safe}.csv")
    if os.path.exists(cache_path):
        try:
            sentiment_df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
            log.info("Comparison: loaded sentiment cache")
        except Exception:
            sentiment_df = None
    else:
        sentiment_df = None

    # ── Build features ────────────────────────────────────────────────────
    pipeline    = FeaturePipeline(cfg)
    X, y, dates = pipeline.build(price_df, sentiment_df)
    splits      = pipeline.split(X, y, dates)

    actual_close = pipeline.inverse_close(splits["y_test"])
    dates_test   = splits["dates_test"]
    n_test       = len(actual_close)

    results = {}

    # ── Helper: compute prediction metrics ───────────────────────────────
    def pred_metrics(actual, predicted, name):
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        mae  = float(mean_absolute_error(actual, predicted))
        return {"name": name, "rmse": round(rmse, 4), "mae": round(mae, 4)}

    # ── Helper: simple backtest from predicted prices ─────────────────────
    def simple_backtest(actual, predicted, cfg):
        """Buy when predicted > prev_actual, sell otherwise."""
        capital  = cfg.initial_capital
        position = 0.0
        entry    = 0.0
        equity   = [capital]
        trades   = []
        sl_mult  = 1.0 - cfg.stop_loss_pct
        tp_mult  = 1.0 + cfg.take_profit_pct

        for i, price in enumerate(actual):
            # predicted is now a return (0.01 = +1%)
            bullish = predicted[i] > 0
            prev    = actual[i-1] if i > 0 else actual[i]

            if position == 0 and bullish:
                position = (capital * cfg.buy_size) / price
                entry    = price
                trades.append({"pnl": 0})
            elif position > 0:
                stop_hit   = price <= entry * sl_mult
                profit_hit = price >= entry * tp_mult
                if stop_hit or profit_hit or not bullish:
                    pnl     = position * (price - entry)
                    capital = capital - position * entry + position * price
                    trades.append({"pnl": round(pnl, 2)})
                    position = 0.0
            equity.append(capital + (position * price if position > 0 else 0))

        if position > 0:
            lp = actual[-1]
            pnl = position * (lp - entry)
            capital += pnl
            trades.append({"pnl": round(pnl, 2)})

        eq     = pd.Series(equity)
        ret    = eq.pct_change().dropna()
        total  = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
        sharpe = (ret.mean() / (ret.std() + 1e-9)) * math.sqrt(252)
        dd     = ((eq - eq.cummax()) / eq.cummax()).min() * 100
        sells  = [t for t in trades if "pnl" in t]
        wins   = [t for t in sells if t["pnl"] > 0]
        wr     = (len(wins) / len(sells) * 100) if sells else 0.0
        # Alpha (Return vs Nifty)
        bench_ret = 0.0
        alpha     = total - bench_ret

        return {
            "total_return_pct": round(total, 2),
            "benchmark_return_pct": round(bench_ret, 2),
            "alpha_pct":        round(alpha, 2),
            "sharpe":           round(sharpe, 2),
            "max_drawdown_pct": round(dd, 2),
            "win_rate_pct":     round(wr, 1),
            "equity_curve":     [round(float(v), 2) for v in eq.tolist()],
        }

    # ── 1. LSTM ───────────────────────────────────────────────────────────
    log.info("Comparison: running LSTM inference...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X.shape[2]
    lstm_model = AIHedgeFundLSTM(input_size, cfg.hidden_size, cfg.num_layers, cfg.dropout_p)
    model_path = cfg.model_path

    # Force re-train LSTM for returns if needed or just use current
    lstm_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    lstm_model.eval().to(device)
    with torch.no_grad():
        preds_sc = lstm_model(torch.tensor(splits["X_test"]).to(device)).squeeze(-1).cpu().numpy()
    
    # Return already
    lstm_pred = preds_sc

    lstm_pm  = pred_metrics(actual_close, lstm_pred, "LSTM")
    lstm_bt  = simple_backtest(actual_close, lstm_pred, cfg)
    results["LSTM"] = {**lstm_pm, **lstm_bt}

    # ── 2. Linear Regression ──────────────────────────────────────────────
    log.info("Comparison: running Linear Regression...")
    X_tr_2d = splits["X_train"].reshape(len(splits["X_train"]), -1)
    X_te_2d = splits["X_test"].reshape(len(splits["X_test"]), -1)
    lr = LinearRegression()
    lr.fit(X_tr_2d, splits["y_train"])
    lr_pred = lr.predict(X_te_2d)

    lr_pm  = pred_metrics(actual_close, lr_pred, "Linear Regression")
    lr_bt  = simple_backtest(actual_close, lr_pred, cfg)
    results["Linear Regression"] = {**lr_pm, **lr_bt}

    # ── 3. Random Forest ──────────────────────────────────────────────────
    log.info("Comparison: running Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr_2d, splits["y_train"])
    rf_pred = rf.predict(X_te_2d)

    rf_pm  = pred_metrics(actual_close, rf_pred, "Random Forest")
    rf_bt  = simple_backtest(actual_close, rf_pred, cfg)
    results["Random Forest"] = {**rf_pm, **rf_bt}

    # ── 4. Moving Average Crossover (SMA20 > SMA50 = buy) ─────────────────
    log.info("Comparison: running Moving Average Crossover...")
    ma_pred = []
    for d in dates_test:
        try:
            row   = price_df.loc[:d].iloc[-1]
            sma20 = float(row.get("SMA_20", 0))
            sma50 = float(row.get("SMA_50", 0))
            close = float(row.get("Close", 0))
            # Predict positive return if SMA20 > SMA50, negative otherwise
            ma_pred.append(0.005 if sma20 > sma50 else -0.005)
        except Exception:
            ma_pred.append(actual_close[len(ma_pred)] if len(ma_pred) < len(actual_close) else 0)
    ma_pred = np.array(ma_pred)

    ma_pm  = pred_metrics(actual_close, ma_pred, "MA Crossover")
    ma_bt  = simple_backtest(actual_close, ma_pred, cfg)
    results["MA Crossover"] = {**ma_pm, **ma_bt}

    # ── 5. ARIMA ──────────────────────────────────────────────────────────
    try:
        from statsmodels.tsa.arima.model import ARIMA
        log.info("Comparison: running ARIMA...")
        train_close = price_df["Close"].iloc[:int(len(price_df) * cfg.train_frac)].values
        arima_preds = []
        history     = list(train_close[-100:])  # use last 100 train points as seed
        test_close  = actual_close

        for i in range(n_test):
            try:
                model_a = ARIMA(history, order=(5, 1, 0))
                fit_a   = model_a.fit()
                yhat    = float(fit_a.forecast(steps=1)[0])
            except Exception:
                yhat = history[-1]
            arima_preds.append(yhat)
            history.append(test_close[i])

        arima_pred = np.array(arima_preds)
        arima_pm   = pred_metrics(actual_close, arima_pred, "ARIMA")
        arima_bt   = simple_backtest(actual_close, arima_pred, cfg)
        results["ARIMA"] = {**arima_pm, **arima_bt}
    except ImportError:
        log.warning("statsmodels not installed — skipping ARIMA. pip install statsmodels")
    except Exception as e:
        log.warning(f"ARIMA failed: {e}")

    log.info("Model comparison complete.")
    return {
        "ticker":  ticker,
        "models":  results,
        "dates":   [str(d.date()) for d in dates_test],
        "actual":  [round(float(v), 2) for v in actual_close],
    }

# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    result = run_full_pipeline("RELIANCE.NS")
    sig    = result["latest_signal"]
    m      = result["metrics"]
    meta   = result["signal_meta"][sig["signal"]]

    print("\n" + "=" * 56)
    print(f"  {result['ticker']}    {meta['emoji']}  {meta['label']}")
    print("=" * 56)
    print(f"  Composite Score : {sig['composite']:.2f}")
    print(f"  Price Signal    : {sig['price_score']:.2f}  (LSTM)")
    print(f"  Sentiment       : {sig['sent_score']:.2f}  (FinBERT/VADER)")
    print(f"  Momentum        : {sig['mom_score']:.2f}  (RSI+MACD)")
    print("-" * 56)
    print(f"  Total Return    : {m['total_return_pct']:+.2f}%")
    print(f"  Sharpe Ratio    : {m['sharpe']:.2f}")
    print(f"  Max Drawdown    : {m['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate        : {m['win_rate_pct']:.1f}%")
    print(f"  Trades Taken    : {m['num_trades']}")
    print(f"  Final Capital   : ₹{m['final_capital']:,.2f}")
    print("=" * 56)