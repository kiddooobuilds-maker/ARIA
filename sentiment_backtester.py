"""
Historical Sentiment Fusion Backtester
Proves the value of integrating News Sentiment (via FinBERT) into a Technical LSTM model.
"""

import os
import json
import time
import logging
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Try to import gnews for historical news gathering
try:
    from gnews import GNews
    GNEWS_AVAILABLE = True
except ImportError:
    GNEWS_AVAILABLE = False

# Import existing architecture from the main project
from ai_hedge_fund import (
    Config, DataIngestor, FeaturePipeline, AIHedgeFundLSTM, Trainer,
    FINBERT_AVAILABLE, hf_pipeline, SEED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

class SentimentBacktester:
    def __init__(self, ticker="RELIANCE.NS", backtest_days=90):
        self.ticker = ticker
        self.backtest_days = backtest_days
        self.cfg = Config(ticker=ticker, epochs=20, seq_len=60) # lower epochs for fast PoC training
        
        # Determine human-readable company name for news searching
        self.company_name = self._get_company_name()
        
        safe = self.ticker.replace(".", "_").replace("/", "_")
        self.cfg.model_path = os.path.join(CACHE_DIR, f"model_backtester_{safe}.pt")
        self.news_cache_file = os.path.join(CACHE_DIR, f"gnews_{self.ticker}_{self.backtest_days}d.json")

        self.finbert = None
        if FINBERT_AVAILABLE:
            log.info("Loading FinBERT...")
            device = 0 if torch.cuda.is_available() else -1
            self.finbert = hf_pipeline(
                "text-classification", model="ProsusAI/finbert", top_k=None, device=device
            )

    def _get_company_name(self):
        # Quick map for common Indian tickers
        tmap = {"RELIANCE.NS": "Reliance Industries", "TCS.NS": "Tata Consultancy", "INFY.NS": "Infosys", "HDFCBANK.NS" : "HDFC Bank"}
        return tmap.get(self.ticker, self.ticker.replace(".NS", ""))

    def fetch_historical_news(self, dates: pd.DatetimeIndex):
        """Scrape news exactly for the given dates using gnews, bypassing API limits."""
        if not GNEWS_AVAILABLE:
            log.warning("GNews not installed. Run `pip install gnews`.")
            return {}

        # Load cache to avoid IP bans during repeated project testing
        cache = {}
        if os.path.exists(self.news_cache_file):
            with open(self.news_cache_file, "r") as f:
                cache = json.load(f)

        news_dict = cache.copy()
        new_fetches = 0

        # Optional generic loading bar for the console
        try:
            from tqdm import tqdm
            date_iter = tqdm(dates, desc="Scraping 90-Day News Array", unit="day", colour="magenta")
        except ImportError:
            date_iter = dates

        for date in date_iter:
            d_str = date.strftime("%Y-%m-%d")
            if d_str in news_dict:
                continue
            
            start_d = date - timedelta(days=3)
            # GNews takes tuples: (YYYY, MM, DD)
            gn = GNews(
                start_date=(start_d.year, start_d.month, start_d.day),
                end_date=(date.year, date.month, date.day),
                max_results=10
            )
            
            query = f'"{self.company_name}" stock India'
            try:
                results = gn.get_news(query)
                headlines = [r.get("title", "") for r in results if r.get("title")]
                news_dict[d_str] = headlines
                new_fetches += 1
                time.sleep(1.5) # Be polite to Google servers
            except Exception as e:
                log.error(f"Failed to fetch news for {d_str}: {e}")
                news_dict[d_str] = []

        if new_fetches > 0:
            with open(self.news_cache_file, "w") as f:
                json.dump(news_dict, f)
            log.info(f"Fetched news for {new_fetches} new historical days. Cache updated.")

        return news_dict

    def score_sentiment(self, headlines):
        """Run FinBERT on the headlines for a single day."""
        if not headlines or not self.finbert:
            return 0.0
        
        scores = []
        for h in headlines:
            try:
                res = self.finbert(h[:512])[0]
                lmap = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
                best = max(res, key=lambda x: x["score"])
                scores.append(lmap.get(best["label"].lower(), 0.0) * best["score"])
            except Exception:
                pass
                
        return float(np.mean(scores)) if scores else 0.0

    def run(self):
        log.info(f"=== Starting Sentiment Fusion Backtester for {self.ticker} ===")
        
        # 1. Base Data Gathering
        ingestor = DataIngestor(self.cfg)
        df_full = ingestor.fetch()
        
        # We need the last N days + seq_len history for our backtest window
        required_rows = self.backtest_days + self.cfg.seq_len
        if len(df_full) < required_rows:
            log.error("Not enough historical data.")
            return

        # Split: Train on everything BEFORE the backtest window
        df_train = df_full.iloc[:-self.backtest_days]
        df_test = df_full.iloc[-self.backtest_days:]
        
        log.info(f"Training Data: {len(df_train)} days. Backtest Data (Out of Sample): {len(df_test)} days.")

        # 2. Tech-Only LSTM Training
        fp = FeaturePipeline(self.cfg)
        # We pass NO sentiment to Train, proving the model relies completely on math
        X_all, y_all, dates_all = fp.build(df_train, None)
        
        # Split train for validation
        split_idx = int(len(X_all) * 0.85)
        splits = {
            "X_train": X_all[:split_idx], "y_train": y_all[:split_idx],
            "X_val": X_all[split_idx:], "y_val": y_all[split_idx:]
        }
        
        model = AIHedgeFundLSTM(input_size=X_all.shape[-1], hidden_size=64, num_layers=2)
        trainer = Trainer(model, self.cfg)
        log.info("Training Technical LSTM Model (No News Baseline)...")
        trainer.train(splits)
        
        model.eval()

        # 3. Simulate Backtest Window Data Preparation
        log.info("Gathering Exact Historical News for Backtest Window...")
        test_dates = df_test.index
        news_data = self.fetch_historical_news(test_dates)

        # Build Technical Test sequences (pad with end of train data to have seq_len history)
        df_test_padded = df_full.iloc[-(self.backtest_days + self.cfg.seq_len):]
        # Transform using the scaler fitted on training data
        scaled_test = fp.scaler.transform(df_test_padded[fp.feature_cols].values)
        
        X_test = []
        for i in range(self.cfg.seq_len, len(scaled_test)):
            X_test.append(scaled_test[i - self.cfg.seq_len : i])
        X_test = torch.tensor(np.array(X_test, dtype=np.float32)).to(trainer.device)

        # Predict Base Technicals
        with torch.no_grad():
            tech_preds = model(X_test).cpu().numpy().flatten()

        # 4. Duel Portfolio Simulation
        # Simulate two $100,000 portfolios over the 90 days.
        port_base = 100_000.0
        port_fusion = 100_000.0
        
        history_base = []
        history_fusion = []
        
        for i, date in enumerate(test_dates):
            close_price = df_test.iloc[i]["Close"]
            d_str = date.strftime("%Y-%m-%d")
            
            # AI Technical Prediction (Expected % return)
            pred_return = tech_preds[i]
            
            # FinBERT Sentiment
            headlines = news_data.get(d_str, [])
            sentiment = self.score_sentiment(headlines)
            
            # --- Strategy A: Base Model (Threshold: expecting > 0.5% return)
            # If pred returns > 0.5%, we use 50% of the portfolio.
            base_alloc = 0.5 if pred_return > 0.005 else (0.2 if pred_return > 0 else 0)
            
            # --- Strategy B: Fusion Model (Threshold adjusted by Sentiment)
            # Fused score: 70% technical, 30% sentiment multiplier
            fused_score = pred_return + (sentiment * 0.01) # sentiment [-1,1] scales as 1% return shift
            fusion_alloc = 0.5 if fused_score > 0.005 else (0.2 if fused_score > 0 else 0)

            # Calculate actual next day return for PnL (assuming holding overnight into next close)
            if i < len(test_dates) - 1:
                next_close = df_test.iloc[i+1]["Close"]
                actual_return = (next_close - close_price) / close_price
            else:
                actual_return = 0.0  # Final day, no next day to return
            
            # Daily PnL
            port_base *= (1 + (base_alloc * actual_return))
            port_fusion *= (1 + (fusion_alloc * actual_return))
            
            history_base.append(port_base)
            history_fusion.append(port_fusion)

        # 5. Result Plotting
        self._plot_results(test_dates, history_base, history_fusion)

    def _plot_results(self, dates, base, fusion):
        plt.figure(figsize=(12, 6))
        plt.plot(dates, base, label="Technical LSTM ONLY (Control)", color="#ff5858", linewidth=2)
        plt.plot(dates, fusion, label="LSTM + FinBERT News Fusion (Test)", color="#00ff88", linewidth=2)
        
        plt.title(f"Proof of Concept: 90-Day Unseen Data Backtest ({self.ticker})", fontsize=14, color="white")
        plt.ylabel("Portfolio Value (₹)", color="white")
        plt.xlabel("Date", color="white")
        
        ax = plt.gca()
        ax.set_facecolor("#161b22")
        plt.gcf().patch.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
            
        plt.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white")
        plt.grid(color="#21262d", linestyle="--")
        
        out_file = "comparison_backtest.png"
        plt.savefig(out_file, bbox_inches="tight", dpi=150)
        log.info(f"Graph generated: {out_file}")
        
        roi_base = ((base[-1] - 100_000) / 100_000) * 100
        roi_fusion = ((fusion[-1] - 100_000) / 100_000) * 100
        log.info(f"Final LSTM Only ROI:  {roi_base:.2f}%")
        log.info(f"Final Fusion ROI:     {roi_fusion:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", type=str, default="RELIANCE.NS")
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    
    bt = SentimentBacktester(ticker=args.ticker, backtest_days=args.days)
    bt.run()
