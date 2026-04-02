import os
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

# Import our existing brain
from ai_hedge_fund import run_full_pipeline, Config, DataIngestor

console = Console()

# ── Configuration ──
INITIAL_CAPITAL = 10_000_000.0  # ₹ 1 Crore
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS",
    "ICICIBANK.NS", "ITC.NS", "LT.NS", "SBIN.NS",
    "BAJFINANCE.NS", "BHARTIARTL.NS"
]

# Portfolio construction rules
MAX_ACTIVE_POSITIONS = 4
MAX_GROSS_EXPOSURE   = 0.95
MAX_POSITION_WEIGHT  = 0.40
MIN_WEIGHT_CHANGE    = 0.02
ENTER_COMPOSITE      = 0.54
HOLD_COMPOSITE       = 0.50
ENTER_PRED_RETURN    = 0.10   # predicted next-day return in %
HOLD_PRED_RETURN     = -0.05
MIN_NEW_TREND_SCORE  = 1.05
MIN_HELD_TREND_SCORE = 0.60

SIGNAL_BONUS = {
    "STRONG_BUY": 1.00,
    "BUY": 0.45,
    "HOLD": 0.05,
    "SELL": -0.20,
    "STRONG_SELL": -0.40,
}

def progress(msg):
    # Suppress verbose pipeline logs so our console output stays clean
    pass


def signed_pct(value):
    return f"{value:+.2f}%"


def compute_trend_score(info):
    """Score price trend using medium-term moving-average structure."""
    price = float(info.get("price", 0.0) or 0.0)
    sma20 = float(info.get("sma20", 0.0) or 0.0)
    sma50 = float(info.get("sma50", 0.0) or 0.0)
    rsi   = float(info.get("rsi", 50.0) or 50.0)

    score = 0.0
    if sma20 > 0 and price > sma20:
        score += 0.45
    if sma50 > 0 and price > sma50:
        score += 0.65
    if sma20 > 0 and sma50 > 0 and sma20 > sma50:
        score += 0.60
    if rsi >= 55:
        score += 0.20
    elif rsi < 45:
        score -= 0.15
    return score


def compute_market_regime(day_data):
    """Estimate how healthy the basket is and scale gross exposure from it."""
    if not day_data:
        return 0.0

    bullish = 0.0
    for info in day_data.values():
        trend = compute_trend_score(info)
        pred_return = float(info.get("pred_return", 0.0) or 0.0)
        if trend >= MIN_NEW_TREND_SCORE and pred_return > 0.10:
            bullish += 1.0
        elif trend >= MIN_HELD_TREND_SCORE and pred_return > -0.05:
            bullish += 0.5

    breadth = bullish / len(day_data)
    return max(0.20, min(1.0, breadth))


def compute_conviction(signal, pred_return, composite, held=False):
    """Blend categorical signal strength with raw model conviction."""
    pred_gate = HOLD_PRED_RETURN if held else ENTER_PRED_RETURN
    comp_gate = HOLD_COMPOSITE if held else ENTER_COMPOSITE

    if signal in {"SELL", "STRONG_SELL"}:
        return 0.0
    if signal == "HOLD" and not held:
        return 0.0
    if pred_return < pred_gate and composite < comp_gate:
        return 0.0

    pred_edge = max(pred_return - pred_gate, 0.0)
    raw_pred_strength = max(pred_return, 0.0)
    comp_edge = max(composite - comp_gate, 0.0)
    return (
        SIGNAL_BONUS.get(signal, 0.0)
        + 1.20 * pred_edge
        + 0.45 * raw_pred_strength
        + 0.60 * comp_edge
    )


def build_target_weights(day_data, shares, portfolio_value, valuation_prices):
    """
    Build sticky, rank-based target weights from the strongest names only.
    Existing positions get a slightly lower bar to reduce unnecessary churn.
    """
    current_weights = {}
    if portfolio_value > 0:
        for ticker, qty in shares.items():
            if qty <= 0:
                continue
            price = valuation_prices.get(ticker)
            if price:
                current_weights[ticker] = (qty * price) / portfolio_value

    regime_strength = compute_market_regime(day_data)
    target_gross_exposure = min(MAX_GROSS_EXPOSURE, 0.15 + 0.80 * regime_strength)
    max_positions = 2 if regime_strength < 0.45 else MAX_ACTIVE_POSITIONS

    candidates = []
    for ticker, info in day_data.items():
        held = shares.get(ticker, 0) > 0
        trend_score = compute_trend_score(info)
        min_trend = MIN_HELD_TREND_SCORE if held else MIN_NEW_TREND_SCORE
        if trend_score < min_trend:
            continue
        conviction = compute_conviction(
            info["signal"],
            info["pred_return"],
            info["composite"],
            held=held,
        )
        if conviction > 0:
            candidates.append((ticker, conviction + 0.15 * trend_score))

    if not candidates:
        return {}

    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)[:max_positions]
    total_score = sum(score for _, score in ranked)
    if total_score <= 0:
        return {}

    raw_weights = {
        ticker: min(MAX_POSITION_WEIGHT, target_gross_exposure * (score / total_score))
        for ticker, score in ranked
    }

    allocated = sum(raw_weights.values())
    if allocated < target_gross_exposure:
        for ticker, _ in ranked:
            room = MAX_POSITION_WEIGHT - raw_weights[ticker]
            if room <= 0:
                continue
            add = min(room, target_gross_exposure - allocated)
            raw_weights[ticker] += add
            allocated += add
            if allocated >= target_gross_exposure - 1e-9:
                break

    target_weights = {}
    for ticker, weight in raw_weights.items():
        current_weight = current_weights.get(ticker, 0.0)
        if abs(weight - current_weight) < MIN_WEIGHT_CHANGE:
            target_weights[ticker] = current_weight
        else:
            target_weights[ticker] = weight
    return target_weights

def run_portfolio_backtest():
    console.print(f"[bold cyan]\n🚀 Initiating Global Portfolio Manager[/bold cyan]")
    console.print(f"Pool: [green]₹{INITIAL_CAPITAL:,.2f}[/green] | Tickers: [yellow]{len(TICKERS)}[/yellow]\n")

    cfg = Config()
    cfg.force_retrain = False     # Use cached models and data
    cfg.use_finbert   = False     # Fast sentiment for backtesting speed

    # 1. Gather all historical predictions and price data
    db = {}
    all_dates = set()

    for ticker in TICKERS:
        console.print(f"Evaluating [bold]{ticker}[/bold]...")
        try:
            # We redirect stdout so we don't spam the console
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            res = run_full_pipeline(ticker, cfg=cfg, progress_callback=progress)
            sys.stdout.close()
            sys.stdout = original_stdout
            
            cdata = res["chart_data"]
            safe = ticker.replace(".", "_").replace("/", "_")
            feature_df = pd.read_csv(os.path.join("cache", f"data_{safe}.csv"))
            feature_df["Date"] = pd.to_datetime(feature_df["Date"]).dt.strftime("%Y-%m-%d")
            feature_lookup = (
                feature_df[["Date", "SMA_20", "SMA_50", "RSI_14"]]
                .drop_duplicates(subset=["Date"])
                .set_index("Date")
                .to_dict("index")
            )
            db[ticker] = {
                "dates": cdata["dates"],
                "prices": cdata["actual_close"],
                "signals": cdata["signals"],
                "pred_return": cdata["pred_return"],
                "composite": cdata["composite"],
                "features": feature_lookup,
            }
            all_dates.update(cdata["dates"])
        except Exception as e:
            sys.stdout = sys.__stdout__
            console.print(f"[red]Failed to evaluate {ticker}: {e}[/red]")

    # Sort all historical dates chronologically
    sorted_dates = sorted(list(all_dates))
    if not sorted_dates:
        console.print("[red]No data found to backtest![/red]")
        return

    benchmark_cfg = Config(
        start_date=sorted_dates[0],
        end_date=sorted_dates[-1],
    )
    benchmark_series = DataIngestor(benchmark_cfg).fetch_benchmark()
    benchmark_lookup = {}
    if benchmark_series is not None and not benchmark_series.empty:
        benchmark_lookup = {
            d.strftime("%Y-%m-%d"): float(v)
            for d, v in benchmark_series.items()
        }
        console.print("[green]Loaded real Nifty 50 (^NSEI) benchmark.[/green]")
    else:
        console.print("[yellow]Could not load real Nifty benchmark; skipping benchmark comparison.[/yellow]")

    # 2. Chunking into 4 periods
    chunk_size = len(sorted_dates) // 4
    for i in range(4):
        console.print(f"\n[bold magenta]=== RUNNING TEST {i+1}/4 ===[/bold magenta]")
        start_idx = i * chunk_size
        end_idx   = (i + 1) * chunk_size if i < 3 else len(sorted_dates)
        chunk_dates = sorted_dates[start_idx:end_idx]

        cash   = INITIAL_CAPITAL
        shares = {t: 0 for t in TICKERS}
        
        portfolio_history = []
        benchmark_history = []
        last_prices = {}
        
        benchmark_values = []

        # Create lookup for the chunk
        data_lookup = {t: {} for t in TICKERS}
        for t in db:
            for j, d in enumerate(db[t]["dates"]):
                if d in chunk_dates:
                    data_lookup[t][d] = {
                        "price": db[t]["prices"][j],
                        "signal": db[t]["signals"][j],
                        "pred_return": db[t]["pred_return"][j],
                        "composite": db[t]["composite"][j],
                        "sma20": db[t]["features"].get(d, {}).get("SMA_20"),
                        "sma50": db[t]["features"].get(d, {}).get("SMA_50"),
                        "rsi": db[t]["features"].get(d, {}).get("RSI_14"),
                    }

        # Initialize first prices for portfolio valuation continuity
        first_prices = {}
        for t in TICKERS:
            for d in chunk_dates:
                if d in data_lookup[t]:
                    first_prices[t] = data_lookup[t][d]["price"]
                    break

        benchmark_chunk = []
        for d in chunk_dates:
            if d in benchmark_lookup:
                benchmark_chunk.append((d, benchmark_lookup[d]))

        benchmark_base = benchmark_chunk[0][1] if benchmark_chunk else None

        for d in chunk_dates:
            current_prices = {}
            day_data = {}

            # Scan market today
            for t in TICKERS:
                if d in data_lookup[t]:
                    current_prices[t] = data_lookup[t][d]["price"]
                    last_prices[t] = current_prices[t]
                    day_data[t] = data_lookup[t][d]

            if not current_prices:
                continue

            # Mark-to-market current portfolio value
            valuation_prices = {t: last_prices.get(t, first_prices.get(t, 0.0)) for t in TICKERS}
            port_val = cash + sum(shares[t] * valuation_prices.get(t, 0.0) for t in TICKERS)
            bench_price = benchmark_lookup.get(d)
            bench_val = None
            if benchmark_base and bench_price is not None:
                bench_val = INITIAL_CAPITAL * (bench_price / benchmark_base)
            target_weights = build_target_weights(day_data, shares, port_val, valuation_prices)

            # Sell first to free cash, then buy the highest-conviction names.
            execution_order = sorted(
                current_prices.items(),
                key=lambda item: target_weights.get(item[0], 0.0),
            )
            for t, price in execution_order:
                current_qty = shares[t]
                target_cap = port_val * target_weights.get(t, 0.0)
                target_qty = math.floor(target_cap / price) if price > 0 else 0
                if target_qty >= current_qty:
                    continue
                cash += (current_qty - target_qty) * price
                shares[t] = target_qty

            for t, price in reversed(execution_order):
                current_qty = shares[t]
                target_cap = port_val * target_weights.get(t, 0.0)
                target_qty = math.floor(target_cap / price) if price > 0 else 0
                if target_qty <= current_qty:
                    continue
                affordable_qty = math.floor(cash / price) if price > 0 else 0
                buy_qty = min(target_qty - current_qty, affordable_qty)
                if buy_qty <= 0:
                    continue
                cash -= buy_qty * price
                shares[t] = current_qty + buy_qty

            # Record today's final value
            final_val = cash + sum(shares[t] * valuation_prices.get(t, 0.0) for t in TICKERS)
            portfolio_history.append((d, final_val))
            if bench_val is not None:
                benchmark_history.append((d, bench_val))

        # 3. Final Evaluation for this chunk
        if not portfolio_history:
            continue
            
        final_eq = pd.Series([val for (d, val) in portfolio_history])
        bench_eq = pd.Series([val for (d, val) in benchmark_history]) if benchmark_history else pd.Series(dtype=float)
        dates_ts = pd.to_datetime([d for (d, val) in portfolio_history])

        total_ret  = (final_eq.iloc[-1] / INITIAL_CAPITAL - 1) * 100
        bench_ret  = (bench_eq.iloc[-1] / INITIAL_CAPITAL - 1) * 100 if not bench_eq.empty else 0.0
        alpha      = total_ret - bench_ret
        max_dd     = ((final_eq - final_eq.cummax()) / final_eq.cummax()).min() * 100
        daily_rets = final_eq.pct_change().dropna()
        sharpe     = (daily_rets.mean() / (daily_rets.std() + 1e-9)) * math.sqrt(252)

        start_date_str = pd.to_datetime(chunk_dates[0]).strftime("%d %b %Y")
        end_date_str   = pd.to_datetime(chunk_dates[-1]).strftime("%d %b %Y")
        plot_name      = f"cache/portfolio_test_period_{i+1}.png"
        
        benchmark_dates_ts = pd.to_datetime([d for (d, val) in benchmark_history]) if benchmark_history else None
        plot_portfolio(
            dates_ts, final_eq, bench_eq, total_ret, sharpe, max_dd, bench_ret,
            i+1, plot_name, start_date_str, end_date_str, benchmark_dates_ts
        )

        table = Table(title=f"💎 Period {i+1}: {start_date_str} to {end_date_str}")
        table.add_column("Metric", style="cyan")
        table.add_column("Result", style="green")
        table.add_row("Total Return", signed_pct(total_ret))
        table.add_row("Benchmark Return", signed_pct(bench_ret))
        table.add_row("Alpha", signed_pct(alpha))
        table.add_row("Max Drawdown", signed_pct(max_dd))
        table.add_row("Sharpe Ratio", f"{sharpe:.2f}")

        console.print(table)
        console.print(f"[green]✅ Saved backtest chart to {plot_name}[/green]")


def plot_portfolio(dates, port_eq, bench_eq, ret, sharpe, draw, benchmark_return, period_num, plot_name, start_date_str, end_date_str, benchmark_dates=None):
    """Generates the main aesthetic dashboard chart for the Portfolio"""
    fig = plt.figure(figsize=(12, 6), facecolor="#0d1117")
    ax  = fig.add_subplot(111)
    ax.set_facecolor("#0d1117")

    # Plot Equity Curves
    ax.plot(dates, port_eq, color="#00ff88", lw=2.5, label=f"AI Fund ({ret:+.1f}%)")
    if bench_eq is not None and len(bench_eq) > 0:
        bench_x = benchmark_dates if benchmark_dates is not None else dates
        ax.plot(bench_x, bench_eq, color="#ffa500", lw=1.5, ls="--", alpha=0.8, label=f"Nifty 50 ({benchmark_return:+.1f}%)")

    # Fill underneath the AI Fund for aesthetic
    ax.fill_between(dates, port_eq, port_eq.min(), color="#00ff88", alpha=0.1)

    # Style
    ax.set_title(f"Period {period_num} ({start_date_str} - {end_date_str}) | Sharpe {sharpe:.2f} | MaxDD {draw:.1f}%", color="#fff", pad=20, fontsize=14)
    ax.tick_params(colors="#aaaaaa", labelsize=10)
    for spine in ax.spines.values(): spine.set_color("#333333")
    
    # Format Y axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"₹{x/1000000:.1f}Cr"))
    
    ax.legend(facecolor="#1a1a2e", edgecolor="#333", labelcolor="#fff")
    
    plt.tight_layout()
    plt.savefig(plot_name, dpi=120)
    plt.close()


if __name__ == "__main__":
    run_portfolio_backtest()
