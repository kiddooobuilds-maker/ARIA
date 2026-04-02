import os
import io
import sys
from ai_hedge_fund import run_full_pipeline, Config

def progress(msg): pass # ignore progress bars to keep output clean

try:
    cfg = Config(force_retrain=True)
    res = run_full_pipeline("RELIANCE.NS", cfg=cfg, progress_callback=progress)
    m = res["metrics"]
    print("=== BASELINE ===")
    print(f"Total Return: {m['total_return_pct']}%")
    print(f"Sharpe Ratio: {m['sharpe']}")
    print(f"Alpha: {m['alpha_pct']}%")
    print(f"Max Drawdown: {m['max_drawdown_pct']}%")
    print(f"Win Rate: {m['win_rate_pct']}%")
    print("================")
except Exception as e:
    import traceback
    traceback.print_exc()
