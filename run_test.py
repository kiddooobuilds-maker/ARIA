import sys
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")

from ai_hedge_fund import Config, run_full_pipeline

def progress(msg):
    print(f"[PROGRESS] {msg}")

cfg = Config(ticker="RELIANCE.NS", epochs=2) # 2 epochs for speed
print("Starting pipeline...")
try:
    res = run_full_pipeline("RELIANCE.NS", cfg, progress_callback=progress)
    print("\n\nPIPELINE COMPLETE.")
except Exception as e:
    print(f"ERROR: {e}")
