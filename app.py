"""
=============================================================================
AI HEDGE FUND — FLASK API SERVER  (rebuilt fresh)
=============================================================================
File        : app.py
Description : REST API that exposes the ML pipeline and ARIA chatbot
              for the React dashboard.

Architecture
------------
  • Background threads run heavy ML work.
  • SSE streams send **only** lightweight progress updates.
  • The full result is fetched by the browser via GET /api/result/<job_id>.
=============================================================================
"""

# ── stdlib ────────────────────────────────────────────────────────────────
import json, math, os, queue, threading, time, uuid, logging, re
from typing import Dict, Optional

# ── third-party ───────────────────────────────────────────────────────────
import requests as http_requests          # renamed to avoid Flask clash
from flask import (Flask, Response, jsonify, request,
                   send_from_directory, stream_with_context)
from flask_cors import CORS

# ── project ───────────────────────────────────────────────────────────────
from ai_hedge_fund import (Config, SignalGenerator,
                           run_full_pipeline)

# ── logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


# =========================================================================
# SAFE JSON ENCODER — handles NaN, Inf, Pandas, Numpy
# =========================================================================
class SafeJSONEncoder(json.JSONEncoder):
    """Encode any Python/Pandas/Numpy object into JSON-safe primitives."""
    def default(self, o):
        if hasattr(o, 'tolist'):      return o.tolist()
        if hasattr(o, 'isoformat'):   return o.isoformat()
        if hasattr(o, 'item'):        return o.item()
        if isinstance(o, float):
            if math.isnan(o) or math.isinf(o):
                return None
        return super().default(o)


def safe_dumps(obj):
    """Convenience wrapper for json.dumps with SafeJSONEncoder."""
    return json.dumps(obj, cls=SafeJSONEncoder)


# =========================================================================
# FLASK APP
# =========================================================================
app = Flask(__name__)
CORS(app)

# ── in-memory job store ──────────────────────────────────────────────────
JOBS: Dict[str, Dict] = {}

# ── Ollama config ────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "phi3:latest"


# =========================================================================
# SERVE FRONTEND
# =========================================================================
@app.route("/")
def root():
    return send_from_directory(os.path.dirname(__file__), "index.html")


# =========================================================================
# HEALTH CHECK
# =========================================================================
@app.route("/api/health")
def health():
    ollama_ok = False
    try:
        r = http_requests.get("http://localhost:11434/api/tags", timeout=2)
        ollama_ok = r.status_code == 200
    except Exception:
        pass
    return jsonify({
        "status":       "ok",
        "ollama":       "connected" if ollama_ok else "not running",
        "ollama_model": OLLAMA_MODEL,
    })


# =========================================================================
# HELPER — run a function in a background thread with SSE progress
# =========================================================================
def _create_job(worker_fn):
    """
    Creates a job entry, spawns a daemon thread that calls worker_fn(q),
    and returns the job_id.  worker_fn should:
      1. Push {"type":"progress","message":"..."} to q.
      2. Return the final result dict.
    On success, job["result"] is stored and q gets {"type":"done","job_id":...}.
    On error, q gets {"type":"error","message":"..."}.
    """
    job_id = str(uuid.uuid4())[:8]
    q      = queue.Queue()
    JOBS[job_id] = {"status": "running", "result": None, "queue": q}

    def _run():
        try:
            result = worker_fn(q)
            JOBS[job_id]["result"] = result
            JOBS[job_id]["status"] = "done"
            q.put({"type": "done", "job_id": job_id})
        except Exception as e:
            log.error(f"Job {job_id} error: {e}", exc_info=True)
            JOBS[job_id]["status"] = "error"
            q.put({"type": "error", "message": str(e)})

    threading.Thread(target=_run, daemon=True).start()
    return job_id


# =========================================================================
# ANALYSE ENDPOINT — single-stock ML pipeline
# =========================================================================
@app.route("/api/analyse", methods=["POST"])
def analyse():
    data   = request.get_json(silent=True) or {}
    ticker = data.get("ticker", "RELIANCE.NS").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400

    cfg = Config(
        ticker     = ticker,
        start_date = data.get("start_date", "2020-01-01"),
        end_date   = data.get("end_date",   Config().end_date),
    )

    def worker(q):
        def push(msg):
            q.put({"type": "progress", "message": msg})
        return run_full_pipeline(ticker, cfg, progress_callback=push)

    job_id = _create_job(worker)
    return jsonify({"job_id": job_id, "ticker": ticker}), 202




# =========================================================================
# RESULT ENDPOINT — browser fetches the full result after "done"
# =========================================================================
@app.route("/api/result/<job_id>")
def get_result(job_id: str):
    if job_id not in JOBS:
        return jsonify({"error": "job not found"}), 404
    job = JOBS[job_id]
    if job["status"] != "done":
        return jsonify({"error": "not ready", "status": job["status"]}), 202
    return Response(safe_dumps(job["result"]), mimetype="application/json")


# =========================================================================
# SSE STATUS STREAM — lightweight progress only, never large data
# =========================================================================
@app.route("/api/status/<job_id>")
def status_stream(job_id: str):
    if job_id not in JOBS:
        return jsonify({"error": "job not found"}), 404

    def generate():
        q = JOBS[job_id]["queue"]
        while True:
            try:
                event = q.get(timeout=15)
                yield f"data: {safe_dumps(event)}\n\n"
                if event["type"] in ("done", "error"):
                    break
            except queue.Empty:
                yield f"data: {safe_dumps({'type': 'heartbeat'})}\n\n"
            except Exception as exc:
                log.error(f"SSE error: {exc}")
                break

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# =========================================================================
# CHATBOT — Rule-based + Ollama fallback
# =========================================================================
class RuleBasedChatbot:
    def answer(self, message: str, context: dict) -> Optional[str]:
        if not context:
            return ("No analysis has been run yet. Please enter a ticker "
                    "(e.g. RELIANCE.NS) and click 'Run Analysis' first.")
        msg     = message.lower().strip()
        sig     = context.get("latest_signal", {})
        metrics = context.get("metrics", {})
        t_info  = context.get("training_info", {})
        trades  = context.get("trade_log", [])
        ticker  = context.get("ticker", "Unknown")

        if any(k in msg for k in ["why","reason","explain signal","basis"]):
            return self._explain_signal(sig, ticker)
        if any(k in msg for k in ["current signal","latest signal","what signal","what's the call"]):
            return self._current_signal(sig, ticker)
        if any(k in msg for k in ["sentiment","news","finbert","vader"]):
            return self._explain_sentiment(sig, ticker)
        if any(k in msg for k in ["performance","return","profit","loss","capital","portfolio"]):
            return self._explain_performance(metrics, ticker)
        if "sharpe" in msg:
            return self._explain_sharpe(metrics)
        if "drawdown" in msg or "draw down" in msg:
            return self._explain_drawdown(metrics)
        if "win rate" in msg or "winrate" in msg:
            return self._explain_winrate(metrics)
        if any(k in msg for k in ["best trade","worst trade","biggest win","biggest loss"]):
            return self._explain_trades(trades)
        if any(k in msg for k in ["model","lstm","accuracy","training","epoch","parameters"]):
            return self._explain_model(t_info)
        if any(k in msg for k in ["risk","danger","careful","warning"]):
            return self._explain_risks(metrics, sig)
        if any(k in msg for k in ["predict","forecast","tomorrow","next","future"]):
            return self._explain_prediction(sig, ticker)
        if any(k in msg for k in ["hello","hi ","hey","help"]):
            return (f"Hi! I'm ARIA — your AI Research & Investment Analyst.\n\n"
                    f"I have the full analysis for {ticker} loaded. "
                    f"You can ask me things like:\n"
                    f"• Why was this call given?\n"
                    f"• Explain the sentiment score\n"
                    f"• What was our best trade?\n"
                    f"• Is the Sharpe ratio good?\n"
                    f"• What are the risks?\n"
                    f"• How accurate is the LSTM model?")
        return None

    # ── helpers ──────────────────────────────────────────────────────────
    def _explain_signal(self, sig, ticker):
        signal    = sig.get("signal","HOLD")
        composite = sig.get("composite",0.5)
        ps = sig.get("price_score",0.5); ss = sig.get("sent_score",0.5); ms = sig.get("mom_score",0.5)
        pd = "bullish" if ps>0.5 else ("bearish" if ps<0.5 else "neutral")
        sd = "positive" if ss>0.55 else ("negative" if ss<0.45 else "neutral")
        md = "bullish" if ms>0.55 else ("bearish" if ms<0.45 else "neutral")
        return (f"📊 Signal Breakdown for {ticker}: {signal}\n\n"
                f"1️⃣ LSTM Price (50%): {ps:.2f} — {pd}\n"
                f"2️⃣ Sentiment (30%): {ss:.2f} — {sd}\n"
                f"3️⃣ Momentum  (20%): {ms:.2f} — {md}\n\n"
                f"Composite: {composite:.4f}")
    def _current_signal(self, sig, ticker):
        return (f"📈 {ticker}: {sig.get('signal','HOLD')} | "
                f"Date: {sig.get('date','N/A')} | Composite: {sig.get('composite',0.5):.4f}")
    def _explain_sentiment(self, sig, ticker):
        ss = sig.get("sent_score",0.5); raw = (ss-0.5)*2
        q = "positive" if raw>0.1 else ("negative" if raw<-0.1 else "neutral")
        return (f"🧠 Sentiment for {ticker}: {ss:.4f} ({q})\n"
                f"FinBERT/VADER news scoring → 30% of final signal")
    def _explain_performance(self, m, ticker):
        return (f"💰 {ticker} Return: {m.get('total_return_pct',0):+.2f}%\n"
                f"Final Capital: ₹{m.get('final_capital',100000):,.2f}\n"
                f"Sharpe: {m.get('sharpe','N/A')} | MaxDD: {m.get('max_drawdown_pct','N/A')}%\n"
                f"Win Rate: {m.get('win_rate_pct','N/A')}% | Trades: {m.get('num_trades','N/A')}")
    def _explain_sharpe(self, m):
        s = m.get("sharpe",0)
        q = "excellent" if s>2 else ("good" if s>1 else ("acceptable" if s>0.5 else "poor"))
        return f"📐 Sharpe Ratio: {s:.2f} — {q}"
    def _explain_drawdown(self, m):
        return f"📉 Max Drawdown: {m.get('max_drawdown_pct',0):.2f}%"
    def _explain_winrate(self, m):
        return f"🎯 Win Rate: {m.get('win_rate_pct',0):.1f}% across {m.get('num_trades',0)} trades"
    def _explain_trades(self, trades):
        sells = [t for t in trades if t.get("pnl") and t["pnl"]!=0]
        if not sells: return "No completed trades with P&L data."
        best  = max(sells, key=lambda t:t.get("pnl",0))
        worst = min(sells, key=lambda t:t.get("pnl",0))
        return (f"📋 Best: {best.get('date','?')} +₹{best.get('pnl',0):.2f}\n"
                f"   Worst: {worst.get('date','?')} ₹{worst.get('pnl',0):.2f}")
    def _explain_model(self, t):
        return (f"🤖 3-layer Bidirectional LSTM\n"
                f"Params: {t.get('parameters',0):,} | Best Epoch: {t.get('best_epoch','N/A')}\n"
                f"Val Loss: {t.get('best_val_loss','N/A')}")
    def _explain_risks(self, m, sig):
        risks = []
        if abs(m.get("max_drawdown_pct",0))>20: risks.append("⚠️ High drawdown")
        if m.get("win_rate_pct",0)<50:          risks.append("⚠️ Win rate <50%")
        if m.get("sharpe",0)<0.5:               risks.append("⚠️ Low Sharpe ratio")
        risks.append("⚠️ Past performance ≠ future results")
        return "🚨 Risk Assessment\n" + "\n".join(risks)
    def _explain_prediction(self, sig, ticker):
        pred = sig.get("pred_close",0); actual = sig.get("actual_close",0)
        diff = ((pred-actual)/actual*100) if actual else 0
        return f"🔮 {ticker}: Predicted ₹{pred:,.2f} vs Actual ₹{actual:,.2f} ({diff:+.2f}%)"


rule_bot = RuleBasedChatbot()


@app.route("/api/chat", methods=["POST"])
def chat():
    data    = request.get_json(silent=True) or {}
    message = data.get("message","").strip()
    history = data.get("history",[])
    context = data.get("context",{})
    if not message:
        return jsonify({"error":"message is required"}), 400

    rule_reply = rule_bot.answer(message, context)
    if rule_reply:
        return jsonify({"reply": rule_reply, "source": "rules"})

    try:
        return _ollama_chat(message, history, context)
    except http_requests.exceptions.ConnectionError:
        return jsonify({
            "reply": ("I don't have a specific answer for that.\n\n"
                      "Try: Why was this call given? / Explain sentiment / "
                      "Best trade? / Sharpe ratio? / Risks?\n\n"
                      "For advanced Q&A install Ollama: ollama.com"),
            "source": "fallback",
        })
    except Exception as e:
        log.error(f"Chat error: {e}")
        return jsonify({"reply": f"Error: {e}", "source": "error"}), 500


def _ollama_chat(message, history, context):
    system_prompt = _build_system_prompt(context)
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history[-10:]:
        if turn.get("role") in ("user","assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": message})
    resp = http_requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL, "messages": messages,
        "stream": False, "options": {"temperature":0.3,"num_ctx":2048},
    }, timeout=120)
    resp.raise_for_status()
    return jsonify({"reply": resp.json()["message"]["content"], "source": "ollama"})


def _build_system_prompt(ctx):
    base = ("You are ARIA — AI Research & Investment Analyst for an AI hedge fund. "
            "Be precise, data-driven, concise.\n\n")
    if not ctx:
        return base + "No analysis loaded yet."
    sig = ctx.get("latest_signal",{}); m = ctx.get("metrics",{}); t = ctx.get("training_info",{})
    trades = ctx.get("trade_log",[])
    recent = ""
    if trades:
        lines = [f"  {tr.get('date','')} {tr.get('action','')} P:{tr.get('price','?')} PnL:{tr.get('pnl','N/A')}"
                 for tr in trades[-5:]]
        recent = "\nRecent trades:\n" + "\n".join(lines)
    return base + f"""
Ticker: {ctx.get('ticker','?')}  Signal: {sig.get('signal','?')}  Composite: {sig.get('composite','?')}
Price Score: {sig.get('price_score','?')}  Sentiment: {sig.get('sent_score','?')}  Momentum: {sig.get('mom_score','?')}
Return: {m.get('total_return_pct','?')}%  Sharpe: {m.get('sharpe','?')}  MaxDD: {m.get('max_drawdown_pct','?')}%
Win Rate: {m.get('win_rate_pct','?')}%  Trades: {m.get('num_trades','?')}{recent}
"""


# =========================================================================
# MAIN
# =========================================================================
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  AI HEDGE FUND — API Server")
    print("=" * 55)
    print("  API  : http://localhost:5000")
    print("  Chat : Rule-based + Ollama (if available)")
    print("  Open : http://localhost:5000 in your browser")
    print("=" * 55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True,
            use_reloader=True)