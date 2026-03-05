import os, logging
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from engines import analyze_stock, Signal, market_regime_engine
from data_feed_tv import IDX_UNIVERSE, fetch_batch, fetch_ihsg, fetch_candles
from screener import run_screener_with_fallback
from notifier import TelegramNotifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ihsg_trader")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
CAPITAL = float(os.environ.get("CAPITAL", "100000000"))
RISK_PER_TRADE = float(os.environ.get("RISK_PER_TRADE", "0.015"))
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "rahasiatrader123")
SCAN_INTERVAL = int(os.environ.get("SCAN_INTERVAL_MINUTES", "15"))

notifier = None
if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
    notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    logger.info(f"Notifier active: {TELEGRAM_CHAT_ID}")
else:
    logger.warning("Notifier inactive")

scheduler = BackgroundScheduler()

# ── Simpan hasil scan terakhir ─────────────────────────────────────────────────
last_scan_results = []
last_scan_time    = None
scan_running      = False
_file_results     = []
_file_scan_time   = None

scan_status_state = {
    "status":       "idle",
    "started_at":   None,
    "finished_at":  None,
    "progress":     0,
    "total":        0,
    "current_stock": None,
    "error":        None,
}

import json
from pathlib import Path
RESULTS_FILE = Path("/app/scan_results.json")

def save_results_to_file(results):
    try:
        data = {
            "scan_time": datetime.now().isoformat(),
            "total": len(results),
            "results": [_serialize_result(r) for r in results]
        }
        RESULTS_FILE.write_text(json.dumps(data, indent=2))
        logger.info(f"Results saved to scan_results.json ({len(results)} saham)")
    except Exception as e:
        logger.error(f"Save results error: {e}")

def _serialize_result(r):
    return {
        "ticker":    r.ticker,
        "signal":    r.signal.value,
        "score":     r.score.total,
        "price":     r.price,
        "entry":     r.risk.entry,
        "stop_loss": r.risk.stop_loss,
        "tp1":       r.risk.tp1,
        "tp2":       r.risk.tp2,
        "rr1":       r.risk.rr1,
        "regime":    r.regime.value if hasattr(r.regime, 'value') else str(r.regime),
        "trend":     r.trend.value if hasattr(r.trend, 'value') else str(r.trend),
        "rel_vol":   r.rel_vol,
        "pattern":   r.pattern,
        "mtf":       r.mtf,
        "scanned_at": datetime.now().isoformat(),
    }

def load_results_from_file():
    try:
        if not RESULTS_FILE.exists():
            return [], None
        data = json.loads(RESULTS_FILE.read_text())
        return data.get("results", []), datetime.fromisoformat(data["scan_time"])
    except Exception as e:
        logger.error(f"Load results error: {e}")
        return [], None

def load_results():
    if last_scan_results:
        return last_scan_results, last_scan_time
    return _file_results, _file_scan_time

def run_scan_job(min_score: int = 70, notify: bool = True, use_screener: bool = True):
    global last_scan_results, last_scan_time, scan_running
    global scan_status_state, _file_results, _file_scan_time

    if scan_running:
        logger.info("Scan already running, skip")
        return

    scan_running = True
    scan_status_state["status"]     = "running"
    scan_status_state["started_at"] = datetime.now().isoformat()
    scan_status_state["finished_at"] = None
    scan_status_state["progress"]   = 0
    scan_status_state["error"]      = None

    try:
        # ── Step 1: Screener dari TradingView ─────────────────────────────
        if use_screener:
            scan_status_state["current_stock"] = "TV SCREENING..."
            from data_feed_tv import IDX_UNIVERSE
            scan_universe = run_screener_with_fallback(fallback_universe=IDX_UNIVERSE)
        else:
            from data_feed_tv import IDX_UNIVERSE
            scan_universe = IDX_UNIVERSE

        scan_status_state["total"] = len(scan_universe)
        logger.info(f"Scan start: {len(scan_universe)} symbols")

        # ── Step 2: Fetch IHSG ─────────────────────────────────────────────
        from data_feed import fetch_ihsg as fetch_ihsg_yf
        ihsg_candles = fetch_ihsg_yf()
        if not ihsg_candles:
            logger.error("IHSG data kosong, skip scan")
            scan_status_state["status"] = "error"
            scan_status_state["error"]  = "IHSG data empty"
            scan_running = False
            return

        # ── Step 3: Fetch batch OHLCV dari TradingView ────────────────────
        candles_map = fetch_batch(scan_universe, "1D")

        # ── Step 4: AI Analysis per saham ─────────────────────────────────
        results = []
        for i, ticker in enumerate(scan_universe):
            scan_status_state["progress"]      = i + 1
            scan_status_state["current_stock"] = ticker
            if ticker not in candles_map:
                continue
            result = analyze_stock(ticker, ticker, candles_map[ticker], ihsg_candles, CAPITAL)
            if result:
                results.append(result)

        logger.info(f"Scan complete: {len(results)} results")
        last_scan_results = results
        last_scan_time    = datetime.now()
        _file_results, _file_scan_time = results, last_scan_time
        save_results_to_file(results)

        scan_status_state["status"]      = "done"
        scan_status_state["finished_at"] = datetime.now().isoformat()
        scan_status_state["current_stock"] = None

        # ── Step 5: Kirim notifikasi Telegram ─────────────────────────────
        if notify and notifier:
            filtered = [r for r in results if r.score.total >= min_score and r.signal != Signal.WAIT]
            for r in filtered:
                notifier.send_signal(r)
            if results:
                notifier.send_scan_summary(results)

    except Exception as e:
        logger.error(f"Scan error: {e}")
        scan_status_state["status"] = "error"
        scan_status_state["error"]  = str(e)
    finally:
        scan_running = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(run_scan_job, "interval", minutes=SCAN_INTERVAL, kwargs={"notify": True})
    scheduler.start()
    logger.info("Scheduler started")
    yield
    scheduler.shutdown()

app = FastAPI(title="IHSG AI Trader", version="6.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"app": "IHSG AI Trader", "version": "6.0.0", "status": "online", "notifier": notifier is not None}

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}

@app.get("/debug-env")
def debug_env():
    token = os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    return {
        "token_exists": bool(token),
        "token_preview": token[:15] + "..." if len(token) > 15 else token,
        "chat_id": chat_id,
        "notifier_active": notifier is not None
    }

@app.get("/test-telegram")
def test_telegram():
    if not notifier:
        return {"status": "error", "message": "Notifier tidak aktif"}
    result = notifier.send("TEST BERHASIL! IHSG AI Trader v6 aktif!")
    return {"status": "ok" if result else "error"}

@app.get("/ihsg")
def ihsg():
    candles = fetch_ihsg()
    regime = market_regime_engine(candles)
    return {"regime": regime.regime.value, "price": regime.price, "ma20": regime.ma20, "ma50": regime.ma50}

@app.get("/scan")
def scan(background_tasks: BackgroundTasks, min_score: int = 70, notify: bool = False):
    global scan_running, last_scan_results, last_scan_time
    if scan_running:
        return {"status": "running", "message": "Scan sedang berjalan, cek /results"}
    background_tasks.add_task(run_scan_job, min_score, notify)
    return {
        "status": "started",
        "message": f"Scan {len(IDX_UNIVERSE)} saham dimulai di background. Cek /results dalam 2-3 menit."
    }

@app.get("/results")
def results(min_score: int = 70, signal: str = None):
    if not last_scan_results:
        return {"status": "no_data", "message": "Belum ada hasil scan. Akses /scan dulu."}
    filtered = [r for r in last_scan_results if r.score.total >= min_score]
    if signal:
        filtered = [r for r in filtered if r.signal.value == signal.upper()]
    ranked = sorted(filtered, key=lambda r: r.score.total, reverse=True)
    return {
        "scan_time": last_scan_time.isoformat() if last_scan_time else None,
        "total_scanned": len(last_scan_results),
        "filtered": len(filtered),
        "scan_running": scan_running,
        "results": [{"ticker": r.ticker, "score": r.score.total, "signal": r.signal.value, "entry": r.risk.entry} for r in ranked]
    }

@app.get("/ranking")
def ranking(top_n: int = 10):
    if not last_scan_results:
        return {"status": "no_data", "message": "Belum ada hasil scan. Akses /scan dulu."}
    ranked = sorted(last_scan_results, key=lambda r: r.score.total, reverse=True)[:top_n]
    return {
        "timestamp": last_scan_time.isoformat() if last_scan_time else None,
        "results": [{"ticker": r.ticker, "score": r.score.total, "signal": r.signal.value} for r in ranked]
    }

@app.post("/webhook/tradingview")
async def webhook(payload: dict):
    if payload.get("secret") != WEBHOOK_SECRET:
        return {"status": "error", "message": "Invalid secret"}
    ticker = payload.get("ticker", "").replace(".JK", "")
    candles = fetch_candles(ticker, "1D")
    ihsg_candles = fetch_ihsg()
    if not candles:
        return {"status": "error", "message": "No data"}
    result = analyze_stock(ticker, ticker, candles, ihsg_candles, CAPITAL)
    if result and notifier and result.signal != Signal.WAIT and result.score.total >= 70:
        notifier.send_signal(result)
    return {"status": "ok", "signal": result.signal.value if result else "WAIT"}
