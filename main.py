import os
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

from engines import analyze_stock, Signal, market_regime_engine
from data_feed import IDX_UNIVERSE, fetch_batch, fetch_ihsg, fetch_multi_timeframe
from notifier import TelegramNotifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
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
    logger.info(f"Notifier active: chat_id={TELEGRAM_CHAT_ID}")
else:
    logger.warning(f"Notifier inactive! TOKEN={'set' if TELEGRAM_TOKEN else 'MISSING'} CHAT_ID={'set' if TELEGRAM_CHAT_ID else 'MISSING'}")

scheduler = BackgroundScheduler()

def auto_scan():
    logger.info(f"Auto scan: {len(IDX_UNIVERSE)} symbols")
    try:
        candles_map = fetch_batch(IDX_UNIVERSE, "1d")
        for ticker, name in IDX_UNIVERSE.items():
            if ticker not in candles_map:
                continue
            result = analyze_stock(ticker, name, candles_map[ticker], CAPITAL, RISK_PER_TRADE)
            if notifier and result.signal != Signal.WAIT and result.score.total >= 70:
                notifier.send_signal(result)
    except Exception as e:
        logger.error(f"Auto scan error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(auto_scan, "interval", minutes=SCAN_INTERVAL)
    scheduler.start()
    logger.info("Scheduler started")
    yield
    scheduler.shutdown()

app = FastAPI(title="IHSG AI Trader", version="3.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"app": "IHSG AI Trader", "version": "3.0.0", "status": "online", "notifier": notifier is not None}

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
        "notifier_active": notifier is not None,
    }

@app.get("/test-telegram")
def test_telegram():
    if not notifier:
        token = os.environ.get("TELEGRAM_TOKEN", "MISSING")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "MISSING")
        return {"status": "error", "token": token[:10] if token != "MISSING" else "MISSING", "chat_id": chat_id}
    result = notifier.send("🧪 <b>TEST BERHASIL!</b>\nIHSG AI Trader aktif! ✅")
    return {"status": "ok" if result else "error", "sent": result}

@app.get("/ihsg")
def ihsg():
    candles = fetch_ihsg()
    from engines import detect_market_regime
    regime = market_regime_engine(candles)
    return {"regime": regime.regime.value, "price": regime.price, "ma20": regime.ma20, "ma50": regime.ma50, "dist_ma50": regime.dist_ma50, "timestamp": datetime.now().isoformat()}

@app.get("/scan")
def scan(signal: str = None, min_score: int = 70, timeframe: str = "1d", notify: bool = False):
    logger.info(f"Starting scan: {len(IDX_UNIVERSE)} symbols | TF={timeframe}")
    candles_map = fetch_batch(IDX_UNIVERSE, timeframe)
    logger.info(f"Batch complete: {len(candles_map)}/{len(IDX_UNIVERSE)} symbols")
    results = []
    for ticker, name in IDX_UNIVERSE.items():
        if ticker not in candles_map:
            continue
        result = analyze_stock(ticker, name, candles_map[ticker], CAPITAL, RISK_PER_TRADE)
        results.append(result)
    logger.info(f"Scan complete: {len(results)} results")
    filtered = [r for r in results if r.score.total >= min_score]
    if signal:
        filtered = [r for r in filtered if r.signal.value == signal.upper()]
    if notify and notifier:
        for r in filtered:
            if r.signal != Signal.WAIT:
                notifier.send_signal(r)
        if filtered:
            notifier.send_scan_summary(results)
    return {"scan_time": datetime.now().isoformat(), "total": len(results), "filtered": len(filtered), "results": [vars(r) for r in filtered]}

@app.get("/ranking")
def ranking(top_n: int = 10, min_score: int = 70):
    candles_map = fetch_batch(IDX_UNIVERSE, "1d")
    results = []
    for ticker, name in IDX_UNIVERSE.items():
        if ticker not in candles_map:
            continue
        result = analyze_stock(ticker, name, candles_map[ticker], CAPITAL, RISK_PER_TRADE)
        results.append(result)
    ranked = sorted(results, key=lambda r: r.score.total, reverse=True)[:top_n]
    return {"timestamp": datetime.now().isoformat(), "top_n": top_n, "results": [vars(r) for r in ranked]}

@app.post("/webhook/tradingview")
async def webhook(payload: dict):
    if payload.get("secret") != WEBHOOK_SECRET:
        return {"status": "error", "message": "Invalid secret"}
    ticker = payload.get("ticker", "").replace(".JK", "")
    name = IDX_UNIVERSE.get(ticker, ticker)
    from data_feed import fetch_candles
    candles = fetch_candles(ticker, "1d")
    if not candles:
        return {"status": "error", "message": "No data"}
    result = analyze_stock(ticker, name, candles, CAPITAL, RISK_PER_TRADE)
    if notifier and result.signal != Signal.WAIT and result.score.total >= 70:
        notifier.send_signal(result)
    return {"status": "ok", "signal": result.signal.value, "score": result.score.total}
