import os, logging
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
from engines import analyze_stock, Signal, market_regime_engine
from data_feed import IDX_UNIVERSE, fetch_batch, fetch_ihsg, fetch_candles
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

def auto_scan():
    try:
        candles_map = fetch_batch(IDX_UNIVERSE, "1d")
        for ticker in IDX_UNIVERSE:
            if ticker not in candles_map:
                continue
            result = analyze_stock(ticker, ticker, candles_map[ticker], CAPITAL, RISK_PER_TRADE)
            if notifier and result.signal != Signal.WAIT and result.score.total >= 70:
                notifier.send_signal(result)
    except Exception as e:
        logger.error(f"Auto scan error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler.add_job(auto_scan, "interval", minutes=SCAN_INTERVAL)
    scheduler.start()
    yield
    scheduler.shutdown()

app = FastAPI(title="IHSG AI Trader", version="4.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"app": "IHSG AI Trader", "status": "online", "notifier": notifier is not None}

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
    result = notifier.send("TEST BERHASIL! IHSG AI Trader aktif!")
    return {"status": "ok" if result else "error"}

@app.get("/ihsg")
def ihsg():
    candles = fetch_ihsg()
    regime = market_regime_engine(candles)
    return {"regime": regime.regime.value, "price": regime.price, "ma20": regime.ma20, "ma50": regime.ma50}

@app.get("/scan")
def scan(signal: str = None, min_score: int = 70, timeframe: str = "1d", notify: bool = False):
    candles_map = fetch_batch(IDX_UNIVERSE, timeframe)
    results = []
    for ticker in IDX_UNIVERSE:
        if ticker not in candles_map:
            continue
        result = analyze_stock(ticker, ticker, candles_map[ticker], CAPITAL, RISK_PER_TRADE)
        results.append(result)
    filtered = [r for r in results if r.score.total >= min_score]
    if signal:
        filtered = [r for r in filtered if r.signal.value == signal.upper()]
    if notify and notifier:
        for r in filtered:
            if r.signal != Signal.WAIT:
                notifier.send_signal(r)
        if results:
            notifier.send_scan_summary(results)
    return {
        "scan_time": datetime.now().isoformat(),
        "total": len(results),
        "filtered": len(filtered),
        "results": [{"ticker": r.ticker, "score": r.score.total, "signal": r.signal.value} for r in filtered]
    }

@app.get("/ranking")
def ranking(top_n: int = 10):
    candles_map = fetch_batch(IDX_UNIVERSE, "1d")
    results = []
    for ticker in IDX_UNIVERSE:
        if ticker not in candles_map:
            continue
        result = analyze_stock(ticker, ticker, candles_map[ticker], CAPITAL, RISK_PER_TRADE)
        results.append(result)
    ranked = sorted(results, key=lambda r: r.score.total, reverse=True)[:top_n]
    return {
        "timestamp": datetime.now().isoformat(),
        "results": [{"ticker": r.ticker, "score": r.score.total, "signal": r.signal.value} for r in ranked]
    }

@app.post("/webhook/tradingview")
async def webhook(payload: dict):
    if payload.get("secret") != WEBHOOK_SECRET:
        return {"status": "error", "message": "Invalid secret"}
    ticker = payload.get("ticker", "").replace(".JK", "")
    candles = fetch_candles(ticker, "1d")
    if not candles:
        return {"status": "error", "message": "No data"}
    result = analyze_stock(ticker, ticker, candles, CAPITAL, RISK_PER_TRADE)
    if notifier and result.signal != Signal.WAIT and result.score.total >= 70:
        notifier.send_signal(result)
    return {"status": "ok", "signal": result.signal.value, "score": result.score.total}
