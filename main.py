"""
═══════════════════════════════════════════════════════
 IHSG AI TRADER — FASTAPI BACKEND SERVER
 
 Endpoints:
   GET  /                    → Status server
   GET  /scan                → Full scan semua saham
   GET  /analyze/{ticker}    → Analisis 1 saham
   GET  /ihsg                → Market regime IHSG
   GET  /ranking             → Ranking saham terbaik
   POST /webhook/tradingview → Terima alert dari TradingView
   GET  /health              → Health check
═══════════════════════════════════════════════════════
"""

import os
import logging
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from data_feed import (
    fetch_candles, fetch_ihsg, fetch_batch,
    fetch_multi_timeframe, IDX_UNIVERSE
)
from engines import analyze_stock, TradeResult, Signal
from notifier import TelegramNotifier

# ── Config ────────────────────────────────────────────
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ihsg_trader")

CAPITAL            = float(os.getenv("CAPITAL", 100_000_000))
RISK_PCT           = float(os.getenv("RISK_PER_TRADE", 0.015))
MIN_SCORE_BUY      = int(os.getenv("MIN_SCORE_BUY", 70))
MIN_SCORE_SELL     = int(os.getenv("MIN_SCORE_SELL", 70))
WEBHOOK_SECRET     = os.getenv("WEBHOOK_SECRET", "ihsg_trader_secret_2024")
SCAN_INTERVAL_MIN  = int(os.getenv("SCAN_INTERVAL_MINUTES", 15))
TELEGRAM_TOKEN     = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "")

# Stock name mapping
STOCK_NAMES = {
    "BBCA": "Bank Central Asia",   "BBRI": "Bank Rakyat Indonesia",
    "TLKM": "Telkom Indonesia",    "ASII": "Astra International",
    "BMRI": "Bank Mandiri",        "UNVR": "Unilever Indonesia",
    "GOTO": "GoTo Gojek Tokopedia","BREN": "Barito Renewables",
    "PGAS": "Perusahaan Gas Negara","MDKA": "Merdeka Copper Gold",
    "INDF": "Indofood Sukses Makmur","KLBF": "Kalbe Farma",
    "ICBP": "Indofood CBP",        "MYOR": "Mayora Indah",
    "SMGR": "Semen Indonesia",     "ANTM": "Aneka Tambang",
    "INKP": "Indah Kiat Pulp",     "PTBA": "Bukit Asam",
    "EMTK": "Elang Mahkota Teknologi","SIDO": "Industri Jamu SIDO MUNCUL",
    "ADRO": "Adaro Energy",        "ITMG": "Indo Tambangraya Megah",
    "TOWR": "Sarana Menara Nusantara","EXCL": "XL Axiata",
    "ISAT": "Indosat Ooredoo Hutchison",
}

# ── In-memory cache ────────────────────────────────────
cache: dict = {
    "last_scan": None,
    "results":   [],
    "ihsg":      None,
    "scan_time": None,
}

# ── Notifier ───────────────────────────────────────────
notifier: Optional[TelegramNotifier] = None
if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
    notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)


# ════════════════════════════════════════════
#  CORE SCAN FUNCTION
# ════════════════════════════════════════════

async def run_full_scan(
    symbols: list[str] = None,
    timeframe: str = "1D",
    notify: bool = True,
) -> list[dict]:
    symbols = symbols or IDX_UNIVERSE
    logger.info(f"Starting scan: {len(symbols)} symbols | TF={timeframe}")

    # Ambil data IHSG
    ihsg_raw = fetch_ihsg()
    if not ihsg_raw:
        logger.error("Failed to fetch IHSG data")
        ihsg_raw = []

    # Batch fetch semua saham
    batch_data = fetch_batch(symbols, timeframe=timeframe)

    results = []
    for sym in symbols:
        raw = batch_data.get(sym)
        if not raw or len(raw) < 60:
            logger.warning(f"{sym}: insufficient data, skipping")
            continue

        name = STOCK_NAMES.get(sym, sym)
        result = analyze_stock(
            symbol=sym,
            name=name,
            candles=raw,
            ihsg_candles=ihsg_raw,
            capital=CAPITAL,
            timeframe=timeframe,
        )
        if result:
            results.append(result)

    # Sort by score descending
    results.sort(key=lambda r: r.score.total, reverse=True)

    # Update cache
    cache["results"]   = results
    cache["scan_time"] = datetime.now().isoformat()
    if ihsg_raw:
        from engines import to_candles, market_regime_engine
        cache["ihsg"] = market_regime_engine(to_candles(ihsg_raw))

    logger.info(f"Scan complete: {len(results)} results")

    # Kirim notifikasi Telegram
    if notify and notifier:
        await _notify_signals(results)

    return [result_to_dict(r) for r in results]


async def _notify_signals(results: list[TradeResult]):
    """Kirim sinyal BUY/SELL ke Telegram."""
    try:
        # Summary
        notifier.send_scan_summary(results)

        # Detail setiap sinyal
        for r in results:
            if r.signal in (Signal.BUY, Signal.SELL) and r.score.total >= MIN_SCORE_BUY:
                notifier.send_signal(r)
    except Exception as e:
        logger.error(f"Telegram notify error: {e}")


def result_to_dict(r: TradeResult) -> dict:
    """Serialize TradeResult ke dict (JSON-safe)."""
    return {
        "ticker":        r.ticker,
        "name":          r.name,
        "price":         r.price,
        "change_pct":    r.change_pct,
        "signal":        r.signal.value,
        "score":         r.score.total,
        "score_category": (
            "HIGH_PROB" if r.score.total >= 85 else
            "TRADEABLE" if r.score.total >= 70 else
            "WATCHLIST" if r.score.total >= 60 else "AVOID"
        ),
        "timeframe":     r.timeframe,
        "mtf_alignment": r.mtf_alignment,
        "score_breakdown": {
            "trend":     r.score.trend_score,
            "momentum":  r.score.momentum_score,
            "volume":    r.score.volume_score,
            "structure": r.score.structure_score,
            "liquidity": r.score.liquidity_score,
            "volatility":r.score.volatility_score,
            "candle":    r.score.candle_score,
        },
        "trade": {
            "entry":     r.risk.entry,
            "stop_loss": r.risk.stop_loss,
            "tp1":       r.risk.tp1,
            "tp2":       r.risk.tp2,
            "rr1":       r.risk.rr1,
            "rr2":       r.risk.rr2,
            "lot_size":  r.risk.lot_size,
            "risk_idr":  r.risk.risk_idr,
        },
        "diagnostics": {
            "regime":     r.regime.regime.value,
            "trend":      r.structure.trend.value,
            "momentum":   r.momentum.momentum.value,
            "volume_sig": r.volume.signal.value,
            "rel_vol":    r.volume.rel_vol,
            "volatility": r.volatility.condition.value,
            "atr":        r.volatility.atr14,
            "liquidity":  r.liquidity.condition.value,
            "breakout":   r.breakout.breakout_type.value,
            "pattern":    r.candle.pattern,
            "bos":        r.structure.bos,
            "choch":      r.structure.choch,
            "hh":         r.structure.hh_count,
            "hl":         r.structure.hl_count,
            "ema20":      r.momentum.ema20,
            "ema50":      r.momentum.ema50,
            "dist_ema20": r.momentum.dist_from_ema20,
            "slope20":    r.momentum.slope20,
        },
    }


# ════════════════════════════════════════════
#  FASTAPI APP
# ════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 IHSG AI Trader Backend starting...")

    # Auto-scan scheduler
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        lambda: run_full_scan(notify=True),
        "interval",
        minutes=SCAN_INTERVAL_MIN,
        id="auto_scan",
    )
    scheduler.start()
    logger.info(f"⏰ Auto-scan scheduled every {SCAN_INTERVAL_MIN} minutes")

    yield  # App running

    # Shutdown
    scheduler.shutdown()
    logger.info("Server stopped.")


app = FastAPI(
    title="IHSG AI Trader API",
    description="Professional AI Trading Framework untuk Bursa Efek Indonesia",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ════════════════════════════════════════════
#  ENDPOINTS
# ════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "app":     "IHSG AI Trader",
        "version": "2.1.0",
        "status":  "online",
        "time":    datetime.now().isoformat(),
        "endpoints": ["/scan", "/analyze/{ticker}", "/ihsg", "/ranking", "/webhook/tradingview"],
    }


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


# ─── SCAN ALL ──────────────────────────────────────────
@app.get("/scan")
async def scan_all(
    background_tasks: BackgroundTasks,
    timeframe: str = Query("1D", description="Timeframe: 5m, 15m, 1H, 1D"),
    notify:    bool = Query(True, description="Kirim ke Telegram"),
    signal:    Optional[str] = Query(None, description="Filter: BUY, SELL, WAIT"),
    min_score: int = Query(0, description="Minimum score filter"),
):
    """
    Scan semua saham IDX dan return ranked results.
    Gunakan background=true untuk scan async.
    """
    results = await run_full_scan(timeframe=timeframe, notify=notify)

    # Filter
    if signal:
        results = [r for r in results if r["signal"] == signal.upper()]
    if min_score > 0:
        results = [r for r in results if r["score"] >= min_score]

    return {
        "scan_time":    cache["scan_time"],
        "total":        len(results),
        "timeframe":    timeframe,
        "results":      results,
    }


# ─── ANALYZE SINGLE STOCK ──────────────────────────────
@app.get("/analyze/{ticker}")
async def analyze_single(
    ticker: str,
    timeframe: str = Query("1D"),
    mtf: bool = Query(False, description="Multi-timeframe analysis"),
):
    """
    Analisis mendalam 1 saham.
    """
    sym = ticker.upper()

    ihsg_raw = fetch_ihsg()
    if not ihsg_raw:
        raise HTTPException(503, "Gagal ambil data IHSG")

    if mtf:
        tf_data = fetch_multi_timeframe(sym)
        if "1D" not in tf_data:
            raise HTTPException(404, f"{sym}: data tidak tersedia")
        candles = tf_data["1D"]
        h1  = tf_data.get("1H")
        m15 = tf_data.get("15m")
    else:
        candles = fetch_candles(sym, timeframe)
        h1 = m15 = None

    if not candles:
        raise HTTPException(404, f"{sym}: data tidak ditemukan")

    name = STOCK_NAMES.get(sym, sym)
    result = analyze_stock(
        symbol=sym, name=name,
        candles=candles, ihsg_candles=ihsg_raw,
        capital=CAPITAL, timeframe=timeframe,
        h1_candles=h1, m15_candles=m15,
    )

    if not result:
        raise HTTPException(500, f"{sym}: analisis gagal")

    return result_to_dict(result)


# ─── IHSG REGIME ───────────────────────────────────────
@app.get("/ihsg")
def get_ihsg():
    """Market regime IHSG saat ini."""
    raw = fetch_ihsg()
    if not raw:
        raise HTTPException(503, "Gagal ambil data IHSG")

    from engines import to_candles, market_regime_engine
    regime = market_regime_engine(to_candles(raw))

    return {
        "regime":    regime.regime.value,
        "price":     regime.ihsg_price,
        "ma20":      regime.ma20,
        "ma50":      regime.ma50,
        "dist_ma50": regime.distance_ma50_pct,
        "timestamp": datetime.now().isoformat(),
    }


# ─── RANKING ───────────────────────────────────────────
@app.get("/ranking")
async def get_ranking(
    top_n:     int = Query(10, description="Jumlah saham teratas"),
    signal:    Optional[str] = Query(None),
    min_score: int = Query(70),
):
    """
    Ranking saham terbaik berdasarkan probability score.
    Gunakan cache hasil scan terakhir.
    """
    if not cache["results"]:
        # Trigger scan jika belum ada data
        await run_full_scan(notify=False)

    results = [result_to_dict(r) for r in cache["results"]]

    if signal:
        results = [r for r in results if r["signal"] == signal.upper()]
    results = [r for r in results if r["score"] >= min_score]
    results = results[:top_n]

    return {
        "scan_time": cache["scan_time"],
        "total":     len(results),
        "ranking":   results,
    }


# ════════════════════════════════════════════
#  TRADINGVIEW WEBHOOK ENDPOINT
# ════════════════════════════════════════════

class TVWebhookPayload(BaseModel):
    """Format payload dari TradingView Pine Script alert."""
    secret:    str
    ticker:    str
    signal:    str           # "BUY" | "SELL"
    price:     float
    timeframe: Optional[str] = "1D"
    score:     Optional[int] = None
    # Optional tambahan dari Pine Script
    volume:    Optional[float] = None
    atr:       Optional[float] = None
    ema20:     Optional[float] = None
    comment:   Optional[str]   = None


@app.post("/webhook/tradingview")
async def tradingview_webhook(
    payload: TVWebhookPayload,
    background_tasks: BackgroundTasks,
):
    """
    Endpoint penerima alert dari TradingView.
    
    Setup di TradingView:
    1. Buat Alert dari Pine Script
    2. Webhook URL: https://your-server.com/webhook/tradingview
    3. Message body (JSON):
    {
      "secret": "ihsg_trader_secret_2024",
      "ticker": "{{ticker}}",
      "signal": "BUY",
      "price":  {{close}},
      "timeframe": "{{interval}}",
      "volume": {{volume}}
    }
    """
    # Validasi secret
    if payload.secret != WEBHOOK_SECRET:
        logger.warning(f"Webhook: invalid secret from {payload.ticker}")
        raise HTTPException(403, "Invalid webhook secret")

    sym = payload.ticker.replace(".JK", "").upper()
    logger.info(f"Webhook received: {sym} | {payload.signal} | price={payload.price}")

    # Jalankan analisis penuh di background
    background_tasks.add_task(
        _process_webhook_signal,
        sym, payload.signal, payload.price, payload.timeframe
    )

    return {
        "status":  "accepted",
        "ticker":  sym,
        "signal":  payload.signal,
        "message": "Analysis queued",
    }


async def _process_webhook_signal(
    symbol:    str,
    tv_signal: str,
    tv_price:  float,
    timeframe: str,
):
    """
    Proses sinyal dari TradingView:
    1. Ambil data terbaru
    2. Jalankan semua AI engine
    3. Validasi sinyal dengan probability score
    4. Kirim ke Telegram jika valid
    """
    logger.info(f"Processing webhook: {symbol} | TV signal={tv_signal}")

    ihsg_raw = fetch_ihsg()
    candles  = fetch_candles(symbol, timeframe)

    if not candles or not ihsg_raw:
        logger.error(f"Webhook [{symbol}]: fetch failed")
        return

    name   = STOCK_NAMES.get(symbol, symbol)
    result = analyze_stock(
        symbol=symbol, name=name,
        candles=candles, ihsg_candles=ihsg_raw,
        capital=CAPITAL, timeframe=timeframe,
    )

    if not result:
        logger.error(f"Webhook [{symbol}]: analysis failed")
        return

    logger.info(
        f"Webhook [{symbol}]: AI signal={result.signal.value} | "
        f"TV signal={tv_signal} | score={result.score.total}"
    )

    # Kirim ke Telegram hanya jika:
    # - AI score memadai
    # - Sinyal AI konfirmasi sinyal TradingView
    ai_confirms = (
        (tv_signal == "BUY"  and result.signal == Signal.BUY)  or
        (tv_signal == "SELL" and result.signal == Signal.SELL)
    )

    if notifier and result.score.total >= MIN_SCORE_BUY:
        if ai_confirms:
            # Konfirmasi penuh
            notifier.send(
                f"✅ <b>KONFIRMASI SINYAL</b>\n"
                f"TradingView + AI sama-sama: <b>{tv_signal} {symbol}</b>\n"
                f"Score: <b>{result.score.total}/100</b>\n"
                f"Entry: <code>{int(result.risk.entry):,}</code> | "
                f"SL: <code>{int(result.risk.stop_loss):,}</code>"
            )
            notifier.send_signal(result)
        else:
            # Divergensi sinyal
            notifier.send(
                f"⚠️ <b>DIVERGENSI SINYAL</b>\n"
                f"TradingView: <b>{tv_signal} {symbol}</b>\n"
                f"AI Engine  : <b>{result.signal.value} {symbol}</b>\n"
                f"Score: {result.score.total}/100 — Tunggu konfirmasi lebih lanjut."
            )


# ════════════════════════════════════════════
#  RUN SERVER
# ════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info",
    )

@app.get("/test-telegram")
def test_telegram():
    """Test kirim pesan Telegram langsung."""
    if not notifier:
        return {"status": "error", "message": "Notifier tidak aktif - cek TELEGRAM_TOKEN dan TELEGRAM_CHAT_ID di Variables"}
    
    result = notifier.send("🧪 <b>TEST BERHASIL!</b>\nIHSG AI Trader server aktif dan Telegram terhubung! ✅")
    return {"status": "ok" if result else "error", "sent": result}

@app.get("/debug-env")
def debug_env():
    import os
    token = os.environ.get("TELEGRAM_TOKEN", "NOT_FOUND")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "NOT_FOUND")
    return {
        "token_exists": bool(token and token != "NOT_FOUND" and token != "isi_nanti"),
        "token_preview": token[:15] + "..." if len(token) > 15 else token,
        "chat_id": chat_id,
        "notifier_active": notifier is not None,
    }
