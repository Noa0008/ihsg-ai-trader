import os, json, logging
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

from engines import analyze_stock, Signal, market_regime_engine, TradeResult, to_candles
from data_feed_tv import IDX_UNIVERSE, fetch_batch, fetch_ihsg, fetch_candles
from notifier import TelegramNotifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ihsg_trader")

# ── Config dari env ────────────────────────────────────────────────────────────
TELEGRAM_TOKEN  = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID= os.environ.get("TELEGRAM_CHAT_ID", "")
CAPITAL         = float(os.environ.get("CAPITAL", "100000000"))
RISK_PER_TRADE  = float(os.environ.get("RISK_PER_TRADE", "0.015"))
WEBHOOK_SECRET  = os.environ.get("WEBHOOK_SECRET", "rahasiatrader123")
SCAN_INTERVAL   = int(os.environ.get("SCAN_INTERVAL_MINUTES", "15"))

# ── FIX #3: Path file persistensi ─────────────────────────────────────────────
RESULTS_FILE = Path("scan_results.json")

# ── Notifier ──────────────────────────────────────────────────────────────────
notifier = None
if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
    notifier = TelegramNotifier(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID)
    logger.info(f"Notifier active: {TELEGRAM_CHAT_ID}")
else:
    logger.warning("Notifier inactive — set TELEGRAM_TOKEN & TELEGRAM_CHAT_ID")

scheduler = BackgroundScheduler()

# ── State ──────────────────────────────────────────────────────────────────────
last_scan_results: list[TradeResult] = []
last_scan_time: Optional[datetime]   = None
scan_running: bool                   = False

# FIX #1: State untuk /scan/status
scan_status_state = {
    "status":        "idle",   # idle | running | done | error
    "started_at":    None,
    "finished_at":   None,
    "progress":      0,
    "total":         len(IDX_UNIVERSE),
    "current_stock": None,
    "error":         None,
}


# ── FIX #3: Helpers persist ────────────────────────────────────────────────────

def _serialize_result(r: TradeResult) -> dict:
    """Konversi TradeResult dataclass → dict JSON-serializable."""
    return {
        "ticker":       r.ticker,
        "name":         r.name,
        "signal":       r.signal.value,
        "score":        r.score.total,
        "score_detail": {
            "trend":      r.score.trend_score,
            "momentum":   r.score.momentum_score,
            "volume":     r.score.volume_score,
            "structure":  r.score.structure_score,
            "liquidity":  r.score.liquidity_score,
            "volatility": r.score.volatility_score,
            "candle":     r.score.candle_score,
        },
        "price":        r.price,
        "change_pct":   r.change_pct,
        "entry":        r.risk.entry,
        "stop_loss":    r.risk.stop_loss,
        "tp1":          r.risk.tp1,
        "tp2":          r.risk.tp2,
        "rr1":          r.risk.rr1,
        "rr2":          r.risk.rr2,
        "lot_size":     r.risk.lot_size,
        "risk_idr":     r.risk.risk_idr,
        "regime":       r.regime.regime.value,
        "trend":        r.structure.trend.value,
        "volume_signal":r.volume.signal.value,
        "rel_vol":      r.volume.rel_vol,
        "pattern":      r.candle.pattern,
        "mtf":          r.mtf_alignment,
        "scanned_at":   datetime.now().isoformat(),
    }


def save_results_to_file(results: list[TradeResult]):
    """FIX #3: Simpan hasil scan ke JSON file agar persist across redeploy."""
    try:
        data = {
            "scan_time": datetime.now().isoformat(),
            "total":     len(results),
            "results":   [_serialize_result(r) for r in results],
        }
        RESULTS_FILE.write_text(json.dumps(data, indent=2))
        logger.info(f"✅ Results saved to {RESULTS_FILE} ({len(results)} saham)")
    except Exception as e:
        logger.error(f"save_results_to_file error: {e}")


def load_results_from_file() -> tuple[list[dict], Optional[datetime]]:
    """FIX #3: Load hasil scan dari file saat startup."""
    if not RESULTS_FILE.exists():
        return [], None
    try:
        data = json.loads(RESULTS_FILE.read_text())
        results  = data.get("results", [])
        scan_time = datetime.fromisoformat(data["scan_time"]) if data.get("scan_time") else None
        logger.info(f"📂 Loaded {len(results)} results from {RESULTS_FILE} (scan: {scan_time})")
        return results, scan_time
    except Exception as e:
        logger.error(f"load_results_from_file error: {e}")
        return [], None


# ── Cached file results (untuk fallback saat last_scan_results kosong) ─────────
_file_results: list[dict] = []
_file_scan_time: Optional[datetime] = None


# ── Core scan job ──────────────────────────────────────────────────────────────

def run_scan_job(min_score: int = 70, notify: bool = True):
    global last_scan_results, last_scan_time, scan_running, scan_status_state

    if scan_running:
        logger.info("Scan already running, skip")
        return

    scan_running = True
    scan_status_state.update({
        "status":        "running",
        "started_at":    datetime.now().isoformat(),
        "finished_at":   None,
        "progress":      0,
        "total":         len(IDX_UNIVERSE),
        "current_stock": None,
        "error":         None,
    })

    try:
        logger.info(f"🔍 Scan start: {len(IDX_UNIVERSE)} symbols")
        ihsg_candles  = fetch_ihsg()
        if not ihsg_candles:
            logger.warning("IHSG fetch gagal, pakai fallback dummy regime")
            from data_feed import fetch_ihsg as fetch_ihsg_yf
            ihsg_candles = fetch_ihsg_yf()
        if not ihsg_candles:
            logger.error("IHSG data kosong total, skip scan")
            scan_status_state["status"] = "error"
            scan_status_state["error"] = "IHSG data empty"
            scan_running = False
            return
        candles_map   = fetch_batch(IDX_UNIVERSE, "1D")
        results       = []

        for i, ticker in enumerate(IDX_UNIVERSE):
            # FIX #1: Update progress realtime
            scan_status_state["progress"]      = i + 1
            scan_status_state["current_stock"] = ticker

            if ticker not in candles_map:
                continue
            result = analyze_stock(
                ticker, ticker, candles_map[ticker], ihsg_candles, CAPITAL
            )
            if result:
                results.append(result)

        logger.info(f"✅ Scan complete: {len(results)} results")

        last_scan_results = results
        last_scan_time    = datetime.now()

        # FIX #3: Simpan ke file
        save_results_to_file(results)

        # Update cached file results
        global _file_results, _file_scan_time
        _file_results   = [_serialize_result(r) for r in results]
        _file_scan_time = last_scan_time

        scan_status_state.update({
            "status":      "done",
            "finished_at": last_scan_time.isoformat(),
            "current_stock": None,
        })

        # Kirim notif Telegram
        if notify and notifier:
            filtered = [
                r for r in results
                if r.score.total >= min_score and r.signal != Signal.WAIT
            ]
            for r in filtered:
                notifier.send_signal(r)
            if results:
                notifier.send_scan_summary(results)

    except Exception as e:
        logger.error(f"Scan error: {e}")
        scan_status_state.update({
            "status": "error",
            "error":  str(e),
        })
    finally:
        scan_running = False


# ── Lifespan: startup + shutdown ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _file_results, _file_scan_time

    # FIX #3: Load data dari file saat startup
    _file_results, _file_scan_time = load_results_from_file()

    if _file_results:
        logger.info(f"📂 {len(_file_results)} hasil scan dimuat dari file")
        scan_status_state.update({
            "status":      "done",
            "finished_at": _file_scan_time.isoformat() if _file_scan_time else None,
            "progress":    len(IDX_UNIVERSE),
        })
    else:
        # FIX #2: Auto-run scan saat startup jika belum ada data
        logger.info("🚀 Tidak ada data tersimpan — memulai scan otomatis...")
        import threading
        t = threading.Thread(target=run_scan_job, kwargs={"notify": True}, daemon=True)
        t.start()

    # Scheduler untuk scan berkala
    scheduler.add_job(
        run_scan_job, "interval",
        minutes=SCAN_INTERVAL,
        kwargs={"notify": True},
    )
    scheduler.start()
    logger.info(f"⏰ Scheduler aktif: scan setiap {SCAN_INTERVAL} menit")

    yield

    scheduler.shutdown()
    logger.info("Scheduler stopped")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="IHSG AI Trader", version="6.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "app":         "IHSG AI Trader",
        "version":     "6.1.0",
        "status":      "online",
        "notifier":    notifier is not None,
        "scan_status": scan_status_state["status"],
        "results_cached": len(_file_results),
    }


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now().isoformat()}


@app.get("/debug-env")
def debug_env():
    token   = os.environ.get("TELEGRAM_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
    return {
        "token_exists":    bool(token),
        "token_preview":   token[:15] + "..." if len(token) > 15 else token,
        "chat_id":         chat_id,
        "notifier_active": notifier is not None,
        "results_file":    str(RESULTS_FILE),
        "results_file_exists": RESULTS_FILE.exists(),
        "results_cached":  len(_file_results),
    }


@app.get("/test-telegram")
def test_telegram():
    if not notifier:
        return {"status": "error", "message": "Notifier tidak aktif"}
    result = notifier.send("✅ TEST BERHASIL! IHSG AI Trader v6.1 aktif!")
    return {"status": "ok" if result else "error"}


@app.get("/ihsg")
def ihsg():
    candles = fetch_ihsg()
    if not candles:
        return {"status": "error", "message": "Gagal fetch IHSG"}
    ihsg_list = to_candles(candles)
    regime    = market_regime_engine(ihsg_list)
    return {
        "regime":   regime.regime.value,
        "price":    regime.ihsg_price,   # ← fix typo (was regime.price)
        "ma20":     regime.ma20,
        "ma50":     regime.ma50,
        "dist_ma50_pct": regime.distance_ma50_pct,
    }


# FIX #1: Endpoint /scan/status ────────────────────────────────────────────────

@app.get("/scan/status")
def get_scan_status():
    """
    Realtime status scan.
    Kembalikan progress, saham yang sedang discan, jumlah results, dll.
    """
    in_memory  = len(last_scan_results)
    from_file  = len(_file_results)

    return {
        **scan_status_state,
        "universe_total":  len(IDX_UNIVERSE),
        "results_memory":  in_memory,
        "results_file":    from_file,
        "results_ready":   max(in_memory, from_file),
        "last_scan_time":  last_scan_time.isoformat() if last_scan_time else (
                           _file_scan_time.isoformat() if _file_scan_time else None
                           ),
        "scan_interval_min": SCAN_INTERVAL,
    }


@app.get("/scan")
def scan(background_tasks: BackgroundTasks, min_score: int = 70, notify: bool = False, force: bool = False):
    global scan_running

    if scan_running:
        return {
            "status":   "running",
            "progress": scan_status_state["progress"],
            "total":    scan_status_state["total"],
            "message":  f"Scan sedang berjalan ({scan_status_state['progress']}/{scan_status_state['total']}). Cek /scan/status",
        }

    existing = last_scan_results or _file_results
    if existing and not force:
        return {
            "status":  "has_data",
            "count":   len(existing),
            "message": "Data sudah ada. Gunakan /scan?force=true untuk scan ulang, atau /results untuk lihat hasil.",
        }

    background_tasks.add_task(run_scan_job, min_score, notify)
    return {
        "status":  "started",
        "total":   len(IDX_UNIVERSE),
        "message": f"Scan {len(IDX_UNIVERSE)} saham dimulai. Cek /scan/status untuk progress.",
    }


@app.get("/results")
def results(min_score: int = 70, signal: str = None):
    # Prioritaskan hasil scan segar dari memory
    if last_scan_results:
        source  = "memory"
        raw     = last_scan_results
        scan_t  = last_scan_time

        filtered = [r for r in raw if r.score.total >= min_score]
        if signal:
            filtered = [r for r in filtered if r.signal.value == signal.upper()]
        ranked = sorted(filtered, key=lambda r: r.score.total, reverse=True)

        return {
            "source":       source,
            "scan_time":    scan_t.isoformat() if scan_t else None,
            "total_scanned":len(raw),
            "filtered":     len(filtered),
            "scan_running": scan_running,
            "results": [
                {
                    "ticker":    r.ticker,
                    "score":     r.score.total,
                    "signal":    r.signal.value,
                    "entry":     r.risk.entry,
                    "stop_loss": r.risk.stop_loss,
                    "tp1":       r.risk.tp1,
                    "tp2":       r.risk.tp2,
                    "rr1":       r.risk.rr1,
                    "regime":    r.regime.regime.value,
                    "trend":     r.structure.trend.value,
                    "rel_vol":   r.volume.rel_vol,
                    "pattern":   r.candle.pattern,
                    "mtf":       r.mtf_alignment,
                }
                for r in ranked
            ],
        }

    # Fallback ke hasil dari file
    if _file_results:
        filtered = [r for r in _file_results if r.get("score", 0) >= min_score]
        if signal:
            filtered = [r for r in filtered if r.get("signal", "") == signal.upper()]
        ranked = sorted(filtered, key=lambda r: r.get("score", 0), reverse=True)

        return {
            "source":       "file",
            "scan_time":    _file_scan_time.isoformat() if _file_scan_time else None,
            "total_scanned":len(_file_results),
            "filtered":     len(filtered),
            "scan_running": scan_running,
            "results":      ranked,
        }

    # Tidak ada data sama sekali
    return {
        "status":       "no_data",
        "scan_running": scan_running,
        "message":      (
            "Scan sedang berjalan, cek /scan/status untuk progress."
            if scan_running else
            "Belum ada hasil scan. Akses /scan dulu."
        ),
    }


@app.get("/ranking")
def ranking(top_n: int = 10):
    source = last_scan_results or _file_results
    if not source:
        return {"status": "no_data", "message": "Belum ada hasil scan. Akses /scan dulu."}

    if last_scan_results:
        ranked = sorted(last_scan_results, key=lambda r: r.score.total, reverse=True)[:top_n]
        return {
            "timestamp": last_scan_time.isoformat() if last_scan_time else None,
            "results": [
                {"ticker": r.ticker, "score": r.score.total, "signal": r.signal.value}
                for r in ranked
            ],
        }
    else:
        ranked = sorted(_file_results, key=lambda r: r.get("score", 0), reverse=True)[:top_n]
        return {
            "timestamp": _file_scan_time.isoformat() if _file_scan_time else None,
            "source":    "file",
            "results":   ranked,
        }


@app.post("/webhook/tradingview")
async def webhook(payload: dict):
    if payload.get("secret") != WEBHOOK_SECRET:
        return {"status": "error", "message": "Invalid secret"}

    ticker      = payload.get("ticker", "").replace(".JK", "")
    candles     = fetch_candles(ticker, "1D")
    ihsg_candles= fetch_ihsg()

    if not candles:
        return {"status": "error", "message": "No data"}

    result = analyze_stock(ticker, ticker, candles, ihsg_candles, CAPITAL)
    if result and notifier and result.signal != Signal.WAIT and result.score.total >= 70:
        notifier.send_signal(result)

    return {
        "status": "ok",
        "signal": result.signal.value if result else "WAIT",
        "score":  result.score.total  if result else 0,
    }
