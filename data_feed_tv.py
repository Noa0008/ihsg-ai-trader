"""
IHSG AI TRADER — DATA FEED (TradingView via tvDatafeed)
Replaces Yahoo Finance with TradingView real-time data.
"""
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger("data_feed_tv")

# ── TvDatafeed setup ──────────────────────────────────────────────────────────
try:
    from tvDatafeed import TvDatafeed, Interval
    _tv = TvDatafeed()  # nologin - cukup untuk IDX real-time
    TV_AVAILABLE = True
    logger.info("TvDatafeed initialized OK")
except Exception as e:
    TV_AVAILABLE = False
    logger.warning(f"TvDatafeed not available: {e}")

# ── Interval mapping ──────────────────────────────────────────────────────────
INTERVAL_MAP = {
    "1m":  Interval.in_1_minute if TV_AVAILABLE else None,
    "5m":  Interval.in_5_minute if TV_AVAILABLE else None,
    "15m": Interval.in_15_minute if TV_AVAILABLE else None,
    "30m": Interval.in_30_minute if TV_AVAILABLE else None,
    "1h":  Interval.in_1_hour if TV_AVAILABLE else None,
    "4h":  Interval.in_4_hour if TV_AVAILABLE else None,
    "1D":  Interval.in_daily if TV_AVAILABLE else None,
    "1W":  Interval.in_weekly if TV_AVAILABLE else None,
}

# ── Daftar saham IDX ──────────────────────────────────────────────────────────
IDX_UNIVERSE = [
    # LQ45 CORE
    "BBCA","BBRI","BMRI","BBNI","TLKM","ASII","UNVR","ICBP","INDF","GGRM",
    "HMSP","KLBF","UNTR","PGAS","PTBA","ADRO","INCO","ANTM","SMGR","INTP",
    "CPIN","JPFA","EXCL","ISAT","TBIG","MNCN","SCMA","EMTK","MTEL","TOWR",
    "INKP","TKIM","ITMG","HRUM","PTRO","BYAN","ELSA","AKRA","ULTJ","MYOR",
    "SIDO","WTON","WSKT","WIKA","PTPP","ADHI","JSMR","BSDE","CTRA","MDKA",
    # BANKING
    "BBTN","BJBR","BJTM","BNGA","BDMN","MEGA","PNBN","BTPS","BRIS","NISP",
    # CONSUMER
    "ACES","MAPI","RALS","LPPF","AMRT","KINO","CLEO","FOOD","SIDO","HEAL",
    # TAMBANG & ENERGI
    "NCKL","ADMR","CUAN","MBMA","TINS","INDY","DOID","GEMS","MBAP","TOBA",
    # PROPERTI
    "PWON","SMRA","LPKR","DMAS","BEST","KIJA","DUTI","MDLN",
    # TEKNOLOGI
    "GOTO","BUKA","DMMX","MTDL","MLPT",
]
IDX_UNIVERSE = list(dict.fromkeys(IDX_UNIVERSE))


def _df_to_candles(df) -> list[dict]:
    """Convert tvDatafeed DataFrame ke format candles list[dict]."""
    if df is None or len(df) == 0:
        return []
    candles = []
    for idx, row in df.iterrows():
        try:
            candles.append({
                "timestamp": str(idx),
                "open":   float(row["open"]),
                "high":   float(row["high"]),
                "low":    float(row["low"]),
                "close":  float(row["close"]),
                "volume": int(row["volume"]),
            })
        except Exception:
            continue
    return candles


def fetch_candles(ticker: str, timeframe: str = "1D", n_bars: int = 200) -> list[dict]:
    """Fetch candles untuk satu ticker dari TradingView."""
    if not TV_AVAILABLE:
        logger.warning("TvDatafeed not available, falling back to yfinance")
        return _fetch_yf_fallback(ticker, timeframe)

    interval = INTERVAL_MAP.get(timeframe, Interval.in_daily)
    try:
        df = _tv.get_hist(ticker, "IDX", interval=interval, n_bars=n_bars)
        candles = _df_to_candles(df)
        if candles:
            logger.debug(f"TV fetch {ticker}: {len(candles)} candles")
            return candles
    except Exception as e:
        logger.warning(f"TV fetch error {ticker}: {e}")

    return _fetch_yf_fallback(ticker, timeframe)


def fetch_batch(tickers: list[str], timeframe: str = "1D", n_bars: int = 200) -> dict[str, list[dict]]:
    """Fetch candles untuk banyak ticker sekaligus."""
    result = {}
    interval = INTERVAL_MAP.get(timeframe, Interval.in_daily)

    for ticker in tickers:
        try:
            if TV_AVAILABLE:
                df = _tv.get_hist(ticker, "IDX", interval=interval, n_bars=n_bars)
                candles = _df_to_candles(df)
                if len(candles) >= 30:
                    result[ticker] = candles
                    continue
        except Exception as e:
            logger.debug(f"TV batch error {ticker}: {e}")

        # Fallback ke yfinance
        candles = _fetch_yf_fallback(ticker, timeframe)
        if len(candles) >= 30:
            result[ticker] = candles

    logger.info(f"Batch fetch: {len(result)}/{len(tickers)} symbols OK")
    return result


def fetch_ihsg(n_bars: int = 200) -> list[dict]:
    """Fetch data IHSG composite index."""
    if TV_AVAILABLE:
        try:
            df = _tv.get_hist("COMPOSITE", "IDX", interval=Interval.in_daily, n_bars=n_bars)
            candles = _df_to_candles(df)
            if candles:
                return candles
        except Exception as e:
            logger.warning(f"TV IHSG error: {e}")

    # Fallback
    return _fetch_yf_fallback("^JKSE", "1D", n_bars)


def _fetch_yf_fallback(ticker: str, timeframe: str, n_bars: int = 200) -> list[dict]:
    """Fallback ke Yahoo Finance jika TvDatafeed gagal."""
    try:
        import yfinance as yf
        tf_map = {"1D": "1d", "1W": "1wk", "1h": "1h", "4h": "1h"}
        interval = tf_map.get(timeframe, "1d")
        ticker_yf = f"{ticker}.JK" if not ticker.startswith("^") else ticker
        df = yf.download(ticker_yf, period="1y", interval=interval,
                        auto_adjust=True, progress=False)
        if df.empty:
            return []
        candles = []
        for idx, row in df.iterrows():
            try:
                candles.append({
                    "timestamp": str(idx),
                    "open":   float(row["Open"]),
                    "high":   float(row["High"]),
                    "low":    float(row["Low"]),
                    "close":  float(row["Close"]),
                    "volume": int(row["Volume"]),
                })
            except Exception:
                continue
        return candles[-n_bars:]
    except Exception as e:
        logger.error(f"YF fallback error {ticker}: {e}")
        return []
