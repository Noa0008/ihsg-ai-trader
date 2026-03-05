"""
═══════════════════════════════════════════════════════
 IHSG AI TRADER — DATA FEED MODULE
 Mengambil data OHLCV dari Yahoo Finance untuk saham IDX
═══════════════════════════════════════════════════════
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# ── Daftar saham IDX liquid ──────────────────────────
IDX_UNIVERSE = [
    "BBCA", "BBRI", "TLKM", "ASII", "BMRI",
    "UNVR", "GOTO", "BREN", "PGAS", "MDKA",
    "INDF", "KLBF", "ICBP", "MYOR", "SMGR",
    "ANTM", "INKP", "PTBA", "EMTK", "SIDO",
    "ADRO", "ITMG", "TOWR", "EXCL", "ISAT",
    "BSDE", "CPIN", "JPFA", "MAPI", "ACES",
]

# Yahoo Finance timeframe mapping
TF_MAP = {
    "5m":  "5m",
    "15m": "15m",
    "1H":  "1h",
    "1D":  "1d",
}

# Yahoo Finance period mapping (max sesuai timeframe)
PERIOD_MAP = {
    "5m":  "5d",
    "15m": "60d",
    "1h":  "6mo",
    "1d":  "1y",
}


def get_yahoo_ticker(symbol: str) -> str:
    """Konversi ticker IDX ke format Yahoo Finance (tambah .JK)"""
    if not symbol.endswith(".JK"):
        return f"{symbol}.JK"
    return symbol


def fetch_candles(
    symbol: str,
    timeframe: str = "1D",
    period: Optional[str] = None,
    min_candles: int = 60,
) -> Optional[list[dict]]:
    """
    Ambil data OHLCV dari Yahoo Finance.
    
    Args:
        symbol:    Ticker IDX (e.g. "BBCA")
        timeframe: "5m", "15m", "1H", "1D"
        period:    Override period (e.g. "3mo", "1y")
        min_candles: Minimum candle yang dibutuhkan
    
    Returns:
        List of candle dict: [{open, high, low, close, volume}, ...]
        None jika gagal
    """
    try:
        ticker_yf = get_yahoo_ticker(symbol)
        tf = TF_MAP.get(timeframe, "1d")
        per = period or PERIOD_MAP.get(tf, "3mo")

        logger.info(f"Fetching {ticker_yf} | TF={tf} | Period={per}")

        ticker_obj = yf.Ticker(ticker_yf)
        df = ticker_obj.history(period=per, interval=tf, auto_adjust=True)

        if df.empty or len(df) < min_candles:
            logger.warning(f"{symbol}: insufficient data ({len(df)} candles)")
            return None

        # Bersihkan data
        df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        df = df[df["Volume"] > 0]

        candles = [
            {
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": int(row["Volume"]),
                "timestamp": str(idx),
            }
            for idx, row in df.iterrows()
        ]

        logger.info(f"{symbol}: {len(candles)} candles loaded")
        return candles

    except Exception as e:
        logger.error(f"fetch_candles error [{symbol}]: {e}")
        return None


def fetch_multi_timeframe(symbol: str) -> dict:
    """
    Ambil data untuk semua timeframe sekaligus.
    Digunakan untuk multi-timeframe confirmation.
    
    Returns:
        {
          "1D":  [...candles],
          "1H":  [...candles],
          "15m": [...candles],
          "5m":  [...candles],
        }
    """
    result = {}
    for tf in ["1D", "1H", "15m", "5m"]:
        candles = fetch_candles(symbol, timeframe=tf, min_candles=30)
        if candles:
            result[tf] = candles
        else:
            logger.warning(f"{symbol} {tf}: no data, skipping")
    return result


def fetch_ihsg(period: str = "1y") -> Optional[list[dict]]:
    """
    Ambil data IHSG (^JKSE) untuk Market Regime detection.
    """
    try:
        df = yf.Ticker("^JKSE").history(period=period, interval="1d", auto_adjust=True)
        if df.empty:
            return None
        df = df.dropna()
        return [
            {
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": int(row["Volume"]),
                "timestamp": str(idx),
            }
            for idx, row in df.iterrows()
        ]
    except Exception as e:
        logger.error(f"fetch_ihsg error: {e}")
        return None


def fetch_batch(
    symbols: list[str],
    timeframe: str = "1D",
    period: Optional[str] = None,
) -> dict[str, list[dict]]:
    """
    Batch fetch untuk multiple symbols sekaligus.
    Lebih efisien menggunakan yf.download().
    """
    try:
        tickers_yf = [get_yahoo_ticker(s) for s in symbols]
        tf = TF_MAP.get(timeframe, "1d")
        per = period or PERIOD_MAP.get(tf, "3mo")

        logger.info(f"Batch fetching {len(symbols)} symbols | TF={tf}")

        df = yf.download(
            tickers=tickers_yf,
            period=per,
            interval=tf,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )

        result = {}
        for sym, ticker_yf in zip(symbols, tickers_yf):
            try:
                if ticker_yf in df.columns.get_level_values(0):
                    sub = df[ticker_yf].dropna()
                else:
                    sub = df.dropna()

                if len(sub) < 30:
                    continue

                result[sym] = [
                    {
                        "open":   float(row["Open"]),
                        "high":   float(row["High"]),
                        "low":    float(row["Low"]),
                        "close":  float(row["Close"]),
                        "volume": int(row["Volume"]),
                        "timestamp": str(idx),
                    }
                    for idx, row in sub.iterrows()
                    if row["Volume"] > 0
                ]
            except Exception as e:
                logger.warning(f"batch parse error [{sym}]: {e}")
                continue

        logger.info(f"Batch complete: {len(result)}/{len(symbols)} symbols")
        return result

    except Exception as e:
        logger.error(f"fetch_batch error: {e}")
        # Fallback ke single fetch
        result = {}
        for sym in symbols:
            candles = fetch_candles(sym, timeframe, period)
            if candles:
                result[sym] = candles
        return result


def get_current_price(symbol: str) -> Optional[float]:
    """Ambil harga terakhir (realtime/delayed)."""
    try:
        ticker_yf = get_yahoo_ticker(symbol)
        info = yf.Ticker(ticker_yf).fast_info
        return float(info.last_price)
    except Exception as e:
        logger.error(f"get_current_price error [{symbol}]: {e}")
        return None
