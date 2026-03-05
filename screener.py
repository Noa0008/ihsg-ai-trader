"""
IHSG AI TRADER — TRADINGVIEW SCREENER
Pakai TradingView Scanner API untuk filter saham IDX secara real-time.
Parameter: Price > 50, Volume > 1M, Rel Volume > 1, Change% > 0
"""
import logging
import requests
import json

logger = logging.getLogger("screener")

TV_SCREENER_URL = "https://scanner.tradingview.com/indonesia/scan"

TV_SCREENER_PAYLOAD = {
    "filter": [
        {
            "left":     "close",
            "operation": "greater",
            "right":    50
        },
        {
            "left":     "volume",
            "operation": "greater",
            "right":    1000000
        },
        {
            "left":     "relative_volume_10d_calc",
            "operation": "greater",
            "right":    1.0
        },
        {
            "left":     "change",
            "operation": "greater",
            "right":    0
        },
    ],
    "options": {
        "lang": "en"
    },
    "markets": ["indonesia"],
    "symbols": {
        "query": {
            "types": ["stock"]
        },
        "tickers": []
    },
    "columns": [
        "name",
        "close",
        "volume",
        "relative_volume_10d_calc",
        "change",
        "change_abs",
        "market_cap_basic",
        "average_volume_10d_calc",
    ],
    "sort": {
        "sortBy":    "relative_volume_10d_calc",
        "sortOrder": "desc"
    },
    "range": [0, 500]
}

HEADERS = {
    "Content-Type":  "application/json",
    "User-Agent":    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Origin":        "https://www.tradingview.com",
    "Referer":       "https://www.tradingview.com/",
}


def run_screener(
    min_price:    float = 50,
    min_volume:   float = 1_000_000,
    min_rel_vol:  float = 1.0,
    min_change:   float = 0.0,
    max_results:  int   = 300,
) -> list[str]:
    """
    Jalankan TradingView screener untuk saham IDX.
    Return list ticker (tanpa .JK suffix).
    """
    payload = json.loads(json.dumps(TV_SCREENER_PAYLOAD))  # deep copy

    # Update filter params sesuai argumen
    for f in payload["filter"]:
        if f["left"] == "close":
            f["right"] = min_price
        elif f["left"] == "volume":
            f["right"] = min_volume
        elif f["left"] == "relative_volume_10d_calc":
            f["right"] = min_rel_vol
        elif f["left"] == "change":
            f["right"] = min_change

    payload["range"] = [0, max_results]

    try:
        resp = requests.post(
            TV_SCREENER_URL,
            json=payload,
            headers=HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        tickers = []
        for item in data.get("data", []):
            # Format: "IDX:BBCA" → "BBCA"
            raw = item.get("s", "")
            ticker = raw.split(":")[-1]
            if ticker:
                tickers.append(ticker)

        logger.info(
            f"TV Screener: {len(tickers)} saham lolos "
            f"(price>{min_price}, vol>{min_volume/1e6:.0f}M, "
            f"relvol>{min_rel_vol}, change>{min_change}%)"
        )

        # Log top 10 untuk debug
        if tickers:
            d = data.get("data", [])
            logger.info("Top 10 by RelVol:")
            for item in d[:10]:
                s = item.get("s","").split(":")[-1]
                v = item.get("d", [])
                if len(v) >= 4:
                    logger.info(f"  {s}: price={v[1]:.0f} vol={v[2]/1e6:.1f}M relvol={v[3]:.2f}x chg={v[4]:.2f}%")

        return tickers

    except requests.exceptions.ConnectionError:
        logger.error("TV Screener: Connection error (network tidak tersedia)")
        return []
    except requests.exceptions.Timeout:
        logger.error("TV Screener: Timeout")
        return []
    except Exception as e:
        logger.error(f"TV Screener error: {e}")
        return []


def run_screener_with_fallback(fallback_universe: list[str] = None) -> list[str]:
    """
    Coba TV screener, fallback ke universe manual jika gagal.
    """
    result = run_screener()
    if result:
        return result

    logger.warning("TV Screener gagal, pakai fallback universe manual")
    if fallback_universe:
        return fallback_universe

    # Fallback ke IDX_UNIVERSE dari data_feed_tv
    from data_feed_tv import IDX_UNIVERSE
    return IDX_UNIVERSE


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tickers = run_screener()
    print(f"\nHasil screener: {len(tickers)} saham")
    print(tickers[:30])
