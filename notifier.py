"""
IHSG AI TRADER — TELEGRAM NOTIFIER (Simple Requests Version)
"""
import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"

    def send(self, message: str) -> bool:
        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "HTML",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("Telegram sent OK")
                return True
            else:
                logger.error(f"Telegram error: {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Telegram exception: {e}")
            return False

    def send_signal(self, result) -> bool:
        if result.signal.value == "WAIT":
            return False

        sig = "🟢 BUY" if result.signal.value == "BUY" else "🔴 SELL"
        now = datetime.now().strftime("%d/%m/%Y %H:%M WIB")
        bar = "█" * round(result.score.total/10) + "░" * (10 - round(result.score.total/10))

        msg = f"""
╔══════════════════════════╗
║  <b>IHSG AI TRADER SIGNAL</b>
╚══════════════════════════╝

{sig} — <b>{result.ticker}</b> ({result.name})
🕐 {now}

<b>📊 SCORE: [{bar}] {result.score.total}/100</b>

<b>📋 TRADE</b>
├ Entry   : <code>{int(result.risk.entry):,}</code>
├ SL      : <code>{int(result.risk.stop_loss):,}</code>
├ TP1     : <code>{int(result.risk.tp1):,}</code>
├ TP2     : <code>{int(result.risk.tp2):,}</code>
└ R:R     : <b>{result.risk.rr1}:1</b>

<b>🔍 DIAGNOSTICS</b>
├ Regime  : {result.regime.regime.value}
├ Trend   : {result.structure.trend.value}
├ Volume  : {result.volume.signal.value} ({result.volume.rel_vol:.1f}x)
├ Pattern : {result.candle.pattern}
└ MTF     : {result.mtf_alignment}

<b>💰 Lot: {result.risk.lot_size:,} lembar</b>
<i>⚠️ Gunakan manajemen risiko yang ketat.</i>
""".strip()
        return self.send(msg)

    def send_scan_summary(self, results: list) -> bool:
        from engines import Signal
        buys  = [r for r in results if r.signal == Signal.BUY]
        sells = [r for r in results if r.signal == Signal.SELL]
        high  = [r for r in results if r.score.total >= 85]
        now   = datetime.now().strftime("%d/%m/%Y %H:%M WIB")

        lines = [
            f"📡 <b>IHSG AI SCAN REPORT</b>",
            f"🕐 {now}",
            f"",
            f"Scan <b>{len(results)}</b> saham IDX:",
            f"🟢 BUY    : <b>{len(buys)}</b>",
            f"🔴 SELL   : <b>{len(sells)}</b>",
            f"⭐ HIGH PROB (≥85): <b>{len(high)}</b>",
            f"",
            f"<b>🏆 TOP 5:</b>",
        ]

        top5 = sorted(results, key=lambda r: r.score.total, reverse=True)[:5]
        for i, r in enumerate(top5, 1):
            icon = "🟢" if r.signal.value=="BUY" else "🔴" if r.signal.value=="SELL" else "⏸"
            lines.append(f"{i}. {icon} <b>{r.ticker}</b> Score:<b>{r.score.total}</b> Entry:{int(r.risk.entry):,}")

        return self.send("\n".join(lines))
