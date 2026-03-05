"""
═══════════════════════════════════════════════════════
 IHSG AI TRADER — TELEGRAM NOTIFIER
 Kirim sinyal trading ke Telegram Bot
═══════════════════════════════════════════════════════
"""

import requests
import logging
from engines import TradeResult, Signal
from datetime import datetime

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token   = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"

    def send(self, message: str, parse_mode: str = "HTML") -> bool:
        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id":    self.chat_id,
                    "text":       message,
                    "parse_mode": parse_mode,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                logger.info("Telegram message sent")
                return True
            else:
                logger.error(f"Telegram error: {resp.text}")
                return False
        except Exception as e:
            logger.error(f"Telegram exception: {e}")
            return False

    def send_signal(self, result: TradeResult) -> bool:
        """Format dan kirim sinyal trading."""
        if result.signal == Signal.WAIT:
            return False  # Tidak kirim WAIT signal

        signal_emoji = "🟢 BUY" if result.signal == Signal.BUY else "🔴 SELL"
        score_bar    = self._score_bar(result.score.total)
        regime_emoji = {"BULLISH": "📈", "BEARISH": "📉", "SIDEWAYS": "📊"}.get(
            result.regime.regime.value, "📊"
        )
        trend_emoji  = {"UPTREND": "⬆️", "DOWNTREND": "⬇️", "SIDEWAYS": "↔️"}.get(
            result.structure.trend.value, "↔️"
        )
        now = datetime.now().strftime("%d/%m/%Y %H:%M WIB")

        msg = f"""
╔══════════════════════════╗
║  <b>IHSG AI TRADER SIGNAL</b>
╚══════════════════════════╝

{signal_emoji} — <b>{result.ticker}</b> ({result.name})
🕐 {now}

<b>📊 PROBABILITY SCORE</b>
{score_bar} <b>{result.score.total}/100</b>

<b>📋 TRADE PARAMETERS</b>
├ Entry      : <code>{int(result.risk.entry):,}</code>
├ Stop Loss  : <code>{int(result.risk.stop_loss):,}</code>
├ Take Profit 1 : <code>{int(result.risk.tp1):,}</code>
├ Take Profit 2 : <code>{int(result.risk.tp2):,}</code>
└ Risk:Reward   : <b>{result.risk.rr1}:1</b>

<b>🔍 MARKET DIAGNOSTICS</b>
├ Regime    : {regime_emoji} {result.regime.regime.value}
├ Trend     : {trend_emoji} {result.structure.trend.value}
├ Momentum  : {result.momentum.momentum.value}
├ Volume    : {result.volume.signal.value} ({result.volume.rel_vol:.1f}x avg)
├ Volatility: {result.volatility.condition.value}
├ Liquidity : {result.liquidity.condition.value}
├ Breakout  : {result.breakout.breakout_type.value}
└ Candle    : {result.candle.pattern}

<b>📐 SCORE BREAKDOWN</b>
Trend {result.score.trend_score}/25 | Mom {result.score.momentum_score}/20 | Vol {result.score.volume_score}/15
Struct {result.score.structure_score}/15 | Liq {result.score.liquidity_score}/10 | Volat {result.score.volatility_score}/10 | Candle {result.score.candle_score}/5

<b>💰 POSITION SIZING</b>
└ Lot Size : <code>{result.risk.lot_size:,}</code> lembar
  (risk 1.5% modal = Rp {int(result.risk.risk_idr):,})

<b>📡 MTF Alignment : {result.mtf_alignment}</b>

<i>⚠️ Gunakan manajemen risiko yang ketat.</i>
""".strip()

        return self.send(msg)

    def send_scan_summary(self, results: list[TradeResult]) -> bool:
        """Kirim ringkasan hasil scan semua saham."""
        buys  = [r for r in results if r.signal == Signal.BUY]
        sells = [r for r in results if r.signal == Signal.SELL]
        high_prob = [r for r in results if r.score.total >= 85]

        now = datetime.now().strftime("%d/%m/%Y %H:%M WIB")

        lines = [
            f"📡 <b>IHSG AI SCAN REPORT</b>",
            f"🕐 {now}",
            f"",
            f"📊 Hasil scan <b>{len(results)}</b> saham:",
            f"🟢 BUY    : <b>{len(buys)}</b>",
            f"🔴 SELL   : <b>{len(sells)}</b>",
            f"⭐ HIGH PROB (≥85): <b>{len(high_prob)}</b>",
            f"",
            f"<b>🏆 TOP 5 RANKED:</b>",
        ]

        top5 = sorted(results, key=lambda r: r.score.total, reverse=True)[:5]
        for i, r in enumerate(top5, 1):
            sig_icon = "🟢" if r.signal == Signal.BUY else "🔴" if r.signal == Signal.SELL else "⏸"
            lines.append(
                f"{i}. {sig_icon} <b>{r.ticker}</b> — Score: <b>{r.score.total}</b> | "
                f"Entry: {int(r.risk.entry):,} | SL: {int(r.risk.stop_loss):,}"
            )

        return self.send("\n".join(lines))

    @staticmethod
    def _score_bar(score: int) -> str:
        filled = round(score / 10)
        empty  = 10 - filled
        bar    = "█" * filled + "░" * empty
        return f"[{bar}]"
