"""
═══════════════════════════════════════════════════════
 IHSG AI TRADER — ANALYSIS ENGINES
 Semua engine analisis teknikal + probability scoring
═══════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


# ════════════════════════════════════════════
#  DATA STRUCTURES
# ════════════════════════════════════════════

class Signal(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"

class Regime(str, Enum):
    BULLISH  = "BULLISH"
    SIDEWAYS = "SIDEWAYS"
    BEARISH  = "BEARISH"

class Trend(str, Enum):
    UPTREND   = "UPTREND"
    DOWNTREND = "DOWNTREND"
    SIDEWAYS  = "SIDEWAYS"

class Momentum(str, Enum):
    BULLISH  = "BULLISH"
    BEARISH  = "BEARISH"
    NEUTRAL  = "NEUTRAL"

class VolCondition(str, Enum):
    EXPANSION   = "EXPANSION"
    COMPRESSION = "COMPRESSION"

class VolSignal(str, Enum):
    SPIKE      = "SPIKE"
    ELEVATED   = "ELEVATED"
    NORMAL     = "NORMAL"
    ABSORPTION = "ABSORPTION"

class LiqCondition(str, Enum):
    CLEAR         = "CLEAR"
    POOL_DETECTED = "POOL_DETECTED"
    STOP_HUNT     = "STOP_HUNT"

class BreakoutType(str, Enum):
    VALID_BULL  = "VALID_BULL"
    VALID_BEAR  = "VALID_BEAR"
    FALSE       = "FALSE"
    WEAK        = "WEAK"
    COMPRESSION = "COMPRESSION"
    NONE        = "NONE"


@dataclass
class Candle:
    open:   float
    high:   float
    low:    float
    close:  float
    volume: int
    timestamp: str = ""

    @property
    def body(self):      return abs(self.close - self.open)
    @property
    def range(self):     return self.high - self.low or 0.0001
    @property
    def upper_wick(self):return self.high - max(self.close, self.open)
    @property
    def lower_wick(self):return min(self.close, self.open) - self.low
    @property
    def body_ratio(self):return self.body / self.range
    @property
    def upper_ratio(self):return self.upper_wick / self.range
    @property
    def lower_ratio(self):return self.lower_wick / self.range
    @property
    def bullish(self):   return self.close >= self.open


@dataclass
class RegimeResult:
    regime: Regime
    ihsg_price: float
    ma20: float
    ma50: float
    distance_ma50_pct: float


@dataclass
class StructureResult:
    trend:      Trend
    hh_count:   int
    hl_count:   int
    lh_count:   int
    ll_count:   int
    bos:        bool
    choch:      bool
    swing_high: float
    swing_low:  float


@dataclass
class MomentumResult:
    momentum:        Momentum
    ema20:           float
    ema50:           float
    ema100:          float
    slope20:         float
    dist_from_ema20: float
    price:           float


@dataclass
class VolatilityResult:
    condition:   VolCondition
    atr14:       float
    atr_sma20:   float
    ratio:       float


@dataclass
class VolumeResult:
    signal:     VolSignal
    rel_vol:    float
    avg_vol:    float
    cur_vol:    int
    divergence: bool


@dataclass
class LiquidityResult:
    condition:        LiqCondition
    stop_hunt:        bool
    sweep_high:       bool
    sweep_low:        bool
    equal_highs:      bool
    equal_lows:       bool


@dataclass
class BreakoutResult:
    breakout_type: BreakoutType
    broke_high:    bool
    broke_low:     bool
    compression:   bool


@dataclass
class CandleResult:
    pattern:        str
    is_bullish:     bool
    pattern_bull:   bool
    pattern_bear:   bool
    body_ratio:     float
    upper_ratio:    float
    lower_ratio:    float


@dataclass
class ScoreBreakdown:
    trend_score:    int
    momentum_score: int
    volume_score:   int
    structure_score:int
    liquidity_score:int
    volatility_score:int
    candle_score:   int
    total:          int


@dataclass
class RiskParams:
    entry:    float
    stop_loss:float
    tp1:      float
    tp2:      float
    rr1:      float
    rr2:      float
    lot_size: int
    risk_idr: float


@dataclass
class TradeResult:
    ticker:    str
    name:      str
    signal:    Signal
    score:     ScoreBreakdown
    risk:      RiskParams
    regime:    RegimeResult
    structure: StructureResult
    momentum:  MomentumResult
    volatility:VolatilityResult
    volume:    VolumeResult
    liquidity: LiquidityResult
    breakout:  BreakoutResult
    candle:    CandleResult
    price:     float
    change_pct:float
    timeframe: str = "1D"
    mtf_alignment: str = "N/A"


# ════════════════════════════════════════════
#  MATH UTILITIES
# ════════════════════════════════════════════

def to_candles(raw: list[dict]) -> list[Candle]:
    return [Candle(**{k: v for k, v in c.items() if k in Candle.__dataclass_fields__}) for c in raw]

def ema(data: np.ndarray, period: int) -> np.ndarray:
    k = 2.0 / (period + 1)
    result = np.zeros_like(data, dtype=float)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = data[i] * k + result[i-1] * (1-k)
    return result

def sma(data: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(data).rolling(period, min_periods=1).mean().values

def true_range(candles: list[Candle]) -> np.ndarray:
    tr = []
    for i, c in enumerate(candles):
        if i == 0:
            tr.append(c.high - c.low)
        else:
            p = candles[i-1]
            tr.append(max(c.high - c.low,
                          abs(c.high - p.close),
                          abs(c.low  - p.close)))
    return np.array(tr)

def calc_atr(candles: list[Candle], period: int = 14) -> np.ndarray:
    return sma(true_range(candles), period)

def find_pivot_highs(highs: np.ndarray, window: int = 5) -> list[tuple[int, float]]:
    pivots = []
    for i in range(window, len(highs) - window):
        if highs[i] == max(highs[i-window:i+window+1]):
            pivots.append((i, highs[i]))
    return pivots

def find_pivot_lows(lows: np.ndarray, window: int = 5) -> list[tuple[int, float]]:
    pivots = []
    for i in range(window, len(lows) - window):
        if lows[i] == min(lows[i-window:i+window+1]):
            pivots.append((i, lows[i]))
    return pivots


# ════════════════════════════════════════════
#  ENGINE 1: MARKET REGIME
# ════════════════════════════════════════════

def market_regime_engine(ihsg_candles: list[Candle]) -> RegimeResult:
    closes = np.array([c.close for c in ihsg_candles])
    ma20 = sma(closes, 20)
    ma50 = sma(closes, 50)

    price = closes[-1]
    m20   = ma20[-1]
    m50   = ma50[-1]
    dist  = (price - m50) / m50 * 100

    if price > m50 and m20 > m50:
        regime = Regime.BULLISH
    elif price < m50 and m20 < m50:
        regime = Regime.BEARISH
    else:
        regime = Regime.SIDEWAYS

    return RegimeResult(
        regime=regime,
        ihsg_price=round(price, 0),
        ma20=round(m20, 0),
        ma50=round(m50, 0),
        distance_ma50_pct=round(dist, 2),
    )


# ════════════════════════════════════════════
#  ENGINE 2: MARKET STRUCTURE
# ════════════════════════════════════════════

def market_structure_engine(candles: list[Candle]) -> StructureResult:
    highs = np.array([c.high  for c in candles])
    lows  = np.array([c.low   for c in candles])

    pivot_h = find_pivot_highs(highs, window=5)
    pivot_l = find_pivot_lows(lows,   window=5)

    last_h = [v for _, v in pivot_h[-3:]]
    last_l = [v for _, v in pivot_l[-3:]]

    hh = hl = lh = ll = 0
    for i in range(1, len(last_h)):
        if last_h[i] > last_h[i-1]: hh += 1
        else: lh += 1
    for i in range(1, len(last_l)):
        if last_l[i] > last_l[i-1]: hl += 1
        else: ll += 1

    if hh >= 1 and hl >= 1:
        trend = Trend.UPTREND
        bos, choch = True, False
    elif lh >= 1 and ll >= 1:
        trend = Trend.DOWNTREND
        bos, choch = False, True
    else:
        trend = Trend.SIDEWAYS
        bos = choch = False

    swing_high = pivot_h[-1][1] if pivot_h else candles[-1].high
    swing_low  = pivot_l[-1][1] if pivot_l else candles[-1].low

    return StructureResult(
        trend=trend, hh_count=hh, hl_count=hl,
        lh_count=lh, ll_count=ll, bos=bos, choch=choch,
        swing_high=round(swing_high, 0),
        swing_low=round(swing_low, 0),
    )


# ════════════════════════════════════════════
#  ENGINE 3: TREND & MOMENTUM
# ════════════════════════════════════════════

def momentum_engine(candles: list[Candle]) -> MomentumResult:
    closes = np.array([c.close for c in candles])
    e20  = ema(closes, 20)
    e50  = ema(closes, 50)
    e100 = ema(closes, 100)

    price = closes[-1]
    ema20, ema50, ema100 = e20[-1], e50[-1], e100[-1]

    slope20 = (e20[-1] - e20[-6]) / e20[-6] * 100 if len(e20) > 6 else 0
    dist    = abs(price - ema20) / ema20 * 100

    if ema20 > ema50 > ema100:
        momentum = Momentum.BULLISH
    elif ema20 < ema50 < ema100:
        momentum = Momentum.BEARISH
    else:
        momentum = Momentum.NEUTRAL

    return MomentumResult(
        momentum=momentum,
        ema20=round(ema20, 0),
        ema50=round(ema50, 0),
        ema100=round(ema100, 0),
        slope20=round(slope20, 3),
        dist_from_ema20=round(dist, 2),
        price=round(price, 0),
    )


# ════════════════════════════════════════════
#  ENGINE 4: VOLATILITY
# ════════════════════════════════════════════

def volatility_engine(candles: list[Candle]) -> VolatilityResult:
    atr  = calc_atr(candles, 14)
    asma = sma(atr, 20)
    cur_atr  = atr[-1]
    cur_sma  = asma[-1]
    ratio    = cur_atr / cur_sma if cur_sma > 0 else 1.0
    cond     = VolCondition.EXPANSION if cur_atr > cur_sma else VolCondition.COMPRESSION

    return VolatilityResult(
        condition=cond,
        atr14=round(cur_atr, 1),
        atr_sma20=round(cur_sma, 1),
        ratio=round(ratio, 3),
    )


# ════════════════════════════════════════════
#  ENGINE 5: VOLUME
# ════════════════════════════════════════════

def volume_engine(candles: list[Candle]) -> VolumeResult:
    vols = np.array([c.volume for c in candles])
    vsma = sma(vols, 20)
    cur_vol = int(vols[-1])
    avg_vol = vsma[-1]
    rel_vol = cur_vol / avg_vol if avg_vol > 0 else 1.0

    if rel_vol > 2.0:   sig = VolSignal.SPIKE
    elif rel_vol > 1.5: sig = VolSignal.ELEVATED
    elif rel_vol < 0.5: sig = VolSignal.ABSORPTION
    else:               sig = VolSignal.NORMAL

    # Divergence: harga naik tapi volume turun (atau sebaliknya)
    recent_c = [c.close  for c in candles[-5:]]
    recent_v = [c.volume for c in candles[-5:]]
    price_up = recent_c[-1] > recent_c[0]
    vol_down = recent_v[-1] < recent_v[0]
    divergence = (price_up and vol_down) or (not price_up and not vol_down)

    return VolumeResult(
        signal=sig,
        rel_vol=round(rel_vol, 3),
        avg_vol=round(avg_vol, 0),
        cur_vol=cur_vol,
        divergence=divergence,
    )


# ════════════════════════════════════════════
#  ENGINE 6: LIQUIDITY
# ════════════════════════════════════════════

def liquidity_engine(candles: list[Candle], structure: StructureResult) -> LiquidityResult:
    sh = structure.swing_high
    sl = structure.swing_low
    last = candles[-1]
    prev = candles[-2] if len(candles) >= 2 else last

    sweep_high = prev.high > sh and last.close < sh
    sweep_low  = prev.low  < sl and last.close > sl
    stop_hunt  = sweep_high or sweep_low

    # Equal highs/lows (dalam 0.3%)
    close_enough = lambda a, b: abs(a - b) / (b or 1) < 0.003
    recent5 = candles[-5:]
    equal_highs = sum(1 for c in recent5 if close_enough(c.high, sh)) >= 2
    equal_lows  = sum(1 for c in recent5 if close_enough(c.low,  sl)) >= 2

    if stop_hunt:
        cond = LiqCondition.STOP_HUNT
    elif equal_highs or equal_lows:
        cond = LiqCondition.POOL_DETECTED
    else:
        cond = LiqCondition.CLEAR

    return LiquidityResult(
        condition=cond,
        stop_hunt=stop_hunt,
        sweep_high=sweep_high,
        sweep_low=sweep_low,
        equal_highs=equal_highs,
        equal_lows=equal_lows,
    )


# ════════════════════════════════════════════
#  ENGINE 7: BREAKOUT
# ════════════════════════════════════════════

def breakout_engine(
    candles: list[Candle],
    structure: StructureResult,
    vol: VolumeResult,
) -> BreakoutResult:
    sh = structure.swing_high
    sl = structure.swing_low
    last = candles[-1]
    prev = candles[-2] if len(candles) >= 2 else last

    vol_ok     = vol.rel_vol > 1.2
    big_candle = last.body_ratio > 0.6
    broke_h    = last.close > sh
    broke_l    = last.close < sl
    prev_broke_h = prev.high > sh and last.close < sh
    prev_broke_l = prev.low  < sl and last.close > sl

    # Range compression: 5 candle terakhir menyusut
    ranges = [c.range for c in candles[-8:]]
    compression = len(ranges) >= 4 and ranges[-1] < ranges[0] * 0.5

    if prev_broke_h or prev_broke_l:
        bt = BreakoutType.FALSE
    elif broke_h and vol_ok and big_candle:
        bt = BreakoutType.VALID_BULL
    elif broke_l and vol_ok and big_candle:
        bt = BreakoutType.VALID_BEAR
    elif (broke_h or broke_l):
        bt = BreakoutType.WEAK
    elif compression:
        bt = BreakoutType.COMPRESSION
    else:
        bt = BreakoutType.NONE

    return BreakoutResult(
        breakout_type=bt,
        broke_high=broke_h,
        broke_low=broke_l,
        compression=compression,
    )


# ════════════════════════════════════════════
#  ENGINE 8: CANDLESTICK PATTERNS
# ════════════════════════════════════════════

BULL_PATTERNS = {
    "HAMMER", "DRAGONFLY_DOJI", "BULL_ENGULFING",
    "BULL_MARUBOZU", "WIDE_RANGE_BULL", "MORNING_STAR",
}
BEAR_PATTERNS = {
    "SHOOTING_STAR", "GRAVESTONE_DOJI", "BEAR_ENGULFING",
    "BEAR_MARUBOZU", "WIDE_RANGE_BEAR", "EVENING_STAR", "PIN_BAR",
}

def candlestick_engine(candles: list[Candle]) -> CandleResult:
    c = candles[-1]
    p = candles[-2] if len(candles) >= 2 else c
    p2 = candles[-3] if len(candles) >= 3 else p

    br = c.body_ratio
    ur = c.upper_ratio
    lr = c.lower_ratio

    pattern = "NEUTRAL"

    # ── Doji variants
    if br < 0.1:
        if lr > 0.4:   pattern = "DRAGONFLY_DOJI"
        elif ur > 0.4: pattern = "GRAVESTONE_DOJI"
        else:          pattern = "DOJI"
    # ── Marubozu
    elif br > 0.88:
        pattern = "BULL_MARUBOZU" if c.bullish else "BEAR_MARUBOZU"
    # ── Hammer
    elif lr > 0.55 and br < 0.35 and c.bullish:
        pattern = "HAMMER"
    # ── Shooting Star
    elif ur > 0.55 and br < 0.35 and not c.bullish:
        pattern = "SHOOTING_STAR"
    # ── Pin Bar
    elif lr > 0.6 or ur > 0.6:
        pattern = "PIN_BAR"
    # ── Engulfing
    elif c.bullish and c.open < p.close and c.close > p.open and not p.bullish:
        pattern = "BULL_ENGULFING"
    elif not c.bullish and c.open > p.close and c.close < p.open and p.bullish:
        pattern = "BEAR_ENGULFING"
    # ── Morning Star (3-candle)
    elif c.bullish and not p2.bullish and p.body_ratio < 0.2 and c.close > p2.close * 0.99:
        pattern = "MORNING_STAR"
    # ── Evening Star (3-candle)
    elif not c.bullish and p2.bullish and p.body_ratio < 0.2 and c.close < p2.close * 1.01:
        pattern = "EVENING_STAR"
    # ── Inside Bar
    elif c.high < p.high and c.low > p.low:
        pattern = "INSIDE_BAR"
    # ── Wide Range
    elif br > 0.6:
        pattern = "WIDE_RANGE_BULL" if c.bullish else "WIDE_RANGE_BEAR"
    else:
        pattern = "SPINNING_TOP"

    return CandleResult(
        pattern=pattern,
        is_bullish=c.bullish,
        pattern_bull=pattern in BULL_PATTERNS,
        pattern_bear=pattern in BEAR_PATTERNS,
        body_ratio=round(br, 3),
        upper_ratio=round(ur, 3),
        lower_ratio=round(lr, 3),
    )


# ════════════════════════════════════════════
#  ENGINE 9: PROBABILITY SCORING MODEL
#  Weights: Trend(25) Mom(20) Vol(15)
#           Struct(15) Liq(10) Volat(10) Candle(5)
# ════════════════════════════════════════════

def probability_score(
    structure:  StructureResult,
    momentum:   MomentumResult,
    volatility: VolatilityResult,
    volume:     VolumeResult,
    liquidity:  LiquidityResult,
    candle:     CandleResult,
    regime:     RegimeResult,
) -> ScoreBreakdown:

    regime_mult = {
        Regime.BULLISH:  1.05,
        Regime.SIDEWAYS: 1.00,
        Regime.BEARISH:  0.95,
    }.get(regime.regime, 1.0)

    # TREND STRENGTH (25)
    if structure.trend == Trend.UPTREND:
        ts = min(25, 18 + (structure.hh_count + structure.hl_count) * 2)
    elif structure.trend == Trend.DOWNTREND:
        ts = min(25, 15 + (structure.lh_count + structure.ll_count) * 2)
    else:
        ts = 7

    # MOMENTUM (20)
    slope_bonus = min(4, abs(momentum.slope20) * 2)
    if momentum.momentum == Momentum.BULLISH:
        ms = min(20, 14 + slope_bonus)
    elif momentum.momentum == Momentum.BEARISH:
        ms = min(20, 11 + slope_bonus)
    else:
        ms = 5

    # VOLUME (15)
    vs = {
        VolSignal.SPIKE:      15,
        VolSignal.ELEVATED:   11,
        VolSignal.NORMAL:      7,
        VolSignal.ABSORPTION:  2,
    }.get(volume.signal, 5)
    if volume.divergence:
        vs = max(0, vs - 5)

    # MARKET STRUCTURE (15)
    ss = 0
    if structure.bos:   ss += 8
    if structure.choch: ss += 4
    if structure.hh_count >= 2 or structure.hl_count >= 2: ss += 3
    ss = min(15, ss)

    # LIQUIDITY (10)
    ls = {
        LiqCondition.CLEAR:         10,
        LiqCondition.POOL_DETECTED:  5,
        LiqCondition.STOP_HUNT:      1,
    }.get(liquidity.condition, 5)

    # VOLATILITY (10)
    vcs = min(10, 6 + volatility.ratio * 2) if volatility.condition == VolCondition.EXPANSION else 3

    # CANDLESTICK (5)
    cs = 5 if (candle.pattern_bull or candle.pattern_bear) else 3 if candle.pattern != "NEUTRAL" else 1

    raw_total = ts + ms + vs + ss + ls + vcs + cs
    total     = min(100, round(raw_total * regime_mult))

    return ScoreBreakdown(
        trend_score=ts,
        momentum_score=ms,
        volume_score=vs,
        structure_score=ss,
        liquidity_score=ls,
        volatility_score=round(vcs),
        candle_score=cs,
        total=total,
    )


# ════════════════════════════════════════════
#  ENGINE 10: SIGNAL GENERATION
# ════════════════════════════════════════════

def generate_signal(
    structure:  StructureResult,
    momentum:   MomentumResult,
    volatility: VolatilityResult,
    volume:     VolumeResult,
    liquidity:  LiquidityResult,
    breakout:   BreakoutResult,
    candle:     CandleResult,
    score:      ScoreBreakdown,
) -> Signal:

    vol_exp     = volatility.condition == VolCondition.EXPANSION
    vol_confirm = volume.signal in (VolSignal.SPIKE, VolSignal.ELEVATED)
    no_trap     = liquidity.condition != LiqCondition.STOP_HUNT
    not_ext     = momentum.dist_from_ema20 < 5.0  # Extension control

    buy_conds = [
        structure.trend == Trend.UPTREND,
        momentum.momentum == Momentum.BULLISH,
        breakout.breakout_type == BreakoutType.VALID_BULL or candle.pattern_bull,
        vol_confirm,
        no_trap,
        vol_exp,
        not_ext,
    ]
    sell_conds = [
        structure.trend == Trend.DOWNTREND,
        momentum.momentum == Momentum.BEARISH,
        breakout.breakout_type == BreakoutType.VALID_BEAR or candle.pattern_bear,
        vol_confirm,
        vol_exp,
    ]

    buy_met  = sum(bool(c) for c in buy_conds)
    sell_met = sum(bool(c) for c in sell_conds)

    if score.total >= 70 and buy_met >= 5:
        return Signal.BUY
    if score.total >= 70 and sell_met >= 4:
        return Signal.SELL
    return Signal.WAIT


# ════════════════════════════════════════════
#  ENGINE 11: RISK MANAGEMENT
# ════════════════════════════════════════════

def _nearest_swing_low(candles: list[Candle], lookback: int = 20) -> float:
    """
    Cari swing low TERDEKAT dalam N candle terakhir.
    Lebih relevan dari swing low global yang bisa sangat jauh.
    """
    recent = candles[-lookback:] if len(candles) >= lookback else candles
    pivots = find_pivot_lows(np.array([c.low for c in recent]), window=3)
    if pivots:
        # Ambil swing low paling dekat (index tertinggi = paling recent)
        return pivots[-1][1]
    # Fallback: low terendah dari 5 candle terakhir
    return min(c.low for c in candles[-5:])


def _nearest_swing_high(candles: list[Candle], lookback: int = 20) -> float:
    """Cari swing high TERDEKAT dalam N candle terakhir."""
    recent = candles[-lookback:] if len(candles) >= lookback else candles
    pivots = find_pivot_highs(np.array([c.high for c in recent]), window=3)
    if pivots:
        return pivots[-1][1]
    return max(c.high for c in candles[-5:])


def risk_engine(
    candles:   list[Candle],
    structure: StructureResult,
    signal:    Signal,
    capital:   float = 100_000_000,
    risk_pct:  float = 0.015,
) -> RiskParams:
    last  = candles[-1]
    entry = last.close
    atr   = calc_atr(candles, 14)[-1]

    # Batas max SL: 7% dari entry (tidak boleh lebih jauh)
    MAX_SL_PCT = 0.07
    # Minimum SL: 1x ATR dari entry (tidak boleh terlalu ketat)
    MIN_SL_DIST = atr * 1.0

    if signal == Signal.BUY:
        # Gunakan swing low TERDEKAT (20 candle), bukan global swing low
        nearest_sl = _nearest_swing_low(candles, lookback=20)
        buffer     = atr * 0.3   # sedikit di bawah swing low
        sl_raw     = nearest_sl - buffer

        # Cap: SL tidak boleh lebih dari 7% di bawah entry
        sl_min_cap = entry * (1 - MAX_SL_PCT)
        sl = max(sl_raw, sl_min_cap)

        # Floor: SL minimal 1 ATR di bawah entry (tidak terlalu ketat)
        sl = min(sl, entry - MIN_SL_DIST)

        sl_dist = entry - sl
        tp1 = entry + sl_dist * 2   # RR 1:2
        tp2 = entry + sl_dist * 3   # RR 1:3

    elif signal == Signal.SELL:
        nearest_sh = _nearest_swing_high(candles, lookback=20)
        buffer     = atr * 0.3
        sl_raw     = nearest_sh + buffer

        sl_max_cap = entry * (1 + MAX_SL_PCT)
        sl = min(sl_raw, sl_max_cap)
        sl = max(sl, entry + MIN_SL_DIST)

        sl_dist = sl - entry
        tp1 = entry - sl_dist * 2
        tp2 = entry - sl_dist * 3

    else:
        # WAIT: gunakan ATR sederhana
        sl_dist = atr * 1.5
        sl  = entry - sl_dist
        tp1 = entry + sl_dist * 2
        tp2 = entry + sl_dist * 3

    sl_dist  = abs(entry - sl)
    rr1      = round(abs(tp1 - entry) / sl_dist, 2) if sl_dist > 0 else 2.0
    rr2      = round(abs(tp2 - entry) / sl_dist, 2) if sl_dist > 0 else 3.0
    risk_idr = capital * risk_pct
    lot_size = int((risk_idr / sl_dist) / 100) * 100 if sl_dist > 0 else 0

    return RiskParams(
        entry=round(entry, 0),
        stop_loss=round(sl, 0),
        tp1=round(tp1, 0),
        tp2=round(tp2, 0),
        rr1=rr1,
        rr2=rr2,
        lot_size=max(100, lot_size),
        risk_idr=round(risk_idr, 0),
    )


# ════════════════════════════════════════════
#  MULTI TIMEFRAME CONFIRMATION
# ════════════════════════════════════════════

def mtf_alignment(
    daily_candles: list[Candle],
    h1_candles:    Optional[list[Candle]],
    m15_candles:   Optional[list[Candle]],
) -> str:
    """
    Returns alignment string:
    STRONG_BULL / STRONG_BEAR / PARTIAL_BULL / PARTIAL_BEAR / MIXED
    """
    scores = []
    for clist in [daily_candles, h1_candles, m15_candles]:
        if clist and len(clist) >= 20:
            m = momentum_engine(clist)
            scores.append(m.momentum)

    bull = scores.count(Momentum.BULLISH)
    bear = scores.count(Momentum.BEARISH)

    if bull == 3:    return "STRONG_BULL"
    if bear == 3:    return "STRONG_BEAR"
    if bull == 2:    return "PARTIAL_BULL"
    if bear == 2:    return "PARTIAL_BEAR"
    return "MIXED"


# ════════════════════════════════════════════
#  MASTER ANALYSIS FUNCTION
# ════════════════════════════════════════════

def analyze_stock(
    symbol:       str,
    name:         str,
    candles:      list[dict],
    ihsg_candles: list[dict],
    capital:      float = 100_000_000,
    timeframe:    str   = "1D",
    h1_candles:   Optional[list[dict]] = None,
    m15_candles:  Optional[list[dict]] = None,
) -> Optional[TradeResult]:
    """
    Jalankan semua engine dan hasilkan TradeResult lengkap.
    """
    if not candles or len(candles) < 60:
        return None

    # Convert ke Candle objects
    c_list      = to_candles(candles)
    ihsg_list   = to_candles(ihsg_candles)
    h1_list     = to_candles(h1_candles)  if h1_candles  else None
    m15_list    = to_candles(m15_candles) if m15_candles  else None

    # Run engines
    regime   = market_regime_engine(ihsg_list)
    struct   = market_structure_engine(c_list)
    momentum = momentum_engine(c_list)
    vol      = volatility_engine(c_list)
    volume   = volume_engine(c_list)
    liq      = liquidity_engine(c_list, struct)
    brk      = breakout_engine(c_list, struct, volume)
    candle   = candlestick_engine(c_list)
    score    = probability_score(struct, momentum, vol, volume, liq, candle, regime)
    signal   = generate_signal(struct, momentum, vol, volume, liq, brk, candle, score)
    risk     = risk_engine(c_list, struct, signal, capital)
    mtf      = mtf_alignment(c_list, h1_list, m15_list)

    last    = c_list[-1]
    prev    = c_list[-2] if len(c_list) >= 2 else last
    chg_pct = (last.close - prev.close) / prev.close * 100

    return TradeResult(
        ticker=symbol,
        name=name,
        signal=signal,
        score=score,
        risk=risk,
        regime=regime,
        structure=struct,
        momentum=momentum,
        volatility=vol,
        volume=volume,
        liquidity=liq,
        breakout=brk,
        candle=candle,
        price=round(last.close, 0),
        change_pct=round(chg_pct, 2),
        timeframe=timeframe,
        mtf_alignment=mtf,
    )
