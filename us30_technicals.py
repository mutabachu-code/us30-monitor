"""
us30_technicals.py — US30 Monitor

CPR, pivots, EMA stack, RSI with momentum decay, ATR, VWAP, divergence and
mean-reversion setups.

TWO INSTRUMENT RULES, both load-bearing:

  * CPR and pivots use ^DJI daily H/L/C. No volume needed, so the cash index
    is the right and most accurate source.
  * VWAP and RVOL must NEVER touch ^DJI — Yahoo reports its volume as 0, so a
    VWAP computed there is a silent divide-by-zero that returns a plausible
    looking number. They come from YM=F or DIA, rescaled to index points.

CPR width is classified against ATR, not against absolute points: DJIA at
~52,700 has a completely different point scale from NAS100 and a fixed
threshold ported across would misclassify every session.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
import data_layer as dl


# ==========================================================================
# Indicator primitives (pure, testable)
# ==========================================================================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, period: int = config.RSI_PERIOD) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    # Zero average loss means an unbroken advance: RSI is 100 by definition.
    return out.where(avg_loss.ne(0.0) | avg_gain.eq(0.0), 100.0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = config.ATR_PERIOD) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series) -> pd.Series:
    """
    Session VWAP. Returns NaN where cumulative volume is zero rather than
    inf/garbage — that NaN is the signal that you pointed this at ^DJI.
    """
    v = pd.to_numeric(volume, errors="coerce").fillna(0.0)
    typical = (high + low + close) / 3.0
    cum_v = v.cumsum()
    cum_pv = (typical * v).cumsum()
    return (cum_pv / cum_v.replace(0.0, np.nan))


def rsi_momentum_decay(rsi_series: pd.Series,
                       lookback: int = config.RSI_DECAY_LOOKBACK) -> float:
    """
    Rate of change of RSI itself. Price can keep making highs while the RSI
    slope rolls over — that decay leads the actual reversal. Negative = the
    move is losing force, positive = it is gaining.
    """
    s = rsi_series.dropna()
    if len(s) < lookback + 1:
        return 0.0
    return float(s.iloc[-1] - s.iloc[-1 - lookback]) / lookback


def find_divergence(price: pd.Series, osc: pd.Series,
                    lookback: int = 40, pivot: int = 3) -> str:
    """
    Classic regular divergence over the last `lookback` bars.
    Returns 'bullish', 'bearish' or 'none'. Pivots need `pivot` bars of
    confirmation on each side, so the two most recent bars can never form one.
    """
    p = price.dropna().tail(lookback)
    o = osc.reindex(p.index).dropna()
    p = p.reindex(o.index)
    if len(p) < pivot * 2 + 5:
        return "none"

    pv = p.to_numpy()
    ov = o.to_numpy()
    highs, lows = [], []
    for i in range(pivot, len(pv) - pivot):
        window = pv[i - pivot:i + pivot + 1]
        if pv[i] == window.max() and (window.argmax() == pivot):
            highs.append(i)
        if pv[i] == window.min() and (window.argmin() == pivot):
            lows.append(i)

    if len(highs) >= 2:
        a, b = highs[-2], highs[-1]
        if pv[b] > pv[a] and ov[b] < ov[a]:
            return "bearish"
    if len(lows) >= 2:
        a, b = lows[-2], lows[-1]
        if pv[b] < pv[a] and ov[b] > ov[a]:
            return "bullish"
    return "none"


# ==========================================================================
# CPR
# ==========================================================================
@dataclass
class CPR:
    pivot: float = float("nan")
    bc: float = float("nan")
    tc: float = float("nan")
    width: float = float("nan")
    width_atr: float = float("nan")
    classification: str = "UNKNOWN"
    virgin: bool = False
    r1: float = float("nan")
    r2: float = float("nan")
    r3: float = float("nan")
    s1: float = float("nan")
    s2: float = float("nan")
    s3: float = float("nan")
    position: str = "UNKNOWN"     # ABOVE / INSIDE / BELOW
    signal: str = "NEUTRAL"


def compute_cpr(prev_high: float, prev_low: float, prev_close: float,
                atr20: float, current: float,
                today_high: float | None = None,
                today_low: float | None = None) -> CPR:
    c = CPR()
    if not all(np.isfinite(x) for x in (prev_high, prev_low, prev_close)):
        return c

    c.pivot = (prev_high + prev_low + prev_close) / 3.0
    c.bc = (prev_high + prev_low) / 2.0
    c.tc = 2.0 * c.pivot - c.bc
    if c.tc < c.bc:
        c.tc, c.bc = c.bc, c.tc
    c.width = c.tc - c.bc

    rng = prev_high - prev_low
    c.r1 = 2.0 * c.pivot - prev_low
    c.s1 = 2.0 * c.pivot - prev_high
    c.r2 = c.pivot + rng
    c.s2 = c.pivot - rng
    c.r3 = prev_high + 2.0 * (c.pivot - prev_low)
    c.s3 = prev_low - 2.0 * (prev_high - c.pivot)

    # Classify against ATR, never against absolute points.
    if np.isfinite(atr20) and atr20 > 0:
        c.width_atr = c.width / atr20
        if c.width_atr < config.CPR_NARROW_ATR:
            c.classification = "NARROW"      # trend day expected
        elif c.width_atr > config.CPR_WIDE_ATR:
            c.classification = "WIDE"        # range / chop day expected
        else:
            c.classification = "MODERATE"

    if np.isfinite(current):
        if current > c.tc:
            c.position = "ABOVE"
        elif current < c.bc:
            c.position = "BELOW"
        else:
            c.position = "INSIDE"

    # Virgin CPR: price has not touched the CPR band today at all.
    if today_high is not None and today_low is not None \
            and np.isfinite(today_high) and np.isfinite(today_low):
        c.virgin = (today_low > c.tc) or (today_high < c.bc)

    if c.position == "ABOVE" and c.classification == "NARROW":
        c.signal = "BULLISH_TREND"
    elif c.position == "BELOW" and c.classification == "NARROW":
        c.signal = "BEARISH_TREND"
    elif c.classification == "WIDE":
        c.signal = "RANGE_FADE"
    elif c.position == "INSIDE":
        c.signal = "NEUTRAL"
    elif c.position == "ABOVE":
        c.signal = "BULLISH_BIAS"
    elif c.position == "BELOW":
        c.signal = "BEARISH_BIAS"
    return c


# ==========================================================================
# Report
# ==========================================================================
@dataclass
class TechnicalReport:
    ok: bool = False
    note: str = ""

    price: float = float("nan")
    atr14: float = float("nan")
    atr20_daily: float = float("nan")

    cpr: CPR = field(default_factory=CPR)

    ema_values: dict[int, float] = field(default_factory=dict)
    ema_stack: str = "MIXED"          # BULLISH / BEARISH / MIXED

    rsi_value: float = float("nan")
    rsi_decay: float = 0.0
    rsi_state: str = "NEUTRAL"

    vwap_value: float = float("nan")
    vwap_slope: float = 0.0
    vwap_distance_pts: float = float("nan")
    vwap_source: str = "unavailable"

    divergence: str = "none"
    mean_reversion: str = "none"

    score: float = 0.0
    confidence: float = 0.0
    flags: list[str] = field(default_factory=list)


def _ema_stack(values: dict[int, float], price: float) -> str:
    need = [p for p in config.EMA_PERIODS if p in values and np.isfinite(values[p])]
    if len(need) < 3 or not np.isfinite(price):
        return "MIXED"
    ordered = [values[p] for p in sorted(need)]
    if all(ordered[i] > ordered[i + 1] for i in range(len(ordered) - 1)) and price > ordered[0]:
        return "BULLISH"
    if all(ordered[i] < ordered[i + 1] for i in range(len(ordered) - 1)) and price < ordered[0]:
        return "BEARISH"
    return "MIXED"


def compute_technicals(daily: pd.DataFrame,
                       intraday: pd.DataFrame | None = None,
                       vol_bars: pd.DataFrame | None = None,
                       vol_scale: float = 1.0,
                       vol_source: str = "unavailable") -> TechnicalReport:
    """
    Pure function.
      daily     : ^DJI daily OHLC  (CPR, pivots, ATR20, EMA stack, divergence)
      intraday  : ^DJI intraday    (live price, RSI, today's H/L)
      vol_bars  : YM=F or DIA intraday OHLCV (VWAP, and ONLY VWAP)
      vol_scale : multiply vol_bars prices by this to reach index points
    """
    rep = TechnicalReport()
    if daily is None or daily.empty or len(daily) < 25:
        rep.note = "need at least 25 daily bars"
        return rep

    d = daily.dropna(subset=["Close"])
    close_d, high_d, low_d = d["Close"], d["High"], d["Low"]

    atr_d = atr(high_d, low_d, close_d, config.ATR_PERIOD)
    rep.atr14 = float(atr_d.iloc[-1]) if atr_d.notna().any() else float("nan")
    atr20 = atr(high_d, low_d, close_d, 20)
    rep.atr20_daily = float(atr20.iloc[-1]) if atr20.notna().any() else float("nan")

    # ---- live price, today's range -------------------------------------
    today_high = today_low = None
    if intraday is not None and not intraday.empty:
        i = intraday.dropna(subset=["Close"])
        if not i.empty:
            rep.price = float(i["Close"].iloc[-1])
            today_high = float(i["High"].max())
            today_low = float(i["Low"].min())
            r_intra = rsi(i["Close"])
            if r_intra.notna().any():
                rep.rsi_value = float(r_intra.iloc[-1])
                rep.rsi_decay = rsi_momentum_decay(r_intra)
            rep.divergence = find_divergence(i["Close"], r_intra)
    if not np.isfinite(rep.price):
        rep.price = float(close_d.iloc[-1])
        r_daily = rsi(close_d)
        if r_daily.notna().any():
            rep.rsi_value = float(r_daily.iloc[-1])
            rep.rsi_decay = rsi_momentum_decay(r_daily)
        rep.divergence = find_divergence(close_d, r_daily)

    # ---- CPR from the PREVIOUS completed session ------------------------
    rep.cpr = compute_cpr(
        prev_high=float(high_d.iloc[-2]) if len(d) > 1 else float("nan"),
        prev_low=float(low_d.iloc[-2]) if len(d) > 1 else float("nan"),
        prev_close=float(close_d.iloc[-2]) if len(d) > 1 else float("nan"),
        atr20=rep.atr20_daily,
        current=rep.price,
        today_high=today_high,
        today_low=today_low,
    )

    # ---- EMA stack ------------------------------------------------------
    for p in config.EMA_PERIODS:
        e = ema(close_d, p)
        if e.notna().any():
            rep.ema_values[p] = float(e.iloc[-1])
    rep.ema_stack = _ema_stack(rep.ema_values, rep.price)

    # ---- RSI state ------------------------------------------------------
    if np.isfinite(rep.rsi_value):
        if rep.rsi_value >= 70:
            rep.rsi_state = "OVERBOUGHT"
        elif rep.rsi_value <= 30:
            rep.rsi_state = "OVERSOLD"
        elif rep.rsi_value >= 55:
            rep.rsi_state = "BULLISH"
        elif rep.rsi_value <= 45:
            rep.rsi_state = "BEARISH"
        if abs(rep.rsi_decay) > 1.5:
            rep.flags.append(
                f"RSI momentum {'decaying' if rep.rsi_decay < 0 else 'building'} "
                f"at {rep.rsi_decay:+.1f}/bar"
            )

    # ---- VWAP — volume-bearing source ONLY ------------------------------
    if vol_bars is not None and not vol_bars.empty and "Volume" in vol_bars:
        v = vol_bars.dropna(subset=["Close"])
        total_vol = pd.to_numeric(v["Volume"], errors="coerce").fillna(0.0).sum()
        if total_vol > 0:
            vw = vwap(v["High"], v["Low"], v["Close"], v["Volume"])
            if vw.notna().any():
                rep.vwap_value = float(vw.iloc[-1]) * vol_scale
                rep.vwap_source = vol_source
                tail = vw.dropna().tail(12)
                if len(tail) >= 4:
                    slope = np.polyfit(range(len(tail)), tail.to_numpy(), 1)[0]
                    rep.vwap_slope = float(slope * vol_scale)
                if np.isfinite(rep.price):
                    rep.vwap_distance_pts = rep.price - rep.vwap_value
        else:
            rep.flags.append(
                "Volume source returned zero volume — VWAP suppressed "
                "(this is what happens if ^DJI is used)"
            )
    else:
        rep.flags.append("No volume source — VWAP and RVOL unavailable")

    # ---- mean reversion --------------------------------------------------
    if np.isfinite(rep.vwap_distance_pts) and np.isfinite(rep.atr14) and rep.atr14 > 0:
        stretch = rep.vwap_distance_pts / rep.atr14
        if stretch > 1.5 and rep.rsi_state == "OVERBOUGHT":
            rep.mean_reversion = "short_setup"
        elif stretch < -1.5 and rep.rsi_state == "OVERSOLD":
            rep.mean_reversion = "long_setup"

    # ---- score -----------------------------------------------------------
    parts: list[float] = []
    cpr_map = {
        "BULLISH_TREND": 1.0, "BULLISH_BIAS": 0.5, "NEUTRAL": 0.0,
        "RANGE_FADE": 0.0, "BEARISH_BIAS": -0.5, "BEARISH_TREND": -1.0,
    }
    parts.append(cpr_map.get(rep.cpr.signal, 0.0))
    parts.append({"BULLISH": 1.0, "BEARISH": -1.0, "MIXED": 0.0}[rep.ema_stack])

    if np.isfinite(rep.rsi_value):
        parts.append(float(np.clip((rep.rsi_value - 50.0) / 25.0, -1.0, 1.0)))
    if np.isfinite(rep.vwap_distance_pts) and np.isfinite(rep.atr14) and rep.atr14 > 0:
        parts.append(float(np.clip(rep.vwap_distance_pts / rep.atr14, -1.0, 1.0)) * 0.6)
        parts.append(float(np.sign(rep.vwap_slope)) * 0.4)

    base = float(np.mean(parts)) if parts else 0.0

    # RSI decay opposing the move drags the score toward neutral.
    if np.isfinite(rep.rsi_value) and abs(rep.rsi_decay) > 1.0 \
            and np.sign(rep.rsi_decay) != np.sign(base) and base != 0:
        base *= 0.6
        rep.flags.append("Score damped: RSI momentum opposing price direction")

    if rep.divergence == "bearish" and base > 0:
        base *= 0.4
        rep.flags.append("Bearish divergence against a long read")
    elif rep.divergence == "bullish" and base < 0:
        base *= 0.4
        rep.flags.append("Bullish divergence against a short read")

    rep.score = float(np.clip(base, -1.0, 1.0) * config.LAYER_WEIGHTS["technicals"])

    conf = 1.0
    if rep.vwap_source == "unavailable":
        conf *= 0.75
    if intraday is None or intraday.empty:
        conf *= 0.8
    if rep.cpr.classification == "UNKNOWN":
        conf *= 0.7
    rep.confidence = float(np.clip(conf, 0.0, 1.0))
    rep.ok = True
    rep.note = f"CPR {rep.cpr.classification}/{rep.cpr.position}, VWAP via {rep.vwap_source}"
    return rep


# ==========================================================================
# Fetch + compute
# ==========================================================================
def get_technicals() -> TechnicalReport:
    try:
        daily = dl.ohlcv(dl.get_index_daily(period="1y"))
        intraday = dl.ohlcv(dl.get_index_intraday(interval="5m", period="1d"))

        vol_fetch = dl.get_volume_source(interval="5m", period="1d")
        vol_bars = dl.ohlcv(vol_fetch)
        scale, source = 1.0, "unavailable"

        if vol_fetch and not vol_bars.empty:
            source = vol_fetch.source
            last_v = float(vol_bars["Close"].iloc[-1])
            ref = float(intraday["Close"].iloc[-1]) if not intraday.empty \
                else (float(daily["Close"].iloc[-1]) if not daily.empty else np.nan)
            # Derive the scale live rather than trusting a 100x constant.
            if np.isfinite(ref) and last_v > 0:
                raw = ref / last_v
                scale = 100.0 if 50.0 < raw < 200.0 else (1.0 if 0.9 < raw < 1.1 else raw)

        return compute_technicals(daily, intraday, vol_bars, scale, source)
    except Exception as exc:  # noqa: BLE001
        return TechnicalReport(note=f"technicals failed: {exc}")
