"""
us30_regime.py — US30 Monitor

Two regimes, and they do different jobs:

  DIRECTIONAL REGIME (trend / chop / mean-revert) from ADX, ATR expansion and
  the efficiency ratio. Feeds the ±10 regime layer and gates C3.

  DISPERSION REGIME from the cross-sectional spread of the 30 components'
  returns. High dispersion means stock-picking is driving the tape and
  index-level technical signals degrade; low dispersion means macro is driving
  everything and index signals work. This SCALES the whole signal's confidence,
  it never flips its direction — a distinction worth keeping strict, because
  treating a confidence input as a directional one is how confluence systems
  end up double-counting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
import data_layer as dl
from us30_technicals import atr


# ==========================================================================
# Primitives
# ==========================================================================
def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Wilder's ADX. Trend strength only — carries no direction."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)

    a = 1.0 / period
    atr_s = tr.ewm(alpha=a, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=a, adjust=False, min_periods=period).mean() / atr_s
    minus_di = 100.0 * minus_dm.ewm(alpha=a, adjust=False, min_periods=period).mean() / atr_s

    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=a, adjust=False, min_periods=period).mean()


def efficiency_ratio(close: pd.Series, period: int = 20) -> float:
    """
    Kaufman efficiency: net displacement over total path travelled.
    1.0 = a straight line, 0.0 = pure noise around a level.
    """
    s = close.dropna().tail(period + 1)
    if len(s) < period + 1:
        return float("nan")
    net = abs(float(s.iloc[-1] - s.iloc[0]))
    path = float(s.diff().abs().sum())
    return net / path if path > 0 else float("nan")


# ==========================================================================
# Report
# ==========================================================================
@dataclass
class RegimeReport:
    ok: bool = False
    note: str = ""

    regime: str = "UNKNOWN"          # TREND / CHOP / MEAN_REVERT / TRANSITION
    direction: str = "NEUTRAL"       # UP / DOWN / NEUTRAL
    adx_value: float = float("nan")
    efficiency: float = float("nan")
    atr_expansion: float = float("nan")   # ATR14 / ATR50

    dispersion: float = float("nan")      # cross-sectional stdev of returns, %
    dispersion_percentile: float = float("nan")
    dispersion_regime: str = "UNKNOWN"    # MACRO_DRIVEN / NORMAL / STOCK_PICKING
    signal_confidence_scalar: float = 1.0

    score: float = 0.0
    confidence: float = 0.0
    flags: list[str] = field(default_factory=list)


def compute_regime(daily: pd.DataFrame,
                   component_closes: pd.DataFrame | None = None) -> RegimeReport:
    """Pure function. `daily` is ^DJI daily OHLC."""
    rep = RegimeReport()
    if daily is None or daily.empty or len(daily) < 55:
        rep.note = "need at least 55 daily bars for regime"
        return rep

    d = daily.dropna(subset=["Close"])
    high, low, close = d["High"], d["Low"], d["Close"]

    adx_s = adx(high, low, close)
    if adx_s.notna().any():
        rep.adx_value = float(adx_s.iloc[-1])
    rep.efficiency = efficiency_ratio(close, 20)

    a14 = atr(high, low, close, 14)
    a50 = atr(high, low, close, 50)
    if a14.notna().any() and a50.notna().any() and float(a50.iloc[-1]) > 0:
        rep.atr_expansion = float(a14.iloc[-1]) / float(a50.iloc[-1])

    # ---- directional regime ---------------------------------------------
    trending = (np.isfinite(rep.adx_value) and rep.adx_value >= 25) or \
               (np.isfinite(rep.efficiency) and rep.efficiency >= 0.45)
    choppy = (np.isfinite(rep.adx_value) and rep.adx_value < 18) and \
             (not np.isfinite(rep.efficiency) or rep.efficiency < 0.30)

    if trending and not choppy:
        rep.regime = "TREND"
    elif choppy:
        rep.regime = "CHOP"
    elif np.isfinite(rep.efficiency) and rep.efficiency < 0.25 \
            and np.isfinite(rep.atr_expansion) and rep.atr_expansion > 1.2:
        rep.regime = "MEAN_REVERT"
    else:
        rep.regime = "TRANSITION"

    if len(close) >= 21:
        net = float(close.iloc[-1] - close.iloc[-21])
        if np.isfinite(net) and abs(net) > 0:
            rep.direction = "UP" if net > 0 else "DOWN"

    # ---- dispersion regime ------------------------------------------------
    if component_closes is not None and not component_closes.empty \
            and len(component_closes) >= 2:
        rets = component_closes.pct_change().dropna(how="all")
        if not rets.empty:
            cross = rets.std(axis=1, skipna=True) * 100.0
            cross = cross.dropna()
            if not cross.empty:
                rep.dispersion = float(cross.iloc[-1])
                if len(cross) >= 20:
                    hist = cross.tail(60)
                    rep.dispersion_percentile = float(
                        (hist <= rep.dispersion).mean() * 100.0
                    )

    if np.isfinite(rep.dispersion_percentile):
        if rep.dispersion_percentile >= 75:
            rep.dispersion_regime = "STOCK_PICKING"
            rep.signal_confidence_scalar = 0.72
            rep.flags.append(
                f"High dispersion ({rep.dispersion:.2f}%, "
                f"{rep.dispersion_percentile:.0f}th pct) — index-level technicals degrade"
            )
        elif rep.dispersion_percentile <= 30:
            rep.dispersion_regime = "MACRO_DRIVEN"
            rep.signal_confidence_scalar = 1.12
            rep.flags.append(
                f"Low dispersion ({rep.dispersion:.2f}%) — macro driving all 30, "
                f"index signals reliable"
            )
        else:
            rep.dispersion_regime = "NORMAL"

    # ---- score -------------------------------------------------------------
    # Regime supplies conviction in the prevailing direction, not a new direction.
    magnitude = 0.0
    if rep.regime == "TREND":
        magnitude = 1.0
    elif rep.regime == "TRANSITION":
        magnitude = 0.3
    elif rep.regime == "MEAN_REVERT":
        magnitude = -0.4      # leans against continuation
    elif rep.regime == "CHOP":
        magnitude = 0.0

    sign = {"UP": 1.0, "DOWN": -1.0, "NEUTRAL": 0.0}[rep.direction]
    rep.score = float(magnitude * sign * config.LAYER_WEIGHTS["regime"])

    conf = 1.0 if np.isfinite(rep.adx_value) else 0.6
    if not np.isfinite(rep.dispersion):
        conf *= 0.85
    rep.confidence = float(np.clip(conf, 0.0, 1.0))
    rep.ok = True
    rep.note = (
        f"{rep.regime}/{rep.direction}, ADX {rep.adx_value:.1f}, "
        f"ER {rep.efficiency:.2f}, dispersion {rep.dispersion_regime}"
    )
    return rep


def get_regime() -> RegimeReport:
    try:
        daily = dl.ohlcv(dl.get_index_daily(period="1y"))
        comp = dl.closes(dl.get_components_daily(), config.COMPONENTS)
        return compute_regime(daily, comp)
    except Exception as exc:  # noqa: BLE001
        return RegimeReport(note=f"regime failed: {exc}")
