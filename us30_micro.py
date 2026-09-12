"""
us30_micro.py — US30 Monitor, PHASE 4 (L3, ±15)

Futures and microstructure: cash-futures basis, overnight and prior-day levels,
relative volume, a bar-derived delta proxy, liquidity sweep detection, and the
round-trip cost budget that C9 enforces.

THREE HONESTY CONSTRAINTS, stated up front because they bound what this layer
can legitimately claim:

  1. DELTA HERE IS A PROXY. Real delta needs Level 2 / footprint data, which
     yfinance does not carry. What this computes is the close's position within
     each bar's range, weighted by that bar's volume. It correlates with real
     delta on trending bars and is close to meaningless on inside bars. Every
     surface that shows it says so, and it is weighted accordingly.

  2. BASIS IS RTH-ONLY. ^DJI does not tick outside 09:30-16:00 ET, so a
     "basis" computed overnight is really just the futures price minus a
     16:00 snapshot. Outside RTH the basis is returned as NaN and C7 does not
     fire, rather than firing on a number that means nothing.

  3. RVOL IS SAME-TIME-OF-DAY. Measured against the same 5-minute slot on
     prior sessions, never a flat daily average — 09:35 always looks like a
     spike beside 12:35, and a naive RVOL would fire a volume-expansion signal
     at every open.

Spread and slippage are INPUTS to this module, not outputs. They come from the
MT5 terminal (`symbol_info_tick`, and your own fill history). yfinance cannot
supply either one reliably.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
import data_layer as dl
from us30_technicals import atr


# ==========================================================================
# Session helpers
# ==========================================================================
def to_et(df: pd.DataFrame) -> pd.DataFrame:
    """Force a DatetimeIndex into US/Eastern. Naive indexes are assumed UTC."""
    if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df if df is not None else pd.DataFrame()
    idx = df.index
    out = df.copy()
    out.index = (idx.tz_localize("UTC") if idx.tz is None else idx).tz_convert(config.MARKET_TZ)
    return out


def session_dates(index: pd.DatetimeIndex) -> pd.Series:
    """
    The trading session each bar belongs to. Globex opens at 18:00 ET for the
    NEXT session, so a Sunday 20:00 bar is part of Monday — getting this wrong
    silently mixes two sessions' overnight ranges together.
    """
    if len(index) == 0:
        return pd.Series(dtype="datetime64[ns]")
    ts = pd.Series(index, index=index)
    base = ts.dt.normalize().dt.tz_localize(None)

    globex_minute = int(config.GLOBEX_START[:2]) * 60 + int(config.GLOBEX_START[3:])
    after_globex = (ts.dt.hour * 60 + ts.dt.minute) >= globex_minute
    sess = base.where(~after_globex, base + pd.Timedelta(days=1))

    # Roll Saturday and Sunday stamps forward to Monday: Friday's 18:00 bars do
    # not exist, but a Sunday-evening Globex bar belongs to Monday's session.
    dow = sess.dt.dayofweek
    return sess + pd.to_timedelta(
        np.where(dow == 5, 2, np.where(dow == 6, 1, 0)), unit="D"
    )


def rth_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True for bars inside 09:30-16:00 ET on a weekday."""
    if len(index) == 0:
        return np.zeros(0, dtype=bool)
    minutes = index.hour * 60 + index.minute
    start = int(config.RTH_START[:2]) * 60 + int(config.RTH_START[3:])
    end = int(config.RTH_END[:2]) * 60 + int(config.RTH_END[3:])
    return (minutes >= start) & (minutes < end) & (index.dayofweek < 5)


# ==========================================================================
# Components
# ==========================================================================
@dataclass
class Levels:
    overnight_high: float = float("nan")
    overnight_low: float = float("nan")
    prior_high: float = float("nan")
    prior_low: float = float("nan")
    prior_close: float = float("nan")
    ib_high: float = float("nan")       # initial balance, first 60 RTH minutes
    ib_low: float = float("nan")
    on_range_position: float = float("nan")   # 0 = at ONL, 1 = at ONH
    state: str = "UNKNOWN"              # INSIDE_ON / ABOVE_ON / BELOW_ON


@dataclass
class Sweep:
    side: str                # "high" or "low"
    level_name: str
    level: float
    penetration: float
    rvol: float
    confirmed: bool
    bars_ago: int

    @property
    def implication(self) -> str:
        return "bearish" if self.side == "high" else "bullish"


@dataclass
class MicroReport:
    ok: bool = False
    note: str = ""
    source: str = ""

    price: float = float("nan")
    atr_intraday: float = float("nan")

    # basis
    basis: float = float("nan")
    basis_mean: float = float("nan")
    basis_std: float = float("nan")
    basis_sigma: float = float("nan")     # C7 reads this
    basis_available: bool = False

    # volume
    rvol: float = float("nan")
    rvol_state: str = "UNKNOWN"           # EXPANSION / NORMAL / DRY
    session_volume: float = float("nan")

    # delta proxy
    cum_delta: float = float("nan")
    delta_slope: float = float("nan")
    delta_divergence: str = "none"        # bullish / bearish / none

    levels: Levels = field(default_factory=Levels)
    sweeps: list[Sweep] = field(default_factory=list)

    # cost
    spread_pts: float = float("nan")
    slippage_pts: float = float("nan")
    round_trip_cost: float = float("nan")
    min_viable_target: float = float("nan")

    score: float = 0.0
    confidence: float = 0.0
    flags: list[str] = field(default_factory=list)

    @property
    def active_sweep(self) -> Sweep | None:
        confirmed = [s for s in self.sweeps if s.confirmed]
        return min(confirmed, key=lambda s: s.bars_ago) if confirmed else None


# ==========================================================================
# Basis
# ==========================================================================
def compute_basis(fut: pd.DataFrame, cash: pd.DataFrame) -> tuple[float, float, float, float, bool]:
    """
    Returns (basis, mean, std, sigma, available).

    Only RTH bars are used. Outside RTH the cash index is a stale 16:00 print,
    so a basis computed against it measures nothing and must not reach C7.
    """
    if fut is None or fut.empty or cash is None or cash.empty:
        return (np.nan,) * 4 + (False,)

    f = to_et(fut).dropna(subset=["Close"])
    c = to_et(cash).dropna(subset=["Close"])
    if f.empty or c.empty:
        return (np.nan,) * 4 + (False,)

    f = f[rth_mask(f.index)]
    c = c[rth_mask(c.index)]
    if f.empty or c.empty:
        return (np.nan,) * 4 + (False,)

    joined = pd.DataFrame({"fut": f["Close"]}).join(
        pd.DataFrame({"cash": c["Close"]}), how="inner"
    ).dropna()
    if len(joined) < config.BASIS_MIN_SAMPLES:
        return (np.nan,) * 4 + (False,)

    spread = joined["fut"] - joined["cash"]
    current = float(spread.iloc[-1])
    mean = float(spread.mean())
    std = float(spread.std(ddof=1))
    sigma = (current - mean) / std if std > 1e-9 else 0.0
    return current, mean, std, float(sigma), True


# ==========================================================================
# Relative volume
# ==========================================================================
def compute_rvol(bars: pd.DataFrame, baseline_days: int = config.RVOL_BASELINE_DAYS) -> tuple[float, pd.Series]:
    """
    Relative volume for the latest bar against the SAME time-of-day slot on
    prior sessions. Returns (rvol, per-bar rvol series for today).
    """
    empty = pd.Series(dtype=float)
    if bars is None or bars.empty or "Volume" not in bars.columns:
        return float("nan"), empty

    b = to_et(bars).dropna(subset=["Close"])
    if b.empty:
        return float("nan"), empty

    vol = pd.to_numeric(b["Volume"], errors="coerce").fillna(0.0)
    if vol.sum() <= 0:
        return float("nan"), empty

    sess = session_dates(b.index)
    slot = b.index.hour * 60 + b.index.minute

    frame = pd.DataFrame({"vol": vol.to_numpy(), "sess": sess.to_numpy(), "slot": slot},
                         index=b.index)
    sessions = pd.Index(pd.unique(frame["sess"]))
    if len(sessions) < 2:
        return float("nan"), empty

    today = sessions[-1]
    history = frame[frame["sess"] != today]
    if history.empty:
        return float("nan"), empty

    recent_sessions = pd.Index(pd.unique(history["sess"]))[-baseline_days:]
    history = history[history["sess"].isin(recent_sessions)]
    baseline = history.groupby("slot")["vol"].median()

    todays = frame[frame["sess"] == today]
    ratios = todays["vol"] / todays["slot"].map(baseline).replace(0.0, np.nan)
    ratios = ratios.replace([np.inf, -np.inf], np.nan)
    latest = float(ratios.dropna().iloc[-1]) if ratios.notna().any() else float("nan")
    return latest, ratios


# ==========================================================================
# Delta proxy
# ==========================================================================
def compute_delta(bars: pd.DataFrame) -> tuple[float, float, pd.Series]:
    """
    Bar-derived delta PROXY: where the close sits in each bar's range, mapped
    to [-1, +1] and weighted by that bar's volume, cumulated over the session.

    An inside bar with a mid-range close contributes ~0, which is correct
    behaviour — the bar genuinely carries no directional information at this
    resolution. It is NOT real delta and is labelled as a proxy everywhere.
    """
    empty = pd.Series(dtype=float)
    if bars is None or bars.empty or "Volume" not in bars.columns:
        return float("nan"), float("nan"), empty

    b = to_et(bars).dropna(subset=["Close", "High", "Low"])
    if b.empty:
        return float("nan"), float("nan"), empty

    sess = session_dates(b.index)
    today = sess.iloc[-1]
    b = b[(sess == today).to_numpy()]
    if b.empty:
        return float("nan"), float("nan"), empty

    rng = (b["High"] - b["Low"]).replace(0.0, np.nan)
    position = ((b["Close"] - b["Low"]) / rng).fillna(0.5)     # 0.5 = no information
    signed = 2.0 * position - 1.0
    vol = pd.to_numeric(b["Volume"], errors="coerce").fillna(0.0)
    delta = (signed * vol).cumsum()

    if delta.empty:
        return float("nan"), float("nan"), empty

    cum = float(delta.iloc[-1])
    tail = delta.tail(12)
    slope = float(np.polyfit(range(len(tail)), tail.to_numpy(), 1)[0]) if len(tail) >= 4 else 0.0
    return cum, slope, delta


def delta_divergence(bars: pd.DataFrame, delta: pd.Series) -> str:
    """Price making a new session extreme that cumulative delta does not confirm."""
    if bars is None or bars.empty or delta.empty or len(delta) < config.DELTA_DIVERGENCE_MIN_BARS:
        return "none"
    b = to_et(bars).reindex(delta.index).dropna(subset=["Close"])
    d = delta.reindex(b.index).dropna()
    if len(d) < config.DELTA_DIVERGENCE_MIN_BARS:
        return "none"

    half = len(d) // 2
    first_px, second_px = b["Close"].iloc[:half], b["Close"].iloc[half:]
    first_d, second_d = d.iloc[:half], d.iloc[half:]
    if first_px.empty or second_px.empty:
        return "none"

    if second_px.max() > first_px.max() and second_d.max() < first_d.max():
        return "bearish"
    if second_px.min() < first_px.min() and second_d.min() > first_d.min():
        return "bullish"
    return "none"


# ==========================================================================
# Levels
# ==========================================================================
def compute_levels(bars: pd.DataFrame, price: float) -> Levels:
    lv = Levels()
    if bars is None or bars.empty:
        return lv

    b = to_et(bars).dropna(subset=["Close", "High", "Low"])
    if b.empty:
        return lv

    sess = session_dates(b.index)
    sessions = pd.Index(pd.unique(sess))
    if len(sessions) == 0:
        return lv
    today = sessions[-1]

    in_today = (sess == today).to_numpy()
    today_bars = b[in_today]
    rth_today = rth_mask(today_bars.index)

    # Overnight = today's session bars BEFORE the RTH open.
    overnight = today_bars[~rth_today]
    if not overnight.empty:
        lv.overnight_high = float(overnight["High"].max())
        lv.overnight_low = float(overnight["Low"].min())

    # Prior session RTH.
    if len(sessions) >= 2:
        prev = sessions[-2]
        prev_bars = b[(sess == prev).to_numpy()]
        prev_rth = prev_bars[rth_mask(prev_bars.index)]
        target = prev_rth if not prev_rth.empty else prev_bars
        if not target.empty:
            lv.prior_high = float(target["High"].max())
            lv.prior_low = float(target["Low"].min())
            lv.prior_close = float(target["Close"].iloc[-1])

    # Initial balance: first 60 RTH minutes of today.
    rth_bars = today_bars[rth_today]
    if not rth_bars.empty:
        first_hour = rth_bars[rth_bars.index < rth_bars.index[0] + pd.Timedelta(minutes=60)]
        if not first_hour.empty:
            lv.ib_high = float(first_hour["High"].max())
            lv.ib_low = float(first_hour["Low"].min())

    if np.isfinite(lv.overnight_high) and np.isfinite(lv.overnight_low) and np.isfinite(price):
        span = lv.overnight_high - lv.overnight_low
        if span > 1e-9:
            lv.on_range_position = (price - lv.overnight_low) / span
        if price > lv.overnight_high:
            lv.state = "ABOVE_ON"
        elif price < lv.overnight_low:
            lv.state = "BELOW_ON"
        else:
            lv.state = "INSIDE_ON"
    return lv


# ==========================================================================
# Liquidity sweeps
# ==========================================================================
def detect_sweeps(bars: pd.DataFrame, levels: Levels, atr_val: float,
                  rvol_series: pd.Series | None = None) -> list[Sweep]:
    """
    A sweep is a level being taken out and REJECTED: price trades through it,
    then closes back inside by at least SWEEP_REJECTION_FRAC of the bar's range.
    A level merely being exceeded is a breakout, not a sweep, and the two imply
    opposite things — so the rejection test is the whole point.
    """
    out: list[Sweep] = []
    if bars is None or bars.empty or not np.isfinite(atr_val) or atr_val <= 0:
        return out

    b = to_et(bars).dropna(subset=["Close", "High", "Low"]).tail(config.SWEEP_LOOKBACK_BARS)
    if b.empty:
        return out

    candidates: list[tuple[str, str, float]] = []
    if np.isfinite(levels.overnight_high):
        candidates.append(("high", "Overnight high", levels.overnight_high))
    if np.isfinite(levels.overnight_low):
        candidates.append(("low", "Overnight low", levels.overnight_low))
    if np.isfinite(levels.prior_high):
        candidates.append(("high", "Prior day high", levels.prior_high))
    if np.isfinite(levels.prior_low):
        candidates.append(("low", "Prior day low", levels.prior_low))
    if np.isfinite(levels.ib_high):
        candidates.append(("high", "Initial balance high", levels.ib_high))
    if np.isfinite(levels.ib_low):
        candidates.append(("low", "Initial balance low", levels.ib_low))

    equal_high, equal_low = _equal_levels(b, atr_val)
    if equal_high is not None:
        candidates.append(("high", "Equal highs", equal_high))
    if equal_low is not None:
        candidates.append(("low", "Equal lows", equal_low))

    min_pen = atr_val * config.SWEEP_MIN_PENETRATION_ATR
    n = len(b)

    for side, name, level in candidates:
        for pos in range(n - 1, max(n - 13, -1), -1):     # last 12 bars only
            bar = b.iloc[pos]
            high, low, close = float(bar["High"]), float(bar["Low"]), float(bar["Close"])
            rng = high - low
            if rng <= 0:
                continue

            if side == "high":
                penetration = high - level
                if penetration < min_pen:
                    continue
                rejected = (high - close) >= rng * config.SWEEP_REJECTION_FRAC and close < level
            else:
                penetration = level - low
                if penetration < min_pen:
                    continue
                rejected = (close - low) >= rng * config.SWEEP_REJECTION_FRAC and close > level
            if not rejected:
                continue

            bar_rvol = float("nan")
            if rvol_series is not None and not rvol_series.empty:
                stamp = b.index[pos]
                if stamp in rvol_series.index:
                    bar_rvol = float(rvol_series.loc[stamp])

            out.append(Sweep(
                side=side, level_name=name, level=float(level),
                penetration=float(penetration), rvol=bar_rvol,
                confirmed=bool(np.isnan(bar_rvol) or bar_rvol >= config.SWEEP_RVOL_CONFIRM),
                bars_ago=n - 1 - pos,
            ))
            break     # one sweep per level, the most recent
    return out


def _equal_levels(bars: pd.DataFrame, atr_val: float) -> tuple[float | None, float | None]:
    """Clusters of swing highs / lows within tolerance — the liquidity pools."""
    tol = atr_val * config.SWEEP_EQUAL_TOL_ATR
    if len(bars) < 9 or tol <= 0:
        return None, None

    highs = bars["High"].to_numpy()
    lows = bars["Low"].to_numpy()
    piv_h, piv_l = [], []
    for i in range(2, len(bars) - 2):
        window_h = highs[i - 2:i + 3]
        window_l = lows[i - 2:i + 3]
        if highs[i] == window_h.max():
            piv_h.append(highs[i])
        if lows[i] == window_l.min():
            piv_l.append(lows[i])

    def cluster(values: list[float], take_max: bool) -> float | None:
        if len(values) < 2:
            return None
        arr = np.sort(np.array(values))[::-1] if take_max else np.sort(np.array(values))
        for i in range(len(arr) - 1):
            if abs(arr[i] - arr[i + 1]) <= tol:
                return float((arr[i] + arr[i + 1]) / 2.0)
        return None

    return cluster(piv_h, True), cluster(piv_l, False)


# ==========================================================================
# Main
# ==========================================================================
def compute_micro(fut_bars: pd.DataFrame,
                  cash_bars: pd.DataFrame | None = None,
                  spread_pts: float = config.DEFAULT_SPREAD_PTS,
                  slippage_pts: float = config.DEFAULT_SLIPPAGE_PTS,
                  source: str = "YM=F",
                  price_scale: float = 1.0) -> MicroReport:
    """Pure function. `fut_bars` must be a volume-bearing source (YM=F or DIA)."""
    rep = MicroReport(source=source)
    if fut_bars is None or fut_bars.empty:
        rep.note = "no futures/volume bars"
        return rep

    b = to_et(fut_bars).dropna(subset=["Close"])
    if len(b) < 20:
        rep.note = f"only {len(b)} bars — need 20"
        return rep

    rep.price = float(b["Close"].iloc[-1]) * price_scale

    a = atr(b["High"], b["Low"], b["Close"], 14)
    rep.atr_intraday = float(a.iloc[-1]) * price_scale if a.notna().any() else float("nan")

    # ---- cost budget ------------------------------------------------------
    rep.spread_pts = float(spread_pts)
    rep.slippage_pts = float(slippage_pts)
    rep.round_trip_cost = rep.spread_pts + rep.slippage_pts
    rep.min_viable_target = rep.round_trip_cost * config.C9_COST_MULTIPLE

    # ---- basis ------------------------------------------------------------
    if cash_bars is not None and not cash_bars.empty:
        (rep.basis, rep.basis_mean, rep.basis_std,
         rep.basis_sigma, rep.basis_available) = compute_basis(b, cash_bars)
        if not rep.basis_available:
            rep.flags.append(
                "Basis unavailable — outside RTH, or too few overlapping bars. "
                "C7 correctly stays silent rather than firing on a stale cash print."
            )

    # ---- volume -----------------------------------------------------------
    rep.rvol, rvol_series = compute_rvol(b)
    if np.isfinite(rep.rvol):
        if rep.rvol >= config.RVOL_SPIKE:
            rep.rvol_state = "EXPANSION"
        elif rep.rvol <= config.RVOL_DRY:
            rep.rvol_state = "DRY"
            rep.flags.append(
                f"Dry tape (RVOL {rep.rvol:.2f}) — breakout reads suppressed"
            )
        else:
            rep.rvol_state = "NORMAL"
    else:
        rep.flags.append("RVOL unavailable — need at least two sessions of bars")

    sess = session_dates(b.index)
    today_bars = b[(sess == sess.iloc[-1]).to_numpy()]
    if not today_bars.empty and "Volume" in today_bars:
        rep.session_volume = float(
            pd.to_numeric(today_bars["Volume"], errors="coerce").fillna(0).sum()
        )

    # ---- delta proxy --------------------------------------------------------
    rep.cum_delta, rep.delta_slope, delta_series = compute_delta(b)
    rep.delta_divergence = delta_divergence(b, delta_series)

    # ---- levels and sweeps ---------------------------------------------------
    raw_price = float(b["Close"].iloc[-1])
    rep.levels = compute_levels(b, raw_price)
    atr_raw = float(a.iloc[-1]) if a.notna().any() else float("nan")
    rep.sweeps = detect_sweeps(b, rep.levels, atr_raw, rvol_series)

    # ---- score ----------------------------------------------------------------
    parts: list[float] = []

    # A confirmed sweep is the strongest single read this layer produces:
    # the level was taken out AND rejected, which implies the opposite direction.
    active = rep.active_sweep
    if active is not None:
        weight = 1.0 if active.bars_ago <= 3 else 0.6
        parts.append((-1.0 if active.side == "high" else 1.0) * weight)
        rep.flags.append(
            f"{active.level_name} swept and rejected {active.bars_ago} bars ago "
            f"({active.penetration:.0f}pts through, RVOL "
            f"{active.rvol:.2f}) — {active.implication}"
            if np.isfinite(active.rvol) else
            f"{active.level_name} swept and rejected {active.bars_ago} bars ago "
            f"— {active.implication}"
        )

    # Delta proxy, deliberately weighted low: it is a proxy, not order flow.
    if np.isfinite(rep.cum_delta) and rep.session_volume and rep.session_volume > 0:
        normalised = float(np.clip(rep.cum_delta / rep.session_volume, -1.0, 1.0))
        parts.append(normalised * 0.5)

    # Overnight range state, but only when volume backs it.
    if rep.levels.state == "ABOVE_ON" and rep.rvol_state == "EXPANSION":
        parts.append(0.8)
    elif rep.levels.state == "BELOW_ON" and rep.rvol_state == "EXPANSION":
        parts.append(-0.8)
    elif rep.levels.state in ("ABOVE_ON", "BELOW_ON") and rep.rvol_state == "DRY":
        parts.append(0.0)
        rep.flags.append(
            f"Price {rep.levels.state.replace('_', ' ').lower()} on dry volume "
            f"— treated as a failed breakout, not a trend"
        )

    base = float(np.mean(parts)) if parts else 0.0

    if rep.delta_divergence == "bearish" and base > 0:
        base *= 0.5
        rep.flags.append("Delta proxy diverging bearish against a long read")
    elif rep.delta_divergence == "bullish" and base < 0:
        base *= 0.5
        rep.flags.append("Delta proxy diverging bullish against a short read")

    rep.score = float(np.clip(base, -1.0, 1.0) * config.LAYER_WEIGHTS["microstructure"])

    conf = 1.0
    if not rep.basis_available:
        conf *= 0.85
    if not np.isfinite(rep.rvol):
        conf *= 0.7
    if not np.isfinite(rep.levels.overnight_high):
        conf *= 0.8
    rep.confidence = float(np.clip(conf, 0.0, 1.0))
    rep.ok = True
    rep.note = (
        f"{source}, RVOL {rep.rvol:.2f} ({rep.rvol_state}), "
        f"{len(rep.sweeps)} sweep(s), basis "
        f"{'live' if rep.basis_available else 'n/a'}"
    )
    return rep


def get_micro(spread_pts: float = config.DEFAULT_SPREAD_PTS,
              slippage_pts: float = config.DEFAULT_SLIPPAGE_PTS) -> MicroReport:
    try:
        fut = dl.get_futures_intraday(interval="5m", period="1mo")
        source = fut.source
        bars = dl.ohlcv(fut)

        # Fall back to the generic volume source (DIA) if futures did not resolve.
        scale = 1.0
        if bars.empty:
            vol = dl.get_volume_source(interval="5m", period="1mo")
            bars = dl.ohlcv(vol)
            source = vol.source
            if not bars.empty and float(bars["Close"].iloc[-1]) < 2000:
                scale = config.DIA_TO_INDEX      # DIA trades near DJIA/100

        cash = dl.ohlcv(dl.get_index_intraday(interval="5m", period="5d"))
        return compute_micro(bars, cash, spread_pts, slippage_pts, source, scale)
    except Exception as exc:  # noqa: BLE001
        return MicroReport(note=f"microstructure failed: {exc}")
