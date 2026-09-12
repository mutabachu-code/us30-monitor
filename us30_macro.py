"""
us30_macro.py — US30 Monitor

Macro layer, weighted for the Dow specifically.

The Dow is 27.8% Financials. That makes rates genuinely TWO-SIDED here in a way
they are not for NAS100: rising yields lift bank net-interest-margin
expectations (GS, JPM, AXP, TRV, V) while pressuring the 18.5% tech bloc. This
module therefore does not apply a blanket "yields up = bearish" rule the way a
Nasdaq model can. It reports the yield delta and leaves the resolution to the
conflict resolver, which combines it with XLF relative strength (C2).

Yahoo quotes ^TNX / ^FVX / ^TYX / ^IRX as yield x 10. The /10 happens here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
import data_layer as dl

_T = config.MACRO_TICKERS


@dataclass
class MacroReport:
    ok: bool = False
    note: str = ""

    us10y: float = float("nan")
    us10y_change_bp: float = 0.0
    us5y: float = float("nan")
    curve_5s10s: float = float("nan")
    curve_change_bp: float = 0.0

    dxy: float = float("nan")
    dxy_change_pct: float = 0.0

    wti: float = float("nan")
    wti_change_pct: float = 0.0

    vix: float = float("nan")
    vxd: float = float("nan")
    vxd_vix_ratio: float = float("nan")

    regime_label: str = "NEUTRAL"
    score: float = 0.0
    confidence: float = 0.0
    drivers: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def _last_and_change(frame: pd.DataFrame, ticker: str) -> tuple[float, float]:
    """Latest value and absolute change vs prior session."""
    if ticker not in frame.columns:
        return float("nan"), 0.0
    s = pd.to_numeric(frame[ticker], errors="coerce").dropna()
    if s.empty:
        return float("nan"), 0.0
    if len(s) < 2:
        return float(s.iloc[-1]), 0.0
    return float(s.iloc[-1]), float(s.iloc[-1] - s.iloc[-2])


def compute_macro(closes: pd.DataFrame) -> MacroReport:
    """Pure function over a close-price frame keyed by macro ticker."""
    rep = MacroReport()
    if closes is None or closes.empty:
        rep.note = "no macro data"
        return rep

    # ---- rates (Yahoo yields are x10) -----------------------------------
    t10, d10 = _last_and_change(closes, _T["us10y"])
    t5, d5 = _last_and_change(closes, _T["us5y"])
    rep.us10y = t10 / 10.0 if np.isfinite(t10) else float("nan")
    rep.us5y = t5 / 10.0 if np.isfinite(t5) else float("nan")
    rep.us10y_change_bp = d10 * 10.0        # x10 quote, then to bp
    if np.isfinite(rep.us10y) and np.isfinite(rep.us5y):
        rep.curve_5s10s = rep.us10y - rep.us5y
        rep.curve_change_bp = (d10 - d5) * 10.0

    # ---- dollar ----------------------------------------------------------
    dxy_tk = _T["dxy"] if _T["dxy"] in closes.columns else _T["dxy_fallback"]
    dxy, ddxy = _last_and_change(closes, dxy_tk)
    rep.dxy = dxy
    if np.isfinite(dxy) and dxy != 0:
        rep.dxy_change_pct = ddxy / (dxy - ddxy) * 100.0 if (dxy - ddxy) else 0.0
    if dxy_tk == _T["dxy_fallback"]:
        rep.flags.append("DXY unavailable — using UUP as proxy")

    # ---- oil -------------------------------------------------------------
    wti, dwti = _last_and_change(closes, _T["wti"])
    rep.wti = wti
    if np.isfinite(wti) and (wti - dwti):
        rep.wti_change_pct = dwti / (wti - dwti) * 100.0

    # ---- volatility ------------------------------------------------------
    rep.vix, _ = _last_and_change(closes, _T["vix"])
    rep.vxd, _ = _last_and_change(closes, _T["vxd"])
    if np.isfinite(rep.vxd) and np.isfinite(rep.vix) and rep.vix > 0:
        rep.vxd_vix_ratio = rep.vxd / rep.vix
    elif not np.isfinite(rep.vxd):
        rep.flags.append("^VXD unavailable — Dow-specific vol read disabled")

    # ---- scoring ---------------------------------------------------------
    parts: list[float] = []

    # Curve steepening is the cleaner bank signal than yield level.
    if np.isfinite(rep.curve_change_bp) and abs(rep.curve_change_bp) > 0.5:
        contrib = float(np.clip(rep.curve_change_bp / 6.0, -1.0, 1.0)) * 0.8
        parts.append(contrib)
        rep.drivers.append(
            f"Curve {'steepening' if rep.curve_change_bp > 0 else 'flattening'} "
            f"{rep.curve_change_bp:+.1f}bp — "
            f"{'supportive of' if rep.curve_change_bp > 0 else 'a drag on'} the 27.8% financials bloc"
        )

    # Large yield moves are destabilising regardless of direction.
    if abs(rep.us10y_change_bp) > config.C2_YIELD_BP:
        parts.append(-0.4 * np.sign(abs(rep.us10y_change_bp)))
        rep.drivers.append(
            f"10y {rep.us10y_change_bp:+.1f}bp to {rep.us10y:.2f}% — "
            f"large enough to move both blocs, direction unresolved without XLF"
        )

    # Strong dollar is a direct Dow headwind: CAT, MMM, HON, PG, KO ~17%.
    if abs(rep.dxy_change_pct) > 0.15:
        parts.append(float(np.clip(-rep.dxy_change_pct / 0.8, -1.0, 1.0)) * 0.7)
        rep.drivers.append(
            f"DXY {rep.dxy_change_pct:+.2f}% — "
            f"{'headwind' if rep.dxy_change_pct > 0 else 'tailwind'} for ~17% foreign-revenue names"
        )

    # Oil: only CVX (2.4%) benefits directly; above ~$90 the input-cost and
    # consumer drag on industrials and discretionary outweighs that by far.
    # The LEVEL matters on its own, not only the daily change — a sustained $95
    # is a standing drag even on a flat day.
    if np.isfinite(rep.wti):
        if rep.wti > 90:
            severity = -0.35 - min((rep.wti - 90) / 20.0, 0.35)
            if rep.wti_change_pct > 1.0:
                severity -= 0.25
            parts.append(max(severity, -1.0))
            rep.drivers.append(
                f"WTI ${rep.wti:.1f} ({rep.wti_change_pct:+.1f}%) — above $90 the "
                f"input-cost and consumer drag outweighs CVX's 2.4% benefit"
            )
        elif abs(rep.wti_change_pct) > 2.5:
            parts.append(float(np.clip(-rep.wti_change_pct / 5.0, -1.0, 1.0)) * 0.3)
            rep.drivers.append(
                f"WTI {rep.wti_change_pct:+.1f}% to ${rep.wti:.1f} — sharp move, "
                f"sentiment effect on industrials"
            )

    # Dow-specific stress relative to the broad market.
    if np.isfinite(rep.vxd_vix_ratio):
        if rep.vxd_vix_ratio > 1.05:
            parts.append(-0.5)
            rep.drivers.append(
                f"VXD/VIX {rep.vxd_vix_ratio:.2f} — Dow-specific stress above market"
            )
        elif rep.vxd_vix_ratio < 0.88:
            parts.append(0.35)
            rep.drivers.append(
                f"VXD/VIX {rep.vxd_vix_ratio:.2f} — Dow calmer than the market"
            )
    if np.isfinite(rep.vix) and rep.vix > 28:
        parts.append(-0.5)
        rep.drivers.append(f"VIX {rep.vix:.1f} — elevated, size down")

    base = float(np.mean(parts)) if parts else 0.0
    rep.score = float(np.clip(base, -1.0, 1.0) * config.LAYER_WEIGHTS["macro"])

    if rep.score > 6:
        rep.regime_label = "SUPPORTIVE"
    elif rep.score < -6:
        rep.regime_label = "HOSTILE"
    elif abs(rep.score) > 2:
        rep.regime_label = "TILTED"

    have = sum(1 for v in (rep.us10y, rep.dxy, rep.wti, rep.vix) if np.isfinite(v))
    rep.confidence = float(np.clip(have / 4.0, 0.0, 1.0))
    rep.ok = True
    rep.note = f"{have}/4 core macro inputs, {len(parts)} active drivers"
    return rep


def get_macro() -> MacroReport:
    try:
        fetch = dl.get_macro_daily()
        frame = dl.closes(fetch)
        rep = compute_macro(frame)
        if fetch.missing:
            rep.flags.append(f"Missing macro tickers: {', '.join(fetch.missing)}")
        return rep
    except Exception as exc:  # noqa: BLE001
        return MacroReport(note=f"macro failed: {exc}")
