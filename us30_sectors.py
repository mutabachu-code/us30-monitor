"""
us30_sectors.py — US30 Monitor

Sector relative strength, re-weighted to DOW sector weights rather than S&P
weights. Two things make this different from a broad-index sector panel:

  * The Dow contains NO utilities and NO real estate, so XLU and XLRE are
    deliberately absent. Including them would read noise from sectors the
    index does not hold.
  * Financials are 27.8% of the Dow versus roughly 13% of the S&P. XLF
    relative strength therefore carries about twice the information here, and
    is the input the C2 rates conflict depends on.

Sector weights are derived live from component prices, not hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
import data_layer as dl


@dataclass
class SectorReport:
    ok: bool = False
    note: str = ""

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    dow_weights: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    xlf_rs: float = float("nan")        # XLF return minus SPY return, %
    xlf_underperforming: bool = False
    risk_tone: str = "NEUTRAL"          # RISK_ON / NEUTRAL / RISK_OFF

    score: float = 0.0
    confidence: float = 0.0
    flags: list[str] = field(default_factory=list)


def dow_sector_weights(component_prices: pd.Series) -> pd.Series:
    """Price weight per sector, derived from live prices."""
    p = pd.to_numeric(component_prices, errors="coerce").dropna()
    if p.empty:
        return pd.Series(dtype=float)
    sectors = pd.Series({t: config.SECTOR_MAP.get(t, "Unknown") for t in p.index})
    return (p.groupby(sectors).sum() / p.sum() * 100.0).sort_values(ascending=False)


def compute_sectors(etf_closes: pd.DataFrame,
                    component_prices: pd.Series,
                    lookback: int = 5) -> SectorReport:
    """Pure function."""
    rep = SectorReport()
    if etf_closes is None or etf_closes.empty or len(etf_closes) < lookback + 1:
        rep.note = "insufficient sector ETF history"
        return rep

    rep.dow_weights = dow_sector_weights(component_prices)

    def ret(tk: str) -> float:
        if tk not in etf_closes.columns:
            return float("nan")
        s = pd.to_numeric(etf_closes[tk], errors="coerce").dropna()
        if len(s) < lookback + 1 or s.iloc[-1 - lookback] == 0:
            return float("nan")
        return float(s.iloc[-1] / s.iloc[-1 - lookback] - 1.0) * 100.0

    spy_ret = ret("SPY")
    rows = []
    for sector, etf in config.SECTOR_ETF.items():
        r = ret(etf)
        if not np.isfinite(r):
            continue
        rows.append({
            "sector": sector,
            "etf": etf,
            "return_pct": round(r, 2),
            "rs_vs_spy": round(r - spy_ret, 2) if np.isfinite(spy_ret) else np.nan,
            "dow_weight_pct": round(float(rep.dow_weights.get(sector, 0.0)), 2),
        })

    if not rows:
        rep.note = "no sector ETFs resolved"
        return rep

    rep.table = pd.DataFrame(rows).set_index("sector")
    rep.table["weighted_rs"] = (
        rep.table["rs_vs_spy"] * rep.table["dow_weight_pct"] / 100.0
    ).round(3)
    rep.table = rep.table.sort_values("dow_weight_pct", ascending=False)

    # ---- XLF: the input C2 needs ----------------------------------------
    if "Financials" in rep.table.index:
        rep.xlf_rs = float(rep.table.loc["Financials", "rs_vs_spy"])
        rep.xlf_underperforming = bool(np.isfinite(rep.xlf_rs) and rep.xlf_rs < -0.25)
        if rep.xlf_underperforming:
            rep.flags.append(
                f"XLF lagging SPY by {abs(rep.xlf_rs):.2f}% — the Dow's largest "
                f"bloc (27.8%) is not participating"
            )

    # ---- risk tone from Dow-weighted defensives vs cyclicals -------------
    cyclicals = ["Financials", "Industrials", "Cons Disc", "Technology", "Materials"]
    defensives = ["Staples", "Health Care"]
    cyc = rep.table.reindex(cyclicals)["weighted_rs"].dropna().sum()
    dfs = rep.table.reindex(defensives)["weighted_rs"].dropna().sum()
    spread = float(cyc - dfs)
    if spread > 0.15:
        rep.risk_tone = "RISK_ON"
    elif spread < -0.15:
        rep.risk_tone = "RISK_OFF"

    total_weighted = float(rep.table["weighted_rs"].dropna().sum())
    base = float(np.clip(total_weighted / 1.2, -1.0, 1.0))
    rep.score = base * config.LAYER_WEIGHTS["sectors"]

    rep.confidence = float(np.clip(len(rep.table) / len(config.SECTOR_ETF), 0.0, 1.0))
    rep.ok = True
    rep.note = f"{len(rep.table)}/{len(config.SECTOR_ETF)} sectors, tone {rep.risk_tone}"
    return rep


def get_sectors() -> SectorReport:
    try:
        etfs = dl.closes(dl.get_sector_daily())
        prices = dl.closes(dl.get_components_daily(), config.COMPONENTS)
        last = prices.dropna(how="all").iloc[-1] if not prices.empty else pd.Series(dtype=float)
        return compute_sectors(etfs, last)
    except Exception as exc:  # noqa: BLE001
        return SectorReport(note=f"sectors failed: {exc}")
