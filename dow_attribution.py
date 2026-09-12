"""
dow_attribution.py — US30 Monitor CORE ENGINE

The Dow is price-weighted, so every component's effect on the index is EXACT,
not estimated:

    contribution_points(i) = (price_i - prev_close_i) / divisor

and the 30 contributions sum to the index change to within rounding. That
reconciliation is this module's correctness test, and it is the single biggest
edge available on this instrument — there is no cap-weighted equivalent.

The divisor is derived live (sum of prices / index level), never hard-coded.
Drift in the derived divisor IS the constituent-change / stock-split detector.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
import data_layer as dl


# ==========================================================================
# Results
# ==========================================================================
@dataclass
class DivisorState:
    value: float = config.DIVISOR_REFERENCE
    derived: bool = False
    drift: float = 0.0
    alert: bool = False
    message: str = ""

    @property
    def points_per_dollar(self) -> float:
        return 1.0 / self.value if self.value else 0.0


@dataclass
class AttributionReport:
    ok: bool = False
    note: str = ""

    divisor: DivisorState = field(default_factory=DivisorState)
    index_level: float = float("nan")
    index_change_pts: float = 0.0
    reconciled_pts: float = 0.0
    reconciliation_error: float = 0.0

    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    sector_points: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    advancers: int = 0
    decliners: int = 0
    unchanged: int = 0
    pw_advance_pts: float = 0.0     # sum of positive contributions
    pw_decline_pts: float = 0.0     # sum of |negative contributions|

    efficiency: float = 0.0         # net / gross, in [-1, 1]
    participation: float = 0.0      # (adv - dec) / n, in [-1, 1]
    top2_share: float = 0.0         # share of GROSS move from 2 biggest movers
    top5_share: float = 0.0

    divergence: bool = False
    narrow_move: bool = False
    flags: list[str] = field(default_factory=list)

    score: float = 0.0              # -25 .. +25
    confidence: float = 0.0         # 0 .. 1

    @property
    def top_contributors(self) -> pd.DataFrame:
        if self.table.empty:
            return self.table
        return self.table.reindex(
            self.table["points"].abs().sort_values(ascending=False).index
        )


# ==========================================================================
# Divisor
# ==========================================================================
def derive_divisor(prices: pd.Series, index_level: float) -> DivisorState:
    """
    divisor = sum(component prices) / index level

    Requires a complete set of prices; a missing name silently shrinks the
    numerator and produces a divisor that is wrong in a direction that looks
    plausible, which is worse than no divisor at all.
    """
    st = DivisorState()
    clean = pd.to_numeric(prices, errors="coerce").dropna()

    if len(clean) < 30 or not np.isfinite(index_level) or index_level <= 0:
        st.message = (
            f"using reference divisor ({len(clean)}/30 prices, "
            f"index={index_level:.2f})" if np.isfinite(index_level)
            else f"using reference divisor ({len(clean)}/30 prices, no index level)"
        )
        return st

    derived = float(clean.sum()) / float(index_level)
    st.value = derived
    st.derived = True
    st.drift = abs(derived / config.DIVISOR_REFERENCE - 1.0)
    st.alert = st.drift > config.DIVISOR_DRIFT_ALERT
    st.message = (
        f"derived {derived:.14f} — drift {st.drift * 100:.3f}% vs reference"
        + (" — CHECK FOR SPLIT OR CONSTITUENT CHANGE" if st.alert else "")
    )
    return st


# ==========================================================================
# Scoring helpers
# ==========================================================================
def _concentration_quality(top2_share: float) -> float:
    """
    Equal-magnitude contributions would put top-2 share near 0.07; real price
    weighting puts a healthy broad day around 0.25-0.35. Quality falls away
    above that and bottoms out at 0.15 for a two-stock move.
    """
    if not np.isfinite(top2_share):
        return 1.0
    q = 1.0 - (top2_share - 0.30) / 0.40
    return float(np.clip(q, 0.15, 1.0))


def _tier_confidence(rep: "AttributionReport", n_valid: int) -> float:
    conf = n_valid / 30.0
    if not rep.divisor.derived:
        conf *= 0.7
    if rep.divisor.alert:
        conf *= 0.6
    if abs(rep.reconciliation_error) > 5.0:
        conf *= 0.5
    return float(np.clip(conf, 0.0, 1.0))


# ==========================================================================
# Main engine
# ==========================================================================
def compute_attribution(
    last: pd.Series,
    prev: pd.Series,
    index_level: float,
    index_prev: float,
    ex_div_today: set[str] | None = None,
) -> AttributionReport:
    """
    Pure function — no I/O. `last`/`prev` are price Series indexed by ticker.
    Separated from the fetch so the test suite can drive it with known answers.
    """
    rep = AttributionReport()
    ex_div_today = ex_div_today or set()

    last = pd.to_numeric(last, errors="coerce")
    prev = pd.to_numeric(prev, errors="coerce")
    common = [t for t in last.index if t in prev.index]
    last, prev = last.reindex(common), prev.reindex(common)
    valid = last.notna() & prev.notna()
    last, prev = last[valid], prev[valid]
    n_valid = int(len(last))

    if n_valid == 0:
        rep.note = "no valid component prices"
        return rep

    rep.divisor = derive_divisor(last, index_level)
    div = rep.divisor.value
    rep.index_level = float(index_level)
    rep.index_change_pts = (
        float(index_level - index_prev)
        if np.isfinite(index_level) and np.isfinite(index_prev) else float("nan")
    )

    dollar = last - prev
    points = dollar / div

    # Ex-dividend days produce a mechanical price drop that is not information.
    for tk in ex_div_today:
        if tk in points.index:
            points[tk] = 0.0
            dollar[tk] = 0.0

    weight = last / last.sum() * 100.0

    rep.table = pd.DataFrame({
        "price": last.round(2),
        "prev": prev.round(2),
        "dollar": dollar.round(2),
        "pct": (dollar / prev * 100.0).round(3),
        "points": points.round(2),
        "weight_pct": weight.round(3),
        "pts_per_1pct": (last * 0.01 / div).round(2),
        "sector": [config.SECTOR_MAP.get(t, "Unknown") for t in last.index],
    })

    rep.reconciled_pts = float(points.sum())
    if np.isfinite(rep.index_change_pts):
        rep.reconciliation_error = rep.reconciled_pts - rep.index_change_pts

    rep.sector_points = (
        rep.table.groupby("sector")["points"].sum().sort_values(ascending=False)
    )

    # ---- breadth -------------------------------------------------------
    adv_mask, dec_mask = points > 0, points < 0
    rep.advancers = int(adv_mask.sum())
    rep.decliners = int(dec_mask.sum())
    rep.unchanged = n_valid - rep.advancers - rep.decliners
    rep.pw_advance_pts = float(points[adv_mask].sum())
    rep.pw_decline_pts = float(-points[dec_mask].sum())

    gross = rep.pw_advance_pts + rep.pw_decline_pts
    net = rep.pw_advance_pts - rep.pw_decline_pts
    rep.efficiency = float(net / gross) if gross > 1e-9 else 0.0
    rep.participation = float((rep.advancers - rep.decliners) / n_valid)

    # ---- concentration -------------------------------------------------
    absolute = points.abs().sort_values(ascending=False)
    if gross > 1e-9:
        rep.top2_share = float(absolute.iloc[:2].sum() / gross)
        rep.top5_share = float(absolute.iloc[:5].sum() / gross)

    # ---- flags ---------------------------------------------------------
    if abs(net) > 1e-9 and abs(rep.participation) > 0.05 \
            and np.sign(net) != np.sign(rep.participation):
        rep.divergence = True
        rep.flags.append(
            f"Breadth divergence: index {net:+.0f}pts but "
            f"{rep.advancers}A/{rep.decliners}D"
        )

    rep.narrow_move = (
        abs(net) >= config.C1_MIN_MOVE_PTS
        and rep.top2_share >= config.C1_TOP2_SHARE
        and abs(net) * rep.efficiency < config.C1_MIN_PW_AD * 10
    )
    if rep.top2_share >= config.C1_TOP2_SHARE:
        top2 = ", ".join(absolute.index[:2])
        rep.flags.append(
            f"Narrow tape: {top2} are {rep.top2_share * 100:.0f}% of the gross move"
        )
    if rep.divisor.alert:
        rep.flags.append("Divisor drift — verify constituents and splits")
    if abs(rep.reconciliation_error) > 5.0 and np.isfinite(rep.reconciliation_error):
        rep.flags.append(
            f"Reconciliation off by {rep.reconciliation_error:+.1f}pts "
            f"— stale or missing component prices"
        )

    # ---- score ---------------------------------------------------------
    blend = 0.55 * rep.efficiency + 0.45 * rep.participation
    quality = _concentration_quality(rep.top2_share)
    if rep.divergence:
        quality *= 0.35
    rep.score = float(
        np.clip(blend, -1.0, 1.0) * quality * config.LAYER_WEIGHTS["attribution"]
    )
    rep.confidence = _tier_confidence(rep, n_valid)
    rep.ok = True
    rep.note = f"{n_valid}/30 components, divisor {'derived' if rep.divisor.derived else 'reference'}"
    return rep


# ==========================================================================
# Fetch + compute
# ==========================================================================
def get_attribution(ex_div_today: set[str] | None = None) -> AttributionReport:
    """Live attribution. Never raises — an unhappy path returns ok=False."""
    try:
        comp = dl.get_components_daily()
        idx = dl.get_index_daily(period="1mo")

        prices = dl.closes(comp, config.COMPONENTS)
        if prices.empty:
            return AttributionReport(note="component prices unavailable")

        last, prev = dl.last_two_closes(prices)
        if last.empty:
            return AttributionReport(note="need at least two component sessions")

        idx_close = dl.closes(idx)
        if idx_close.empty or len(idx_close.dropna()) < 2:
            level = prev_level = float("nan")
        else:
            series = idx_close.iloc[:, 0].dropna()
            level, prev_level = float(series.iloc[-1]), float(series.iloc[-2])

        rep = compute_attribution(last, prev, level, prev_level, ex_div_today)
        if comp.missing:
            rep.flags.append(f"Missing prices: {', '.join(comp.missing)}")
        return rep
    except Exception as exc:  # noqa: BLE001
        return AttributionReport(note=f"attribution failed: {exc}")


def rank_by_weight(rep: AttributionReport, n: int = 8) -> list[str]:
    """
    Live top-N by PRICE weight — never hard-code this list. A split in GS would
    drop it from 1st to roughly 20th overnight.
    """
    if rep.table.empty:
        return config.COMPONENTS[:n]
    return list(rep.table["weight_pct"].sort_values(ascending=False).index[:n])
