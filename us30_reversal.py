"""
us30_reversal.py — Reversal readiness. OBSERVATION ONLY.

This module scores nothing, emits no trades, and does not touch the master
signal. It exists to test one hypothesis before any code is allowed to act on
it:

    A continuation signal asks "which way is price going?" — a confluence
    question, answered by summing evidence, which is what L1-L7 do.

    A reversal asks "is this move dying, HERE, at THIS level?" — a SEQUENTIAL
    question. Stretch, then a level, then a sweep, then rejection, then the
    internals turn. Scrambled, those same ingredients mean nothing. A weighted
    sum cannot express "in this order", which is why bolting reversal inputs
    onto the seven layers would dilute them into noise.

So this is a state machine, and for now it only watches.

WHY THE MASTER SIGNAL MISSES THESE
----------------------------------
Not a bug — design. C3 blocks directional entries in CHOP, which is what a
decelerating selloff often reads as right at the turn. C12 blocks
mean-reversion setups in short gamma, which is exactly the environment a flush
into support creates. And confluence peaks in the MIDDLE of a move: at the
actual low most layers are still bearish, so by the time enough of them flip to
clear the ±25 gate, the reversal is already 100+ points old. All three are
correct for continuation trades. They simply make reversals unreachable by
tuning.

STATE IS DERIVED, NOT STORED
----------------------------
Every state is recomputed from the current bars on each run. Streamlit reruns
constantly, and a stored state machine would drift out of sync with the data it
claims to describe. The only thing persisted is the observation LOG, so that
"watch it for a week" produces evidence instead of impressions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config

# --------------------------------------------------------------------------
# Stale-config tolerance
# --------------------------------------------------------------------------
_CFG_DEFAULTS: dict[str, object] = {
    "REVERSAL_STRETCH_ATR": 1.5,
    "REVERSAL_RANGE_ATR": 1.2,
    "REVERSAL_LEVEL_PROXIMITY_ATR": 0.30,
    "REVERSAL_EXPECTED_MOVE_USED": 0.70,
    "REVERSAL_SWEEP_MAX_BARS": 12,
    "REVERSAL_MIN_EVIDENCE": 3,
    "REVERSAL_TREND_MIN_EVIDENCE": 4,
    "REVERSAL_BREADTH_GAP": 0.25,
    "REVERSAL_CONCENTRATION": 0.45,
    "REVERSAL_LOG_PATH": "us30_reversal_log.csv",
    "REVERSAL_FOLLOWUP_MINUTES": 30,
    "REVERSAL_DEDUP_MINUTES": 30,
}


def _cfg(name: str):
    return getattr(config, name, _CFG_DEFAULTS[name])


def config_health() -> list[str]:
    return [n for n in _CFG_DEFAULTS if not hasattr(config, n)]


DORMANT, ARMED, TRIGGERED, CONFIRMED = "DORMANT", "ARMED", "TRIGGERED", "CONFIRMED"


# ==========================================================================
# Levels
# ==========================================================================
@dataclass
class Level:
    name: str
    price: float
    kind: str            # support / resistance
    source: str          # pivots / session / options / vwap
    distance_pts: float = float("nan")
    distance_atr: float = float("nan")


def collect_levels(technicals=None, micro=None, gamma_levels=None) -> list[Level]:
    """
    Every level the existing engines already know about. The ARM state requires
    price to be AT one of these — no level, no arm. That is the guard against
    catching a falling knife in open air, and it is the single most important
    rule in this module.
    """
    out: list[Level] = []

    cpr = getattr(technicals, "cpr", None)
    if cpr is not None:
        for attr, label, kind in [
            ("s1", "Pivot S1", "support"), ("s2", "Pivot S2", "support"),
            ("s3", "Pivot S3", "support"), ("r1", "Pivot R1", "resistance"),
            ("r2", "Pivot R2", "resistance"), ("r3", "Pivot R3", "resistance"),
            ("bc", "CPR bottom", "support"), ("tc", "CPR top", "resistance"),
        ]:
            val = getattr(cpr, attr, float("nan"))
            if np.isfinite(val):
                out.append(Level(label, float(val), kind, "pivots"))

    levels = getattr(micro, "levels", None)
    if levels is not None:
        for attr, label, kind in [
            ("overnight_low", "Overnight low", "support"),
            ("overnight_high", "Overnight high", "resistance"),
            ("prior_low", "Prior day low", "support"),
            ("prior_high", "Prior day high", "resistance"),
            ("ib_low", "Initial balance low", "support"),
            ("ib_high", "Initial balance high", "resistance"),
        ]:
            val = getattr(levels, attr, float("nan"))
            if np.isfinite(val):
                out.append(Level(label, float(val), kind, "session"))

    for lvl in (gamma_levels or []):
        price = lvl.get("index_level", float("nan"))
        if np.isfinite(price):
            out.append(Level(
                f"Gamma wall ({lvl.get('tickers', '')})", float(price),
                lvl.get("kind", "support"), "options"))
    return out


def nearest_level(levels: list[Level], price: float, atr: float,
                  kind: str | None = None) -> Level | None:
    """Closest level of the requested kind, with distance annotated."""
    if not levels or not np.isfinite(price):
        return None
    best, best_gap = None, float("inf")
    for lvl in levels:
        if kind is not None and lvl.kind != kind:
            continue
        gap = abs(price - lvl.price)
        if gap < best_gap:
            best, best_gap = lvl, gap
    if best is None:
        return None
    best.distance_pts = price - best.price
    best.distance_atr = (best.distance_pts / atr) if np.isfinite(atr) and atr > 0 else float("nan")
    return best


# ==========================================================================
# Report
# ==========================================================================
@dataclass
class Evidence:
    key: str
    label: str
    present: bool
    detail: str = ""


@dataclass
class ReversalReport:
    ok: bool = False
    note: str = ""

    state: str = DORMANT
    direction: str = "NONE"          # BULLISH / BEARISH reversal expected
    price: float = float("nan")
    atr: float = float("nan")

    stretch_atr: float = float("nan")
    overnight_range_atr: float = float("nan")   # context only, never arms
    expected_move_used: float = float("nan")

    level: Level | None = None
    sweep_level: str = ""
    sweep_bars_ago: int = -1

    evidence: list[Evidence] = field(default_factory=list)
    evidence_count: int = 0
    evidence_required: int = 3

    regime: str = ""
    suppressed_by: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def evidence_frame(self) -> pd.DataFrame:
        if not self.evidence:
            return pd.DataFrame()
        return pd.DataFrame([{
            "evidence": e.label, "present": "yes" if e.present else "no",
            "detail": e.detail,
        } for e in self.evidence])

    @property
    def fingerprint(self) -> str:
        lvl = self.level.name if self.level else "none"
        return f"{self.direction}|{lvl}|{self.state}"


# ==========================================================================
# Assessment
# ==========================================================================
def assess(technicals=None, micro=None, attribution=None, options=None,
           regime=None, gamma_levels=None) -> ReversalReport:
    """Pure. Derives the whole state from current reports — nothing stored."""
    rep = ReversalReport()

    price = float(getattr(technicals, "price", float("nan")))
    atr = float(getattr(technicals, "atr14", float("nan")))
    rep.price, rep.atr = price, atr
    rep.regime = str(getattr(regime, "regime", "") or "")

    if not np.isfinite(price) or not np.isfinite(atr) or atr <= 0:
        rep.note = "need price and ATR"
        return rep

    # ---- 1. is the move extended? -----------------------------------------
    vwap_dist = float(getattr(technicals, "vwap_distance_pts", float("nan")))
    rep.stretch_atr = vwap_dist / atr if np.isfinite(vwap_dist) else float("nan")

    # Overnight range is CONTEXT ONLY, never an extension trigger. It measures
    # what happened before the cash session, which says nothing about how far
    # the current move has travelled — using it to arm would fire on a quiet
    # day that simply followed a busy night.
    lv = getattr(micro, "levels", None)
    on_high = getattr(lv, "overnight_high", float("nan")) if lv else float("nan")
    on_low = getattr(lv, "overnight_low", float("nan")) if lv else float("nan")
    if np.isfinite(on_high) and np.isfinite(on_low):
        rep.overnight_range_atr = (on_high - on_low) / atr

    exp_move = float(getattr(options, "expected_move_pts", float("nan")))
    moved = abs(float(getattr(attribution, "index_change_pts", float("nan"))))
    if np.isfinite(exp_move) and exp_move > 0 and np.isfinite(moved):
        rep.expected_move_used = moved / exp_move

    # Two measures of extension, both of which describe THIS move: distance
    # from VWAP in ATR, and how much of the options-implied daily move is spent.
    stretched = (
        (np.isfinite(rep.stretch_atr)
         and abs(rep.stretch_atr) >= float(_cfg("REVERSAL_STRETCH_ATR")))
        or (np.isfinite(rep.expected_move_used)
            and rep.expected_move_used >= float(_cfg("REVERSAL_EXPECTED_MOVE_USED")))
    )

    # Direction of the reversal is OPPOSITE the extension.
    if np.isfinite(rep.stretch_atr) and abs(rep.stretch_atr) > 0.2:
        rep.direction = "BULLISH" if rep.stretch_atr < 0 else "BEARISH"
    elif np.isfinite(moved) and moved > 0:
        chg = float(getattr(attribution, "index_change_pts", 0.0))
        rep.direction = "BULLISH" if chg < 0 else "BEARISH"

    if not stretched or rep.direction == "NONE":
        rep.ok = True
        rep.note = "no extension to reverse"
        return rep

    # ---- 2. is price AT a level? -------------------------------------------
    want = "support" if rep.direction == "BULLISH" else "resistance"
    levels = collect_levels(technicals, micro, gamma_levels)
    rep.level = nearest_level(levels, price, atr, want)

    near = (rep.level is not None and np.isfinite(rep.level.distance_atr)
            and abs(rep.level.distance_atr) <= float(_cfg("REVERSAL_LEVEL_PROXIMITY_ATR")))
    if not near:
        rep.state = DORMANT
        rep.ok = True
        rep.note = ("extended, but not at a level — no arm. This is the "
                    "falling-knife guard doing its job.")
        return rep

    rep.state = ARMED

    # ---- 3. was the level swept AND rejected? -------------------------------
    want_side = "low" if rep.direction == "BULLISH" else "high"
    max_bars = int(_cfg("REVERSAL_SWEEP_MAX_BARS"))
    for sweep in getattr(micro, "sweeps", []) or []:
        if getattr(sweep, "side", "") != want_side:
            continue
        if int(getattr(sweep, "bars_ago", 999)) > max_bars:
            continue
        if not getattr(sweep, "confirmed", False):
            rep.flags.append(
                f"{getattr(sweep, 'level_name', 'a level')} was swept but on thin "
                f"volume — not counted as a trigger")
            continue
        rep.state = TRIGGERED
        rep.sweep_level = str(getattr(sweep, "level_name", ""))
        rep.sweep_bars_ago = int(getattr(sweep, "bars_ago", -1))
        break

    # ---- 4. have the internals turned? ---------------------------------------
    bullish = rep.direction == "BULLISH"
    decay = float(getattr(technicals, "rsi_decay", 0.0))
    div = str(getattr(technicals, "divergence", "none"))
    delta_div = str(getattr(micro, "delta_divergence", "none"))
    participation = float(getattr(attribution, "participation", float("nan")))
    efficiency = float(getattr(attribution, "efficiency", float("nan")))
    top2 = float(getattr(attribution, "top2_share", float("nan")))

    decay_ok = decay > 0.3 if bullish else decay < -0.3
    rep.evidence.append(Evidence(
        "rsi_decay", "RSI momentum leaving the move", bool(decay_ok),
        f"decay {decay:+.2f}/bar"))

    div_ok = div == ("bullish" if bullish else "bearish")
    rep.evidence.append(Evidence(
        "divergence", "Price/RSI divergence", bool(div_ok), f"divergence: {div}"))

    delta_ok = delta_div == ("bullish" if bullish else "bearish")
    rep.evidence.append(Evidence(
        "delta", "Delta proxy diverging", bool(delta_ok),
        f"delta divergence: {delta_div} (proxy, not order flow)"))

    # THE Dow-specific one. On a price-weighted index attribution is exact, so
    # "breadth is better than the point move implies" is a measurable
    # exhaustion statement rather than an impression.
    gap = float("nan")
    breadth_ok = False
    if np.isfinite(participation) and np.isfinite(efficiency):
        gap = participation - efficiency
        breadth_ok = gap >= float(_cfg("REVERSAL_BREADTH_GAP")) if bullish \
            else gap <= -float(_cfg("REVERSAL_BREADTH_GAP"))
    rep.evidence.append(Evidence(
        "breadth", "Breadth turning ahead of price", bool(breadth_ok),
        f"participation {participation:+.2f} vs efficiency {efficiency:+.2f} "
        f"(gap {gap:+.2f})" if np.isfinite(gap) else "unavailable"))

    conc_ok = np.isfinite(top2) and top2 >= float(_cfg("REVERSAL_CONCENTRATION"))
    rep.evidence.append(Evidence(
        "concentration", "Move narrowing to few names", bool(conc_ok),
        f"top-2 share {top2 * 100:.0f}% of gross" if np.isfinite(top2) else "unavailable"))

    rep.evidence_count = sum(1 for e in rep.evidence if e.present)

    # In a genuine trend day every level fails. Demand more there.
    rep.evidence_required = int(_cfg("REVERSAL_TREND_MIN_EVIDENCE")) \
        if rep.regime == "TREND" else int(_cfg("REVERSAL_MIN_EVIDENCE"))
    if rep.regime == "TREND":
        rep.flags.append(
            "TREND regime — evidence bar raised, because in a real trend day "
            "every support level fails and this is where a reversal engine bleeds")

    if rep.state == TRIGGERED and rep.evidence_count >= rep.evidence_required:
        rep.state = CONFIRMED

    # ---- what the master signal would have done with this ------------------
    if rep.state in (TRIGGERED, CONFIRMED):
        if rep.regime == "CHOP":
            rep.suppressed_by.append("C3 (chop blocks directional entries)")
        if str(getattr(options, "gamma_regime", "")) == "SHORT_GAMMA":
            rep.suppressed_by.append("C12 (short gamma blocks fades)")
        if str(getattr(technicals, "mean_reversion", "none")) == "none":
            rep.suppressed_by.append("no mean-reversion flag on L2")

    rep.ok = True
    rep.note = (f"{rep.state} / {rep.direction} at "
                f"{rep.level.name if rep.level else '—'}, "
                f"{rep.evidence_count}/{rep.evidence_required} evidence")
    return rep


# ==========================================================================
# Observation log — so a week of watching produces evidence, not impressions
# ==========================================================================
LOG_COLUMNS = [
    "observed_at", "state", "direction", "level_name", "level_price",
    "price_at_observation", "sweep_level", "sweep_bars_ago",
    "evidence_count", "evidence_required", "evidence_present",
    "stretch_atr", "overnight_range_atr", "expected_move_used",
    "regime", "gamma_regime", "suppressed_by",
    "followup_at", "price_after", "move_pts", "went_the_right_way",
]


def empty_log() -> pd.DataFrame:
    frame = pd.DataFrame(columns=LOG_COLUMNS)
    for col in LOG_COLUMNS:
        frame[col] = frame[col].astype(object)
    return frame


def coerce_log(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in LOG_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    text = ["observed_at", "state", "direction", "level_name", "sweep_level",
            "evidence_present", "regime", "gamma_regime", "suppressed_by",
            "followup_at", "went_the_right_way"]
    for col in text:
        out[col] = out[col].astype(object).where(out[col].notna(), "")
    for col in [c for c in LOG_COLUMNS if c not in text]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[LOG_COLUMNS]


def log_observation(store, rep: ReversalReport, options=None,
                    now: datetime | None = None) -> tuple[bool, str]:
    """
    Record TRIGGERED and CONFIRMED observations only. ARMED happens many times
    a day and logging it would bury the signal in its own noise.
    """
    if rep.state not in (TRIGGERED, CONFIRMED):
        return False, f"state {rep.state} — only TRIGGERED/CONFIRMED are logged"

    now = now or datetime.now(ZoneInfo(config.MARKET_TZ))
    frame = coerce_log(store.load())

    window = float(_cfg("REVERSAL_DEDUP_MINUTES"))
    if not frame.empty:
        same = frame[
            (frame["direction"].astype(str) == rep.direction)
            & (frame["level_name"].astype(str) == (rep.level.name if rep.level else ""))
            & (frame["state"].astype(str) == rep.state)
        ]
        if not same.empty:
            stamps = pd.to_datetime(same["observed_at"], errors="coerce", utc=True).dropna()
            cutoff = pd.Timestamp(now)
            cutoff = cutoff.tz_convert("UTC") if cutoff.tzinfo else cutoff.tz_localize("UTC")
            if not stamps.empty and (stamps >= cutoff - timedelta(minutes=window)).any():
                return False, f"duplicate within {window:.0f}min"

    row = {
        "observed_at": pd.Timestamp(now).isoformat(),
        "state": rep.state, "direction": rep.direction,
        "level_name": rep.level.name if rep.level else "",
        "level_price": rep.level.price if rep.level else np.nan,
        "price_at_observation": rep.price,
        "sweep_level": rep.sweep_level, "sweep_bars_ago": rep.sweep_bars_ago,
        "evidence_count": rep.evidence_count,
        "evidence_required": rep.evidence_required,
        "evidence_present": ";".join(e.key for e in rep.evidence if e.present),
        "stretch_atr": rep.stretch_atr, "overnight_range_atr": rep.overnight_range_atr,
        "expected_move_used": rep.expected_move_used,
        "regime": rep.regime,
        "gamma_regime": str(getattr(options, "gamma_regime", "")) if options else "",
        "suppressed_by": "; ".join(rep.suppressed_by),
        "followup_at": "", "price_after": np.nan,
        "move_pts": np.nan, "went_the_right_way": "",
    }
    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    return (True, "logged") if store.save(frame) else (False, "save failed")


def fill_followups(store, current_price: float,
                   now: datetime | None = None) -> int:
    """
    For observations old enough, record what price actually did afterwards.

    This is the whole point of the exercise. Without it, "watch it for a week"
    means eyeballing the dashboard at random moments and remembering the times
    it looked right — which is how a system gets trusted on survivorship.
    """
    if not np.isfinite(current_price):
        return 0
    now = now or datetime.now(ZoneInfo(config.MARKET_TZ))
    frame = coerce_log(store.load())
    if frame.empty:
        return 0

    stamps = pd.to_datetime(frame["observed_at"], errors="coerce", utc=True)
    cutoff = pd.Timestamp(now)
    cutoff = cutoff.tz_convert("UTC") if cutoff.tzinfo else cutoff.tz_localize("UTC")
    minutes = float(_cfg("REVERSAL_FOLLOWUP_MINUTES"))

    due = (frame["followup_at"].astype(str) == "") & \
          (stamps <= cutoff - timedelta(minutes=minutes)) & stamps.notna()
    count = int(due.sum())
    if not count:
        return 0

    for idx in frame.index[due]:
        observed = float(pd.to_numeric(frame.at[idx, "price_at_observation"],
                                       errors="coerce"))
        if not np.isfinite(observed):
            continue
        move = current_price - observed
        bullish = str(frame.at[idx, "direction"]) == "BULLISH"
        frame.at[idx, "followup_at"] = pd.Timestamp(now).isoformat()
        frame.at[idx, "price_after"] = float(current_price)
        frame.at[idx, "move_pts"] = round(move, 1)
        frame.at[idx, "went_the_right_way"] = \
            "yes" if (move > 0) == bullish and abs(move) > 5 else \
            ("no" if abs(move) > 5 else "flat")
    store.save(frame)
    return count


def summarise(frame: pd.DataFrame) -> dict:
    """Plain counts. Deliberately not dressed up as performance."""
    out = {"observations": 0, "resolved": 0, "right": 0, "wrong": 0,
           "flat": 0, "hit_rate": float("nan"), "avg_move": float("nan"),
           "by_state": pd.DataFrame()}
    if frame is None or frame.empty:
        return out
    frame = coerce_log(frame)
    out["observations"] = int(len(frame))
    resolved = frame[frame["went_the_right_way"].astype(str).isin(["yes", "no", "flat"])]
    out["resolved"] = int(len(resolved))
    if resolved.empty:
        return out

    verdicts = resolved["went_the_right_way"].astype(str)
    out["right"] = int((verdicts == "yes").sum())
    out["wrong"] = int((verdicts == "no").sum())
    out["flat"] = int((verdicts == "flat").sum())
    decisive = out["right"] + out["wrong"]
    if decisive:
        out["hit_rate"] = out["right"] / decisive
    moves = pd.to_numeric(resolved["move_pts"], errors="coerce").dropna()
    if not moves.empty:
        out["avg_move"] = float(moves.mean())

    rows = []
    for state, grp in resolved.groupby(resolved["state"].astype(str)):
        v = grp["went_the_right_way"].astype(str)
        right, wrong = int((v == "yes").sum()), int((v == "no").sum())
        mv = pd.to_numeric(grp["move_pts"], errors="coerce").dropna()
        rows.append({
            "state": state, "n": len(grp), "right": right, "wrong": wrong,
            "hit_%": round(right / (right + wrong) * 100, 1) if right + wrong else None,
            "avg_move_pts": round(float(mv.mean()), 1) if not mv.empty else None,
        })
    out["by_state"] = pd.DataFrame(rows).set_index("state") if rows else pd.DataFrame()
    return out


class ReversalStore:
    """
    CSV store for the observation log. Its own schema, so it reuses the
    journal's atomic write but never its columns.

    Same durability caveat as the journal: on Streamlit Cloud the filesystem is
    wiped on reboot and redeploy, so export before you lose a week of watching.
    """

    def __init__(self, path: str | None = None):
        self.path = str(path or _cfg("REVERSAL_LOG_PATH"))

    def load(self) -> pd.DataFrame:
        import os
        if not os.path.exists(self.path):
            return empty_log()
        try:
            return coerce_log(pd.read_csv(self.path))
        except Exception:  # noqa: BLE001
            return empty_log()

    def save(self, frame: pd.DataFrame) -> bool:
        import us30_journal as journal
        writer = journal.CsvStore(self.path)          # atomic temp-then-replace
        return writer.save(coerce_log(frame))

    @property
    def location(self) -> str:
        import os
        return os.path.abspath(self.path)

    @property
    def durable(self) -> bool:
        import us30_journal as journal
        return not journal.on_ephemeral_host()


class MemoryLogStore:
    """In-process store for the tests."""

    def __init__(self, frame: pd.DataFrame | None = None):
        self._frame = empty_log() if frame is None else coerce_log(frame)

    def load(self) -> pd.DataFrame:
        return coerce_log(self._frame)

    def save(self, frame: pd.DataFrame) -> bool:
        self._frame = coerce_log(frame)
        return True

    location = "memory (not persisted)"
    durable = False


def get_store(path: str | None = None) -> ReversalStore:
    return ReversalStore(path)
