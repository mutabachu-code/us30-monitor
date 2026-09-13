"""
us30_journal.py — US30 Monitor, PHASE 7

The signal ledger, and the only part of this repo that can honestly answer the
question the whole build started with.

Everything up to here is a hypothesis. Six phases of engines produce a number,
and that number's relationship to money is entirely unmeasured. This module
records what the dashboard claimed, what actually happened, and computes the
metrics that matter — expectancy in index points, profit factor, max
consecutive losses — rather than the one that feels best.

FOUR POSITIONS THIS MODULE TAKES
--------------------------------
1. WIN RATE IS ALWAYS REPORTED WITH A CONFIDENCE INTERVAL. "62%" from 21 trades
   is not a fact, it is a number with a 95% interval running roughly 41%-79%.
   Reporting the bare percentage is how a system talks its owner into trusting
   noise. Every win rate here carries its Wilson interval and its n.

2. EXPECTANCY IS THE HEADLINE, NOT WIN RATE. Break-even win rate is
   (1 + cost/risk) / (R + 1). At 0.5:1 reward-to-risk with realistic costs that
   is 75% — so a 75% win rate there earns exactly nothing. The journal grades
   against expectancy in points and profit factor, which is what compounds.

3. EVERY LAYER SCORE IS STORED. Not just the composite. Without per-layer
   columns you can never afterwards ask which layers actually carried
   information and which were decoration, and that question is worth more than
   the win rate.

4. THE FORWARD TEST NEEDS DAYS *AND* TRADES. 30 trades from one week is one
   market regime wearing a sample's clothing. Both gates must clear.

PERSISTENCE WARNING
-------------------
Streamlit Cloud's filesystem is EPHEMERAL. It is wiped on every reboot and
redeploy. A journal written only there will silently lose a 60-day forward
test, and you would not find out until you went looking for the results. See
`storage_warning()` and the Journal tab: export regularly, and prefer running
the durable copy on the MT5 host, where the filesystem actually persists.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
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
    "JOURNAL_PATH": "us30_journal.csv",
    "JOURNAL_DEDUP_MINUTES": 20,
    "FORWARD_TEST_MIN_DAYS": 60,
    "FORWARD_TEST_MIN_TRADES": 30,
    "JOURNAL_MIN_SAMPLE": 30,
    "JOURNAL_CI_Z": 1.96,
    "TARGET_EXPECTANCY_PTS": 8.0,
    "TARGET_PROFIT_FACTOR": 1.4,
}


def _cfg(name: str):
    return getattr(config, name, _CFG_DEFAULTS[name])


def config_health() -> list[str]:
    return [n for n in _CFG_DEFAULTS if not hasattr(config, n)]


# ==========================================================================
# Schema
# ==========================================================================
COLUMNS = [
    "signal_id", "logged_at", "session_block",
    "direction", "score", "conviction", "coverage", "confidence",
    "entry", "stop", "tp1", "tp2", "risk_pts", "rr", "lot_multiplier",
    # Per-layer scores: the columns that let you ask, later, which layers
    # actually predicted anything.
    "l1_attribution", "l2_technicals", "l3_micro", "l4_macro",
    "l5_regime", "l6_options", "l7_sectors",
    "regime", "dispersion_regime", "gamma_regime",
    "conflicts", "blocked", "block_reasons",
    "spread_pts", "slippage_pts",
    "status", "exit_at", "exit_price", "exit_reason",
    "realised_pts", "realised_r", "actual_entry", "actual_slippage_pts",
    "paper", "notes",
]

OPEN = "OPEN"
WIN = "WIN"
LOSS = "LOSS"
SCRATCH = "SCRATCH"
EXPIRED = "EXPIRED"
CLOSED_STATUSES = (WIN, LOSS, SCRATCH)


# Dtypes have to be pinned explicitly. A CSV whose text columns are entirely
# empty reads back as float64, and pandas 3 then REFUSES to write a string into
# it ("Invalid value for dtype 'float64'"). That fails on the very first outcome
# recorded against a fresh journal — the real storage path, which an in-memory
# test never exercises.
TEXT_COLUMNS = [
    "signal_id", "logged_at", "session_block", "direction", "conviction",
    "regime", "dispersion_regime", "gamma_regime", "conflicts",
    "block_reasons", "status", "exit_at", "exit_reason", "notes",
]
BOOL_COLUMNS = ["blocked", "paper"]
NUMERIC_COLUMNS = [c for c in COLUMNS if c not in TEXT_COLUMNS + BOOL_COLUMNS]


def coerce_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Force every column to a dtype that accepts what we later write into it."""
    out = frame.copy()
    for col in COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    for col in TEXT_COLUMNS:
        out[col] = out[col].astype(object).where(out[col].notna(), "")
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in BOOL_COLUMNS:
        out[col] = out[col].astype(object)
    return out[COLUMNS]


def empty_frame() -> pd.DataFrame:
    return coerce_schema(pd.DataFrame(columns=COLUMNS))


# ==========================================================================
# Storage
# ==========================================================================
class MemoryStore:
    """In-process store. Used by the tests and as a safe fallback."""

    def __init__(self, frame: pd.DataFrame | None = None):
        self._frame = empty_frame() if frame is None else coerce_schema(frame)

    def load(self) -> pd.DataFrame:
        # Coerce on the way out too, so MemoryStore and CsvStore behave
        # identically — the dtype bug only ever appeared on one of them.
        return coerce_schema(self._frame)

    def save(self, frame: pd.DataFrame) -> bool:
        self._frame = coerce_schema(frame)
        return True

    @property
    def location(self) -> str:
        return "memory (not persisted)"

    @property
    def durable(self) -> bool:
        return False


class CsvStore:
    """
    CSV on the local filesystem, written atomically.

    Durable on the MT5 host. NOT durable on Streamlit Cloud, which wipes the
    filesystem on reboot and redeploy — `durable` reports that honestly rather
    than letting the caller assume the data is safe.
    """

    def __init__(self, path: str | None = None):
        self.path = str(path or _cfg("JOURNAL_PATH"))

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            return empty_frame()
        try:
            frame = pd.read_csv(self.path)
        except Exception:  # noqa: BLE001
            return empty_frame()
        return coerce_schema(frame)

    def save(self, frame: pd.DataFrame) -> bool:
        """Write to a temp file then replace, so a crash mid-write cannot
        truncate an existing journal to nothing."""
        try:
            directory = os.path.dirname(os.path.abspath(self.path)) or "."
            os.makedirs(directory, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
            os.close(fd)
            frame.to_csv(tmp, index=False)
            os.replace(tmp, self.path)
            return True
        except Exception:  # noqa: BLE001
            return False

    @property
    def location(self) -> str:
        return os.path.abspath(self.path)

    @property
    def durable(self) -> bool:
        return not on_ephemeral_host()


def on_ephemeral_host() -> bool:
    """
    Best-effort detection of Streamlit Cloud, whose filesystem does not survive
    a reboot. Deliberately errs toward assuming ephemeral: a false warning
    costs an export, a false all-clear costs the forward test.
    """
    markers = ("STREAMLIT_RUNTIME_ENV", "STREAMLIT_SHARING_MODE",
               "STREAMLIT_SERVER_ADDRESS")
    if any(os.environ.get(m) for m in markers):
        return True
    return os.path.exists("/mount/src")


def storage_warning(store) -> str:
    """The sentence the UI must show. Empty when storage really is durable."""
    if getattr(store, "durable", False):
        return ""
    return (
        "This journal is NOT durable here. Streamlit Cloud wipes the filesystem "
        "on every reboot and redeploy, so a 60-day forward test written only to "
        "this host will disappear without warning. Export the CSV regularly, and "
        "run the durable copy on the MT5 host where the filesystem persists."
    )


# ==========================================================================
# Fingerprinting
# ==========================================================================
def fingerprint(direction: str, score: float, session_block: str,
                entry: float) -> str:
    """
    Identity for a setup, coarse on purpose. The dashboard re-renders on every
    interaction, so without this one setup would be logged dozens of times and
    the sample would be fiction — inflated n and heavily correlated rows.

    Score buckets to the nearest 10 and entry to the nearest 25 index points,
    so a drifting recomputation of the same setup collapses to one row.
    """
    bucket_score = int(round(float(score) / 10.0)) if np.isfinite(score) else 0
    bucket_entry = int(round(float(entry) / 25.0)) if np.isfinite(entry) else 0
    raw = f"{direction}|{bucket_score}|{session_block}|{bucket_entry}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def is_duplicate(frame: pd.DataFrame, fp: str, now: datetime,
                 window_minutes: float | None = None) -> bool:
    if frame is None or frame.empty or "signal_id" not in frame.columns:
        return False
    window = float(_cfg("JOURNAL_DEDUP_MINUTES")) if window_minutes is None else window_minutes
    matches = frame[frame["signal_id"].astype(str) == fp]
    if matches.empty:
        return False
    stamps = pd.to_datetime(matches["logged_at"], errors="coerce", utc=True).dropna()
    if stamps.empty:
        return False
    cutoff = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo \
        else pd.Timestamp(now).tz_localize("UTC")
    return bool((stamps >= cutoff - timedelta(minutes=window)).any())


# ==========================================================================
# Logging
# ==========================================================================
def _layer_score(signal, key: str) -> float:
    for layer in getattr(signal, "layers", []) or []:
        if getattr(layer, "key", "") == key and getattr(layer, "available", False):
            return round(float(getattr(layer, "score", 0.0)), 2)
    return float("nan")


def row_from_signal(signal, now: datetime, regime=None, options=None,
                    paper: bool = True) -> dict:
    """Pure: turn a MasterSignal into a journal row."""
    plan = getattr(signal, "plan", None)
    fp = fingerprint(getattr(signal, "direction", "NEUTRAL"),
                     getattr(signal, "final_score", 0.0),
                     getattr(signal, "session_block", "UNKNOWN"),
                     getattr(plan, "entry", float("nan")) if plan else float("nan"))
    return {
        "signal_id": fp,
        "logged_at": pd.Timestamp(now).isoformat(),
        "session_block": getattr(signal, "session_block", "UNKNOWN"),
        "direction": getattr(signal, "direction", "NEUTRAL"),
        "score": round(float(getattr(signal, "final_score", 0.0)), 2),
        "conviction": getattr(signal, "conviction", "NEUTRAL"),
        "coverage": round(float(getattr(signal, "coverage", 0.0)), 3),
        "confidence": round(float(getattr(signal, "confidence", 0.0)), 3),
        "entry": getattr(plan, "entry", np.nan) if plan else np.nan,
        "stop": getattr(plan, "stop", np.nan) if plan else np.nan,
        "tp1": getattr(plan, "tp1", np.nan) if plan else np.nan,
        "tp2": getattr(plan, "tp2", np.nan) if plan else np.nan,
        "risk_pts": getattr(plan, "risk_pts", np.nan) if plan else np.nan,
        "rr": getattr(plan, "rr", np.nan) if plan else np.nan,
        "lot_multiplier": getattr(plan, "lot_multiplier", np.nan) if plan else np.nan,
        "l1_attribution": _layer_score(signal, "attribution"),
        "l2_technicals": _layer_score(signal, "technicals"),
        "l3_micro": _layer_score(signal, "microstructure"),
        "l4_macro": _layer_score(signal, "macro"),
        "l5_regime": _layer_score(signal, "regime"),
        "l6_options": _layer_score(signal, "options"),
        "l7_sectors": _layer_score(signal, "sectors"),
        "regime": getattr(regime, "regime", "") if regime else "",
        "dispersion_regime": getattr(regime, "dispersion_regime", "") if regime else "",
        "gamma_regime": getattr(options, "gamma_regime", "") if options else "",
        "conflicts": ";".join(c.code for c in getattr(signal, "conflicts", []) or []),
        "blocked": bool(getattr(signal, "blocked", False)),
        "block_reasons": "; ".join(getattr(signal, "block_reasons", []) or []),
        "spread_pts": np.nan,
        "slippage_pts": np.nan,
        "status": OPEN,
        "exit_at": "", "exit_price": np.nan, "exit_reason": "",
        "realised_pts": np.nan, "realised_r": np.nan,
        "actual_entry": np.nan, "actual_slippage_pts": np.nan,
        "paper": bool(paper),
        "notes": "",
    }


def log_signal(store, signal, now: datetime | None = None, regime=None,
               options=None, paper: bool = True,
               spread_pts: float = float("nan"),
               slippage_pts: float = float("nan")) -> tuple[bool, str]:
    """
    Record an actionable signal. Returns (logged, reason).

    Only actionable setups are logged. A blocked signal is the system working
    correctly, not a trade, and filling the ledger with blocked rows would make
    every rate computed from it meaningless.
    """
    now = now or datetime.now(ZoneInfo(config.MARKET_TZ))
    plan = getattr(signal, "plan", None)
    if not getattr(signal, "ok", False):
        return False, "signal not ok"
    if getattr(signal, "blocked", False):
        return False, "blocked — not a trade, deliberately not logged"
    if plan is None or not getattr(plan, "valid", False):
        return False, "no actionable plan"

    frame = store.load()
    row = row_from_signal(signal, now, regime, options, paper)
    row["spread_pts"] = spread_pts
    row["slippage_pts"] = slippage_pts

    if is_duplicate(frame, row["signal_id"], now):
        return False, f"duplicate within {_cfg('JOURNAL_DEDUP_MINUTES')}min window"

    frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
    return (True, "logged") if store.save(frame) else (False, "save failed")


def record_outcome(store, signal_id: str, exit_price: float,
                   exit_at: datetime | None = None, exit_reason: str = "",
                   actual_entry: float | None = None,
                   notes: str = "") -> tuple[bool, str]:
    """
    Close a logged signal and compute its realised result.

    Realised points are measured from the ACTUAL fill when one is supplied,
    not from the planned entry. The difference between the two is slippage,
    and slippage is exactly the thing a backtest lies about.
    """
    frame = coerce_schema(store.load())
    if frame.empty:
        return False, "journal is empty"
    mask = frame["signal_id"].astype(str) == str(signal_id)
    open_mask = mask & (frame["status"].astype(str) == OPEN)
    if not open_mask.any():
        return False, "no open row with that id"

    idx = frame.index[open_mask][-1]
    planned_entry = float(pd.to_numeric(frame.at[idx, "entry"], errors="coerce"))
    entry_used = float(actual_entry) if actual_entry is not None and np.isfinite(actual_entry) \
        else planned_entry
    direction = str(frame.at[idx, "direction"])
    risk = float(pd.to_numeric(frame.at[idx, "risk_pts"], errors="coerce"))

    sign = 1.0 if direction == "LONG" else -1.0
    realised = (float(exit_price) - entry_used) * sign

    frame.at[idx, "exit_at"] = pd.Timestamp(
        exit_at or datetime.now(ZoneInfo(config.MARKET_TZ))).isoformat()
    frame.at[idx, "exit_price"] = float(exit_price)
    frame.at[idx, "exit_reason"] = exit_reason
    frame.at[idx, "realised_pts"] = round(realised, 2)
    frame.at[idx, "realised_r"] = round(realised / risk, 3) if risk and np.isfinite(risk) and risk > 0 else np.nan
    if actual_entry is not None and np.isfinite(actual_entry) and np.isfinite(planned_entry):
        frame.at[idx, "actual_entry"] = float(actual_entry)
        frame.at[idx, "actual_slippage_pts"] = round(
            abs(float(actual_entry) - planned_entry), 2)
    if notes:
        frame.at[idx, "notes"] = notes

    if realised > 0.5:
        frame.at[idx, "status"] = WIN
    elif realised < -0.5:
        frame.at[idx, "status"] = LOSS
    else:
        frame.at[idx, "status"] = SCRATCH

    return (True, str(frame.at[idx, "status"])) if store.save(frame) else (False, "save failed")


def expire_stale(store, now: datetime | None = None,
                 max_age_hours: float = 24.0) -> int:
    """Open rows older than max_age_hours never got an outcome. Mark them
    EXPIRED rather than leaving them to inflate the open count forever."""
    now = now or datetime.now(ZoneInfo(config.MARKET_TZ))
    frame = coerce_schema(store.load())
    if frame.empty:
        return 0
    stamps = pd.to_datetime(frame["logged_at"], errors="coerce", utc=True)
    cutoff = pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo \
        else pd.Timestamp(now).tz_localize("UTC")
    stale = (frame["status"].astype(str) == OPEN) & \
            (stamps < cutoff - timedelta(hours=max_age_hours))
    count = int(stale.sum())
    if count:
        frame.loc[stale, "status"] = EXPIRED
        store.save(frame)
    return count


# ==========================================================================
# Statistics
# ==========================================================================
def wilson_interval(wins: int, n: int, z: float | None = None) -> tuple[float, float]:
    """
    Wilson score interval for a proportion.

    Used instead of a bare percentage everywhere, because "62%" from 21 trades
    is not a fact — it is a number whose 95% interval runs from roughly 41% to
    79%, which is the difference between a business and a coin flip.
    """
    z = float(_cfg("JOURNAL_CI_Z")) if z is None else z
    if n <= 0:
        return float("nan"), float("nan")
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - margin), min(1.0, centre + margin)


def max_consecutive(series: pd.Series, value: str) -> int:
    run = best = 0
    for item in series:
        if str(item) == value:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


@dataclass
class Metrics:
    n_closed: int = 0
    n_open: int = 0
    n_expired: int = 0

    wins: int = 0
    losses: int = 0
    scratches: int = 0

    win_rate: float = float("nan")
    win_rate_low: float = float("nan")
    win_rate_high: float = float("nan")

    expectancy_pts: float = float("nan")
    expectancy_r: float = float("nan")
    profit_factor: float = float("nan")
    total_pts: float = float("nan")

    avg_win_pts: float = float("nan")
    avg_loss_pts: float = float("nan")
    max_consecutive_losses: int = 0

    avg_slippage_pts: float = float("nan")

    first_trade: str = ""
    last_trade: str = ""
    days_elapsed: int = 0

    by_regime: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_session: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_conviction: pd.DataFrame = field(default_factory=pd.DataFrame)
    layer_correlation: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    sample_adequate: bool = False
    forward_test_complete: bool = False
    verdict: str = "INSUFFICIENT DATA"
    warnings: list[str] = field(default_factory=list)

    @property
    def win_rate_text(self) -> str:
        if not np.isfinite(self.win_rate):
            return "—"
        return (f"{self.win_rate * 100:.0f}% "
                f"(95% CI {self.win_rate_low * 100:.0f}–{self.win_rate_high * 100:.0f}%, "
                f"n={self.n_closed})")


def _group_metrics(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame()
    rows = []
    for key, grp in frame.groupby(frame[column].fillna("").astype(str)):
        if not key:
            continue
        pts = pd.to_numeric(grp["realised_pts"], errors="coerce").dropna()
        if pts.empty:
            continue
        wins = int((grp["status"].astype(str) == WIN).sum())
        n = int(len(grp))
        lo, hi = wilson_interval(wins, n)
        rows.append({
            column: key, "n": n, "wins": wins,
            "win_%": round(wins / n * 100, 1),
            "CI_low_%": round(lo * 100, 1), "CI_high_%": round(hi * 100, 1),
            "expectancy_pts": round(float(pts.mean()), 1),
            "total_pts": round(float(pts.sum()), 1),
        })
    out = pd.DataFrame(rows)
    return out.set_index(column).sort_values("n", ascending=False) if not out.empty else out


def compute_metrics(frame: pd.DataFrame) -> Metrics:
    """Pure. Everything the app reports about performance comes from here."""
    m = Metrics()
    if frame is None or frame.empty:
        m.warnings.append("Journal is empty — nothing has been recorded yet.")
        return m

    status = frame["status"].astype(str)
    m.n_open = int((status == OPEN).sum())
    m.n_expired = int((status == EXPIRED).sum())

    closed = frame[status.isin(CLOSED_STATUSES)].copy()
    m.n_closed = int(len(closed))
    if m.n_closed == 0:
        m.warnings.append(
            f"{m.n_open} signals logged but none closed yet — record outcomes "
            f"before any statistic here means anything."
        )
        return m

    closed["realised_pts"] = pd.to_numeric(closed["realised_pts"], errors="coerce")
    closed = closed.dropna(subset=["realised_pts"])
    m.n_closed = int(len(closed))
    if m.n_closed == 0:
        m.warnings.append("Closed rows carry no realised points — outcomes incomplete.")
        return m

    cs = closed["status"].astype(str)
    m.wins = int((cs == WIN).sum())
    m.losses = int((cs == LOSS).sum())
    m.scratches = int((cs == SCRATCH).sum())

    m.win_rate = m.wins / m.n_closed
    m.win_rate_low, m.win_rate_high = wilson_interval(m.wins, m.n_closed)

    pts = closed["realised_pts"]
    m.expectancy_pts = float(pts.mean())
    m.total_pts = float(pts.sum())
    rs = pd.to_numeric(closed["realised_r"], errors="coerce").dropna()
    m.expectancy_r = float(rs.mean()) if not rs.empty else float("nan")

    gains = pts[pts > 0].sum()
    pains = abs(pts[pts < 0].sum())
    m.profit_factor = float(gains / pains) if pains > 0 else float("inf") if gains > 0 else float("nan")
    m.avg_win_pts = float(pts[pts > 0].mean()) if (pts > 0).any() else float("nan")
    m.avg_loss_pts = float(pts[pts < 0].mean()) if (pts < 0).any() else float("nan")

    ordered = closed.sort_values("logged_at")
    m.max_consecutive_losses = max_consecutive(ordered["status"].astype(str), LOSS)

    slip = pd.to_numeric(closed["actual_slippage_pts"], errors="coerce").dropna()
    m.avg_slippage_pts = float(slip.mean()) if not slip.empty else float("nan")

    stamps = pd.to_datetime(ordered["logged_at"], errors="coerce", utc=True).dropna()
    if not stamps.empty:
        m.first_trade = stamps.iloc[0].strftime("%Y-%m-%d")
        m.last_trade = stamps.iloc[-1].strftime("%Y-%m-%d")
        m.days_elapsed = int((stamps.iloc[-1] - stamps.iloc[0]).days) + 1

    m.by_regime = _group_metrics(closed, "regime")
    m.by_session = _group_metrics(closed, "session_block")
    m.by_conviction = _group_metrics(closed, "conviction")

    # Which layers actually predicted anything? This is the question the
    # per-layer columns exist to answer.
    corr = {}
    for col in ("l1_attribution", "l2_technicals", "l3_micro", "l4_macro",
                "l5_regime", "l6_options", "l7_sectors"):
        vals = pd.to_numeric(closed[col], errors="coerce")
        pair = pd.DataFrame({"x": vals, "y": pts}).dropna()
        if len(pair) >= 10 and pair["x"].std(ddof=0) > 1e-9:
            corr[col] = round(float(pair["x"].corr(pair["y"])), 3)
    m.layer_correlation = pd.Series(corr).sort_values(ascending=False) \
        if corr else pd.Series(dtype=float)

    # ---- gates and verdict ------------------------------------------------
    min_n = int(_cfg("JOURNAL_MIN_SAMPLE"))
    min_days = int(_cfg("FORWARD_TEST_MIN_DAYS"))
    min_trades = int(_cfg("FORWARD_TEST_MIN_TRADES"))

    m.sample_adequate = m.n_closed >= min_n
    m.forward_test_complete = (m.n_closed >= min_trades) and (m.days_elapsed >= min_days)

    if not m.sample_adequate:
        m.warnings.append(
            f"n={m.n_closed} is below the {min_n}-trade floor. The win-rate "
            f"interval spans {(m.win_rate_high - m.win_rate_low) * 100:.0f} "
            f"percentage points — that is noise, not a result."
        )
    if not m.forward_test_complete:
        m.warnings.append(
            f"Forward test incomplete: {m.n_closed}/{min_trades} trades over "
            f"{m.days_elapsed}/{min_days} days. BOTH gates must clear — 30 trades "
            f"from one week is one market regime, not a sample."
        )
    if m.scratches > m.n_closed * 0.25:
        m.warnings.append(
            f"{m.scratches} of {m.n_closed} closed at scratch — check whether "
            f"targets are inside the noise band."
        )
    if np.isfinite(m.avg_slippage_pts) and np.isfinite(m.expectancy_pts) \
            and m.expectancy_pts > 0 and m.avg_slippage_pts > m.expectancy_pts * 0.5:
        m.warnings.append(
            f"Average slippage {m.avg_slippage_pts:.1f}pts is over half of "
            f"expectancy {m.expectancy_pts:.1f}pts — execution is eating the edge."
        )

    target_e = float(_cfg("TARGET_EXPECTANCY_PTS"))
    target_pf = float(_cfg("TARGET_PROFIT_FACTOR"))
    if not m.forward_test_complete:
        m.verdict = "FORWARD TEST IN PROGRESS"
    elif m.expectancy_pts >= target_e and m.profit_factor >= target_pf:
        m.verdict = "MEETS TARGETS"
    elif m.expectancy_pts > 0:
        m.verdict = "POSITIVE BUT BELOW TARGET"
    else:
        m.verdict = "NEGATIVE EXPECTANCY"
    return m


# ==========================================================================
# Break-even reference — the maths from the research doc, live
# ==========================================================================
def breakeven_win_rate(rr: float, cost_pts: float, risk_pts: float) -> float:
    """
    (1 + cost/risk) / (R + 1)

    At R=0.5 with a 40pt stop and 5pts of cost this returns exactly 0.75 — so a
    75% win rate there is break-even, not success. Exposed as a function so the
    app can show the identity against the journal's own realised numbers rather
    than restating it as a claim.
    """
    if risk_pts <= 0 or rr < 0:
        return float("nan")
    return (1.0 + cost_pts / risk_pts) / (rr + 1.0)


def grade_against_breakeven(m: Metrics, cost_pts: float) -> dict:
    """Compare realised win rate to what this system's own R actually needs."""
    out = {"required": float("nan"), "achieved": m.win_rate,
           "margin": float("nan"), "verdict": "unknown"}
    if not np.isfinite(m.win_rate) or m.n_closed == 0:
        return out
    if np.isfinite(m.avg_win_pts) and np.isfinite(m.avg_loss_pts) and m.avg_loss_pts != 0:
        rr = abs(m.avg_win_pts / m.avg_loss_pts)
        risk = abs(m.avg_loss_pts)
        req = breakeven_win_rate(rr, cost_pts, risk)
        out["required"] = req
        out["margin"] = m.win_rate - req
        if not np.isfinite(req):
            out["verdict"] = "unknown"
        elif m.win_rate_low > req:
            out["verdict"] = "clears break-even with confidence"
        elif m.win_rate > req:
            out["verdict"] = "above break-even, but the interval still straddles it"
        else:
            out["verdict"] = "below break-even"
    return out


def get_store(path: str | None = None):
    """Default store. CSV, with an honest durability flag."""
    return CsvStore(path)
