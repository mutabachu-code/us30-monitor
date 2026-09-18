"""
us30_calendar.py — US30 Monitor, PHASE 6

Event risk, earnings, ex-dividends and cross-index confirmation. This is NOT
an eighth scoring layer — it is a set of GATES. The seven layers still sum to
±100; what this module supplies is the data C4, C5, C8 and C14 need in order
to block or downgrade what those layers produced. Keeping it out of the score
is deliberate: "there is an FOMC meeting in twenty minutes" is not a bullish
or bearish opinion, it is a reason not to have a position.

THREE DATA-QUALITY POSITIONS
----------------------------
1. NFP IS COMPUTED, NOT LISTED. Non-farm payrolls is the first Friday of the
   month, so it is derived from a rule and can never go stale. Anything
   derivable is derived.

2. FOMC AND CPI ARE HARDCODED, AND HARDCODED TABLES ROT. They are taken from
   federalreserve.gov and the BLS schedule, verified on the date in
   CALENDAR_VERIFIED_ON, and each table carries its own end date. Past that
   end date the module says "past the end of the table" rather than reporting
   no upcoming events — a calendar that silently returns nothing is worse than
   no calendar, because it reads as an all-clear.

3. yfinance EARNINGS DATES ARE OFTEN WRONG. They are frequently stale,
   sometimes months out of date, and occasionally missing entirely for names
   that definitely report. Every earnings date here is labelled with its
   source and confidence, C4 only widens stops rather than blocking, and the
   UI tells you to verify before trusting it. An ex-dividend date inferred
   from historical cadence is marked ESTIMATED and never as fact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import config
import data_layer as dl

# --------------------------------------------------------------------------
# Stale-config tolerance
# --------------------------------------------------------------------------
_CFG_DEFAULTS: dict[str, object] = {
    "CALENDAR_VERIFIED_ON": "2026-09-13",
    "CALENDAR_STALE_AFTER_DAYS": 120,
    "EVENT_BLACKOUT_BEFORE_MIN": 30,
    "EVENT_BLACKOUT_AFTER_MIN": 15,
    "EVENT_HORIZON_DAYS": 10,
    "EARNINGS_WINDOW_HOURS": 24,
    "CROSS_CORR_WINDOW": 20,
    "CROSS_CORR_HIGH": 0.70,
    "CROSS_CORR_LOW": 0.35,
    "CROSS_DIRECTION_LOOKBACK": 5,
}


def _cfg(name: str):
    return getattr(config, name, _CFG_DEFAULTS[name])


def config_health() -> list[str]:
    return [n for n in _CFG_DEFAULTS if not hasattr(config, n)]


# ==========================================================================
# Hardcoded tables — source and verification date recorded on purpose
# ==========================================================================
# FOMC decision days (the SECOND day of each two-day meeting; that is when the
# statement lands at 14:00 ET). Source: federalreserve.gov/monetarypolicy/
# fomccalendars.htm, verified 13 Sep 2026.
FOMC_DECISION_DAYS: list[tuple[str, bool]] = [
    # (date, has Summary of Economic Projections)
    ("2026-01-28", False), ("2026-03-18", True), ("2026-04-29", False),
    ("2026-06-17", True), ("2026-07-29", False), ("2026-09-16", True),
    ("2026-10-28", False), ("2026-12-09", True),
    ("2027-01-27", False), ("2027-03-17", True), ("2027-04-28", False),
    ("2027-06-09", True), ("2027-07-28", False), ("2027-09-15", True),
    ("2027-10-27", False), ("2027-12-08", True),
]
FOMC_TABLE_ENDS = date(2027, 12, 8)

# CPI releases, 08:30 ET. Source: BLS schedule, cross-checked against
# usinflationcalculator.com, verified 13 Sep 2026.
CPI_RELEASE_DAYS: list[str] = [
    "2026-01-13", "2026-02-13", "2026-03-11", "2026-04-10", "2026-05-12",
    "2026-06-10", "2026-07-14", "2026-08-12", "2026-09-11", "2026-10-14",
    "2026-11-10", "2026-12-10",
]
CPI_TABLE_ENDS = date(2026, 12, 10)

EVENT_TIMES = {"FOMC": (14, 0), "CPI": (8, 30), "NFP": (8, 30)}


def nfp_dates(start: date, end: date) -> list[date]:
    """
    Non-farm payrolls: first Friday of each month. Derived from the rule rather
    than listed, so this part of the calendar can never go stale.
    """
    out: list[date] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        first = date(cursor.year, cursor.month, 1)
        # weekday(): Monday=0 ... Friday=4
        offset = (4 - first.weekday()) % 7
        candidate = first + timedelta(days=offset)
        if start <= candidate <= end:
            out.append(candidate)
        cursor = date(cursor.year + (cursor.month == 12),
                      1 if cursor.month == 12 else cursor.month + 1, 1)
    return out


# ==========================================================================
# Results
# ==========================================================================
@dataclass
class Event:
    name: str
    when: datetime          # tz-aware, US/Eastern
    impact: str = "HIGH"
    detail: str = ""

    def minutes_until(self, now: datetime) -> float:
        return (self.when - now).total_seconds() / 60.0


@dataclass
class EarningsEntry:
    ticker: str
    when: datetime | None = None
    source: str = "unavailable"      # calendar / earnings_dates / unavailable
    confidence: str = "none"         # reported / estimated / none
    note: str = ""


@dataclass
class DividendEntry:
    ticker: str
    ex_date: date | None = None
    amount: float = float("nan")
    source: str = "unavailable"      # calendar / inferred / unavailable
    estimated: bool = False


@dataclass
class CalendarReport:
    ok: bool = False
    note: str = ""

    now: datetime | None = None
    upcoming: list[Event] = field(default_factory=list)
    active_blackout: Event | None = None
    next_event: Event | None = None

    earnings: list[EarningsEntry] = field(default_factory=list)
    earnings_within_window: list[str] = field(default_factory=list)

    dividends: list[DividendEntry] = field(default_factory=list)
    ex_div_today: list[str] = field(default_factory=list)

    cross_index: dict = field(default_factory=dict)

    calendar_age_days: int = 0
    calendar_stale: bool = False
    past_table_end: bool = False
    flags: list[str] = field(default_factory=list)

    @property
    def events_frame(self) -> pd.DataFrame:
        if not self.upcoming:
            return pd.DataFrame()
        return pd.DataFrame([{
            "event": e.name,
            "when (ET)": e.when.strftime("%a %d %b %H:%M"),
            "in": _humanise(e.minutes_until(self.now)) if self.now else "",
            "impact": e.impact,
            "detail": e.detail,
        } for e in self.upcoming])


def _humanise(minutes: float) -> str:
    if minutes < 0:
        return "passed"
    if minutes < 60:
        return f"{minutes:.0f}m"
    if minutes < 60 * 24:
        return f"{minutes / 60:.1f}h"
    return f"{minutes / 1440:.1f}d"


# ==========================================================================
# Economic events
# ==========================================================================
def build_events(now: datetime, horizon_days: int | None = None) -> tuple[list[Event], bool]:
    """Upcoming high-impact events. Returns (events, past_table_end)."""
    horizon_days = int(_cfg("EVENT_HORIZON_DAYS")) if horizon_days is None else horizon_days
    tz = ZoneInfo(config.MARKET_TZ)
    today = now.date()
    horizon = today + timedelta(days=horizon_days)

    def at(d: date, kind: str) -> datetime:
        h, m = EVENT_TIMES[kind]
        return datetime(d.year, d.month, d.day, h, m, tzinfo=tz)

    events: list[Event] = []
    for iso, sep in FOMC_DECISION_DAYS:
        d = date.fromisoformat(iso)
        if today <= d <= horizon:
            events.append(Event("FOMC", at(d, "FOMC"), "HIGH",
                                "statement 14:00 ET" + (" + SEP / dot plot" if sep else "")))
    for iso in CPI_RELEASE_DAYS:
        d = date.fromisoformat(iso)
        if today <= d <= horizon:
            events.append(Event("CPI", at(d, "CPI"), "HIGH", "08:30 ET, pre-open"))
    for d in nfp_dates(today, horizon):
        events.append(Event("NFP", at(d, "NFP"), "HIGH", "08:30 ET, pre-open"))

    events.sort(key=lambda e: e.when)

    # Past the end of a hardcoded table, silence is not an all-clear.
    past_end = (today > FOMC_TABLE_ENDS) or (today > CPI_TABLE_ENDS)
    return events, past_end


def find_blackout(events: list[Event], now: datetime) -> Event | None:
    """The event whose blackout window we are currently inside, if any."""
    before = float(_cfg("EVENT_BLACKOUT_BEFORE_MIN"))
    after = float(_cfg("EVENT_BLACKOUT_AFTER_MIN"))
    for e in events:
        mins = e.minutes_until(now)
        if -after <= mins <= before:
            return e
    return None


# ==========================================================================
# Earnings
# ==========================================================================
def _coerce_dates(value) -> list[pd.Timestamp]:
    """yfinance returns earnings dates as a scalar, a list, or a DataFrame."""
    out: list[pd.Timestamp] = []
    if value is None:
        return out
    candidates = value if isinstance(value, (list, tuple, set, pd.Series, np.ndarray)) else [value]
    for item in candidates:
        try:
            ts = pd.Timestamp(item)
            if pd.notna(ts):
                out.append(ts)
        except Exception:  # noqa: BLE001
            continue
    return out


def parse_earnings(ticker: str, calendar_obj, earnings_df,
                   now: datetime | None = None) -> EarningsEntry:
    """
    Pure parser over whatever yfinance handed back. Split out from the fetch so
    the test suite can drive every shape yfinance is known to return.

    `now` is injectable because a parser that reads the wall clock cannot be
    tested deterministically — fixtures written against absolute dates silently
    rot into failures as those dates pass.
    """
    tz = ZoneInfo(config.MARKET_TZ)
    entry = EarningsEntry(ticker=ticker)

    dates: list[pd.Timestamp] = []
    source = "unavailable"

    if isinstance(calendar_obj, dict):
        for key in ("Earnings Date", "earningsDate", "Earnings High"):
            if key in calendar_obj:
                dates = _coerce_dates(calendar_obj[key])
                if dates:
                    source = "calendar"
                    break
    elif isinstance(calendar_obj, pd.DataFrame) and not calendar_obj.empty:
        for key in ("Earnings Date", "earningsDate"):
            if key in calendar_obj.index:
                dates = _coerce_dates(calendar_obj.loc[key].tolist())
                if dates:
                    source = "calendar"
                    break

    if not dates and isinstance(earnings_df, pd.DataFrame) and not earnings_df.empty:
        dates = _coerce_dates(list(earnings_df.index))
        if dates:
            source = "earnings_dates"

    if not dates:
        entry.note = "no earnings date available"
        return entry

    now = now or datetime.now(tz)
    future = []
    for ts in dates:
        dt = ts.to_pydatetime()
        dt = dt.replace(tzinfo=tz) if dt.tzinfo is None else dt.astimezone(tz)
        if dt >= now - timedelta(days=1):
            future.append(dt)

    if not future:
        entry.source = source
        entry.note = "only past earnings dates — yfinance data is stale here"
        return entry

    entry.when = min(future)
    entry.source = source
    entry.confidence = "reported"
    days = (entry.when - now).total_seconds() / 86400.0
    entry.note = f"{entry.when:%a %d %b} ({days:+.1f}d) via {source}"
    return entry


def parse_ex_dividend(ticker: str, calendar_obj, dividend_series) -> DividendEntry:
    """
    Ex-dividend date. A price-weighted index drops mechanically when a
    component goes ex-div: UNH or GS going ex at $3/share is an 18-point index
    drop that carries no information at all.

    Prefers the reported date. Falls back to inferring the next date from
    historical cadence, and marks that clearly as ESTIMATED.
    """
    entry = DividendEntry(ticker=ticker)

    reported = None
    if isinstance(calendar_obj, dict):
        for key in ("Ex-Dividend Date", "exDividendDate"):
            if key in calendar_obj:
                got = _coerce_dates(calendar_obj[key])
                if got:
                    reported = got[0]
                    break
    elif isinstance(calendar_obj, pd.DataFrame) and not calendar_obj.empty:
        for key in ("Ex-Dividend Date", "exDividendDate"):
            if key in calendar_obj.index:
                got = _coerce_dates(calendar_obj.loc[key].tolist())
                if got:
                    reported = got[0]
                    break

    if reported is not None:
        entry.ex_date = reported.date()
        entry.source = "calendar"
        entry.estimated = False
    elif isinstance(dividend_series, pd.Series) and len(dividend_series) >= 4:
        # Infer from cadence. Explicitly an estimate.
        idx = pd.DatetimeIndex(dividend_series.index)
        gaps = pd.Series(idx).diff().dropna()
        if not gaps.empty:
            median_gap = gaps.median()
            last = idx[-1]
            nxt = last + median_gap
            entry.ex_date = nxt.date()
            entry.source = "inferred"
            entry.estimated = True

    if isinstance(dividend_series, pd.Series) and not dividend_series.empty:
        try:
            entry.amount = float(dividend_series.iloc[-1])
        except Exception:  # noqa: BLE001
            pass
    return entry


# ==========================================================================
# Cross-index (C8)
# ==========================================================================
def compute_cross_index(dji: pd.Series, ndx: pd.Series) -> dict:
    """Rolling return correlation and NDX direction for the C8 conflict."""
    out = {"correlation": float("nan"), "ndx_direction": 0,
           "dji_direction": 0, "regime": "UNKNOWN"}
    if dji is None or ndx is None or dji.empty or ndx.empty:
        return out

    joined = pd.DataFrame({"dji": dji, "ndx": ndx}).dropna()
    window = int(_cfg("CROSS_CORR_WINDOW"))
    if len(joined) < window + 2:
        return out

    rets = joined.pct_change().dropna()
    if len(rets) < window:
        return out
    recent = rets.tail(window)
    corr = float(recent["dji"].corr(recent["ndx"]))
    out["correlation"] = corr

    look = int(_cfg("CROSS_DIRECTION_LOOKBACK"))
    if len(joined) > look:
        ndx_chg = float(joined["ndx"].iloc[-1] - joined["ndx"].iloc[-1 - look])
        dji_chg = float(joined["dji"].iloc[-1] - joined["dji"].iloc[-1 - look])
        out["ndx_direction"] = int(np.sign(ndx_chg))
        out["dji_direction"] = int(np.sign(dji_chg))

    if np.isfinite(corr):
        if corr >= float(_cfg("CROSS_CORR_HIGH")):
            out["regime"] = "COUPLED"
        elif corr <= float(_cfg("CROSS_CORR_LOW")):
            out["regime"] = "ROTATION"
        else:
            out["regime"] = "MIXED"
    return out


# ==========================================================================
# Assembly
# ==========================================================================
def build_report(now: datetime,
                 earnings: list[EarningsEntry],
                 dividends: list[DividendEntry],
                 cross: dict) -> CalendarReport:
    """Pure assembly — no I/O, so the whole thing is testable."""
    rep = CalendarReport(now=now, earnings=earnings, dividends=dividends,
                         cross_index=cross)

    rep.upcoming, rep.past_table_end = build_events(now)
    rep.next_event = rep.upcoming[0] if rep.upcoming else None
    rep.active_blackout = find_blackout(rep.upcoming, now)

    window_h = float(_cfg("EARNINGS_WINDOW_HOURS"))
    for e in earnings:
        if e.when is None:
            continue
        hours = (e.when - now).total_seconds() / 3600.0
        if -2.0 <= hours <= window_h:
            rep.earnings_within_window.append(e.ticker)

    today = now.date()
    for d in dividends:
        if d.ex_date == today:
            rep.ex_div_today.append(d.ticker)
            if d.estimated:
                rep.flags.append(
                    f"{d.ticker} ex-dividend date is INFERRED from historical "
                    f"cadence, not reported — verify before trusting the "
                    f"attribution adjustment"
                )

    # ---- staleness -------------------------------------------------------
    try:
        verified = date.fromisoformat(str(_cfg("CALENDAR_VERIFIED_ON")))
        rep.calendar_age_days = (today - verified).days
        rep.calendar_stale = rep.calendar_age_days > int(_cfg("CALENDAR_STALE_AFTER_DAYS"))
    except Exception:  # noqa: BLE001
        rep.calendar_age_days = 0

    if rep.calendar_stale:
        rep.flags.append(
            f"Hardcoded FOMC/CPI tables were verified {rep.calendar_age_days} days "
            f"ago — re-check them against federalreserve.gov and the BLS schedule"
        )
    if rep.past_table_end:
        rep.flags.append(
            "Past the end of a hardcoded table — event list is INCOMPLETE. "
            "An empty calendar here is not an all-clear."
        )
    unresolved = [e.ticker for e in earnings if e.when is None]
    if unresolved:
        rep.flags.append(
            f"No usable earnings date for {', '.join(unresolved)} — yfinance "
            f"earnings data is frequently stale, so C4 cannot protect these names"
        )

    rep.ok = True
    rep.note = (
        f"{len(rep.upcoming)} events in horizon, "
        f"{len(rep.earnings_within_window)} earnings in window, "
        f"{len(rep.ex_div_today)} ex-div today, "
        f"cross {cross.get('regime', 'UNKNOWN')}"
    )
    return rep


def get_calendar(tickers: list[str] | None = None) -> CalendarReport:
    """Live calendar. Never raises."""
    tz = ZoneInfo(config.MARKET_TZ)
    now = datetime.now(tz)
    tickers = list(tickers or config.COMPONENTS[:int(getattr(config, "OPTIONS_TOP_N", 8))])

    earnings: list[EarningsEntry] = []
    dividends: list[DividendEntry] = []
    try:
        import yfinance as yf
        have_yf = True
    except Exception:  # noqa: BLE001
        have_yf = False

    if have_yf and not dl.breaker_open():
        for tk in tickers:
            cal_obj, earn_df, divs = None, None, None
            try:
                t = yf.Ticker(tk)
                try:
                    cal_obj = t.calendar
                except Exception:  # noqa: BLE001
                    cal_obj = None
                try:
                    earn_df = t.get_earnings_dates(limit=8)
                except Exception:  # noqa: BLE001
                    earn_df = None
                try:
                    divs = t.dividends
                except Exception:  # noqa: BLE001
                    divs = None
            except Exception:  # noqa: BLE001
                pass
            earnings.append(parse_earnings(tk, cal_obj, earn_df))
            dividends.append(parse_ex_dividend(tk, cal_obj, divs))
    else:
        earnings = [EarningsEntry(ticker=tk, note="yfinance unavailable") for tk in tickers]
        dividends = [DividendEntry(ticker=tk) for tk in tickers]

    cross = {}
    try:
        dji = dl.closes(dl.get_index_daily(period="6mo"))
        ndx = dl.closes(dl.get_cross_index(period="6mo"))
        if not dji.empty and not ndx.empty:
            cross = compute_cross_index(dji.iloc[:, 0], ndx.iloc[:, 0])
    except Exception:  # noqa: BLE001
        cross = {}

    rep = build_report(now, earnings, dividends, cross)
    if not have_yf:
        rep.flags.append("yfinance unavailable — earnings and ex-dividend gates inactive")
    return rep
