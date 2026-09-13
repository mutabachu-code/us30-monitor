"""
config.py — US30 Monitor
Single source of truth for constituents, tickers, thresholds and cache TTLs.

DESIGN NOTE
-----------
The DJIA is PRICE-weighted. Nothing here may assume cap weighting.
COMPONENTS is a seed list only; the live divisor check in dow_attribution.py
is what actually detects constituent changes. Never hard-code weights.
"""

from __future__ import annotations

# Bumped whenever a phase ADDS constants. app.py compares this against what the
# modules expect and names the stale file, instead of dying on a redacted
# AttributeError before a single panel renders.
#   1 = phases 0-3   2 = phase 4 (microstructure)   3 = phase 5 (options)
#   4 = phase 6 (calendar)
CONFIG_VERSION = 4

# ----------------------------------------------------------------------------
# Constituents (seed list — verified 12 Sep 2026, post GOOGL/VZ swap of 29 Jun 2026)
# ----------------------------------------------------------------------------
COMPONENTS: list[str] = [
    "GS", "CAT", "MSFT", "UNH", "AMGN", "TRV", "V", "JPM", "GOOGL", "AAPL",
    "AXP", "SHW", "HD", "JNJ", "AMZN", "MCD", "CRM", "IBM", "NVDA", "CVX",
    "BA", "HON", "MMM", "MRK", "PG", "CSCO", "WMT", "DIS", "KO", "NKE",
]

# GICS-ish sector map. The Dow has NO utilities and NO real estate.
SECTOR_MAP: dict[str, str] = {
    "GS": "Financials", "JPM": "Financials", "V": "Financials",
    "AXP": "Financials", "TRV": "Financials",
    "CAT": "Industrials", "BA": "Industrials", "HON": "Industrials",
    "MMM": "Industrials",
    "MSFT": "Technology", "AAPL": "Technology", "CRM": "Technology",
    "IBM": "Technology", "NVDA": "Technology", "CSCO": "Technology",
    "UNH": "Health Care", "AMGN": "Health Care", "JNJ": "Health Care",
    "MRK": "Health Care",
    "GOOGL": "Communication", "DIS": "Communication",
    "HD": "Cons Disc", "AMZN": "Cons Disc", "MCD": "Cons Disc", "NKE": "Cons Disc",
    "WMT": "Staples", "PG": "Staples", "KO": "Staples",
    "CVX": "Energy",
    "SHW": "Materials",
}

# Sector -> SPDR ETF. Deliberately excludes XLU/XLRE: the Dow contains neither.
SECTOR_ETF: dict[str, str] = {
    "Financials": "XLF",
    "Industrials": "XLI",
    "Technology": "XLK",
    "Health Care": "XLV",
    "Communication": "XLC",
    "Cons Disc": "XLY",
    "Staples": "XLP",
    "Energy": "XLE",
    "Materials": "XLB",
}

# Bellwethers get their own panel: two distinct macro stories, 20.9% of the index.
BELLWETHERS: dict[str, str] = {
    "GS": "Financial conditions / capital markets",
    "CAT": "Global industrial demand",
}

# ----------------------------------------------------------------------------
# Index / instrument tickers
# ----------------------------------------------------------------------------
IDX_SPOT = "^DJI"        # cash index. NOTE: volume field is always 0.
IDX_FUT = "YM=F"         # E-mini Dow. $5 x DJIA, 1.00 index point tick.
IDX_FUT_MICRO = "MYM=F"  # fallback if YM=F is unavailable
IDX_ETF = "DIA"          # ~ DJIA/100, real volume
NDX_SPOT = "^NDX"        # cross-index confirmation (C8)

# Volume-bearing proxies, in preference order. ^DJI is deliberately absent.
VOLUME_SOURCES: list[str] = [IDX_FUT, IDX_ETF, IDX_FUT_MICRO]

MACRO_TICKERS: dict[str, str] = {
    "us10y": "^TNX",        # quoted as yield x 10
    "us5y": "^FVX",
    "us30y": "^TYX",
    "us13w": "^IRX",
    "dxy": "DX-Y.NYB",
    "dxy_fallback": "UUP",
    "wti": "CL=F",
    "gold": "GC=F",
    "vix": "^VIX",
    "vxd": "^VXD",          # Dow-specific vol; verify availability in phase 0
    "spy": "SPY",
}

DIA_TO_INDEX = 100.0   # DIA price x 100 ~ DJIA level (re-derived live, see data_layer)

# ----------------------------------------------------------------------------
# Divisor
# ----------------------------------------------------------------------------
# Published value as of the 29 Jun 2026 reconstitution. Used ONLY as a sanity
# anchor and cold-start fallback — the live value is derived every refresh as
#     divisor = sum(component_prices) / DJI_level
DIVISOR_REFERENCE = 0.16824816528350
DIVISOR_DRIFT_ALERT = 0.001   # 0.1% drift => likely split or constituent change

# ----------------------------------------------------------------------------
# Cache TTLs (seconds). The 30-name batch is the heavy call — keep it honest.
# ----------------------------------------------------------------------------
TTL_COMPONENTS = 900     # 15 min, matches mag7-monitor convention
TTL_INTRADAY = 300       # 5 min
TTL_MACRO = 900
TTL_DAILY = 3600
TTL_OPTIONS = 900
TTL_CALENDAR = 21600     # 6 h

# ----------------------------------------------------------------------------
# Signal thresholds
# ----------------------------------------------------------------------------
# Layer budgets. Sum = 100. No rescaling step, deliberately.
LAYER_WEIGHTS: dict[str, int] = {
    "attribution": 25,
    "technicals": 20,
    "microstructure": 15,
    "macro": 15,
    "regime": 10,
    "options": 10,
    "sectors": 5,
}

CONVICTION_TIERS = [
    (70, "VERY STRONG"),
    (45, "STRONG"),
    (25, "MODERATE"),
    (12, "WEAK"),
    (0, "NEUTRAL"),
]

# C1 move-quality gate
C1_MIN_MOVE_PTS = 150.0
C1_TOP2_SHARE = 0.65
C1_MIN_PW_AD = 5.0

# C2 rates conflict
C2_YIELD_BP = 8.0

# C7 basis dislocation
C7_BASIS_SIGMA = 2.0

# C9 cost gate — expected move must clear round-trip cost by this multiple
C9_COST_MULTIPLE = 3.0
DEFAULT_SPREAD_PTS = 4.0      # overridden by live MT5 spread when available
DEFAULT_SLIPPAGE_PTS = 2.0

# C10 agreement upgrade
C10_MIN_LAYERS_AGREE = 6

# Technicals
RSI_PERIOD = 14
RSI_DECAY_LOOKBACK = 5
ATR_PERIOD = 14
EMA_PERIODS = (9, 21, 50, 200)
CPR_NARROW_ATR = 0.25    # CPR width < 0.25 x ATR20 => narrow (trend day)
CPR_WIDE_ATR = 0.60      # CPR width > 0.60 x ATR20 => wide (range day)

# Risk
ATR_STOP_MULT = 1.2
TP1_R = 1.0
TP2_R = 2.0
MIN_TARGET_PTS = 15.0    # YM ticks in whole points; below this, structure eats the edge

# ----------------------------------------------------------------------------
# Microstructure (phase 4)
# ----------------------------------------------------------------------------
# RVOL is measured against the SAME TIME OF DAY on prior sessions, never against
# a flat daily average — 09:35 always looks like a volume spike next to 12:35.
RVOL_BASELINE_DAYS = 20
RVOL_SPIKE = 1.5          # >= this is an expansion
RVOL_DRY = 0.6            # <= this is a dead tape; suppress breakout reads

# Liquidity sweeps: price takes out a level then closes back inside it.
SWEEP_LOOKBACK_BARS = 78          # roughly one RTH session of 5m bars
SWEEP_EQUAL_TOL_ATR = 0.12        # highs within this x ATR count as "equal"
SWEEP_MIN_PENETRATION_ATR = 0.05  # must genuinely break the level, not touch it
SWEEP_REJECTION_FRAC = 0.5        # close must come back this far inside the range
SWEEP_RVOL_CONFIRM = 1.3          # a sweep on dry volume is noise

# Delta proxy. This is BAR-DERIVED, not real order flow — no L2 data in yfinance.
DELTA_DIVERGENCE_MIN_BARS = 12

# Cash-futures basis. Only meaningful during RTH: ^DJI does not tick outside it.
BASIS_MIN_SAMPLES = 20

# Regular trading hours and the Globex overnight window, US/Eastern.
RTH_START = "09:30"
RTH_END = "16:00"
GLOBEX_START = "18:00"    # bars at or after this belong to the NEXT session

# ----------------------------------------------------------------------------
# Options (phase 5)
# ----------------------------------------------------------------------------
# DIA options trade ~13.9k contracts/day against QQQ's ~1.53M, and DJX index
# options ~3.0k. Strike-level OI that thin produces gamma walls that move
# around day to day, so the QQQ engine from mag7-monitor does NOT port here.
# Instead: pull the top N names by PRICE weight, all of which have deep chains,
# and weight each one's gamma by its DJIA point contribution. Those names are
# roughly half the index, so aggregated component positioning is a truer read
# on Dow dealer exposure than DIA itself.
OPTIONS_TOP_N = 8
OPTIONS_MIN_OI_PER_NAME = 2000      # total OI on the chosen expiry
OPTIONS_MIN_WEIGHT_COVERED = 0.30   # below this the layer reports unavailable
OPTIONS_MAX_DTE = 45                # ignore far-dated expiries
OPTIONS_MIN_DTE_HOURS = 2.0         # T floor: gamma explodes as T -> 0
OPTIONS_MONEYNESS_BAND = 0.15       # keep strikes within +/-15% of spot
OPTIONS_SKEW_DELTA = 0.10           # skew measured at +/-10% moneyness
DIA_THIN_OI = 50_000                # below this, the DIA cross-check is noise
DEFAULT_RISK_FREE = 0.04            # fallback if ^IRX does not resolve

# Gamma regime thresholds on the weight-normalised aggregate, range -1..+1.
GAMMA_LONG_THRESHOLD = 0.15         # dealers long gamma  -> suppresses moves
GAMMA_SHORT_THRESHOLD = -0.15       # dealers short gamma -> amplifies moves
EXPECTED_MOVE_EXHAUSTION = 0.95     # today's move vs expected move

# ----------------------------------------------------------------------------
# Calendar (phase 6)
# ----------------------------------------------------------------------------
# A hardcoded calendar goes silently wrong. This date is what us30_calendar
# compares against to tell you the tables need re-checking, and the FOMC/CPI
# lists below carry their own end dates so the app can say "past the end of
# the table" instead of quietly reporting no upcoming events.
CALENDAR_VERIFIED_ON = "2026-09-13"
CALENDAR_STALE_AFTER_DAYS = 120

# Blackout either side of a high-impact release, in minutes. A 15-90 minute
# scalp has no edge through an FOMC statement.
EVENT_BLACKOUT_BEFORE_MIN = 30
EVENT_BLACKOUT_AFTER_MIN = 15
EVENT_HORIZON_DAYS = 10

# C4: a top-8 name reporting inside this window is 60-250 points of gap risk.
EARNINGS_WINDOW_HOURS = 24

# C8: rolling window for the DJIA/NDX return correlation.
CROSS_CORR_WINDOW = 20
CROSS_CORR_HIGH = 0.70
CROSS_CORR_LOW = 0.35
CROSS_DIRECTION_LOOKBACK = 5

# Time-of-day blocks (US/Eastern). Score the same setup differently by block.
SESSION_BLOCKS = [
    ("09:30", "10:00", "OPEN_AUCTION", 0.6),
    ("10:00", "11:30", "MORNING_TREND", 1.0),
    ("11:30", "13:30", "LUNCH_CHOP", 0.5),
    ("13:30", "15:00", "AFTERNOON_TREND", 1.0),
    ("15:00", "15:45", "LATE_TREND", 0.9),
    ("15:45", "16:00", "MOC_IMBALANCE", 0.4),
]

MARKET_TZ = "America/New_York"
