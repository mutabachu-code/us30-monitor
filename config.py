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
