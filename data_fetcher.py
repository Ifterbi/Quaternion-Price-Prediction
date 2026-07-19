"""Data fetcher module for Bitcoin price data.

Handles downloading, cleaning, and validating OHLCV price data
from Yahoo Finance via the yfinance library. Supports fallback
tickers and robust handling of MultiIndex DataFrames introduced
in yfinance 0.2.51+.
"""

import yfinance as yf
import pandas as pd
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Attempt to import defaults from the project config module.
# If the config module doesn't exist yet, use sensible hard-coded defaults.
# ---------------------------------------------------------------------------
try:
    from config import (
        TICKER_PRIMARY,
        TICKER_FALLBACK,
        DEFAULT_START_DATE,
        DEFAULT_INTERVAL,
    )
except ImportError:
    TICKER_PRIMARY = "BTC-AUD"
    TICKER_FALLBACK = "BTC-USD"
    DEFAULT_START_DATE = "2020-01-01"
    DEFAULT_INTERVAL = "1d"

# Columns every valid OHLCV DataFrame must contain.
EXPECTED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns that yfinance >= 0.2.51 may return.

    Recent versions of yfinance can return a MultiIndex with levels
    ``('Price', 'Ticker')``. This helper drops the ``Ticker`` level so
    downstream code can work with simple column names.

    Args:
        df: DataFrame whose columns may be a MultiIndex.

    Returns:
        DataFrame with single-level column names.
    """
    if isinstance(df.columns, pd.MultiIndex):
        logger.debug(
            "Detected MultiIndex columns (levels=%s). Flattening.",
            df.columns.names,
        )
        # yfinance typically puts Price as the first level and Ticker as
        # the second.  Dropping the Ticker level is the safest approach.
        if "Ticker" in df.columns.names:
            df.columns = df.columns.droplevel("Ticker")
        else:
            # Fallback: just take the first level.
            df.columns = df.columns.get_level_values(0)
    return df


def _download_ticker(
    ticker: str,
    start: str,
    end: Optional[str],
    interval: str,
) -> Optional[pd.DataFrame]:
    """Download data for a single ticker, returning ``None`` on failure.

    Args:
        ticker: Yahoo Finance ticker symbol (e.g. ``"BTC-AUD"``).
        start: Start date string (``"YYYY-MM-DD"``).
        end: Optional end date string. ``None`` means *today*.
        interval: Bar interval (e.g. ``"1d"``, ``"1h"``).

    Returns:
        A DataFrame of OHLCV data, or ``None`` if the download yielded
        no usable rows.
    """
    logger.info("Attempting to download data for %s …", ticker)
    try:
        df = yf.download(ticker, start=start, end=end, interval=interval)
    except Exception:
        logger.exception("yfinance raised an exception for ticker %s", ticker)
        return None

    if df is None or df.empty:
        logger.warning("No data returned for ticker %s.", ticker)
        return None

    df = _flatten_multiindex_columns(df)
    return df


def fetch_bitcoin_data(
    ticker_primary: str = TICKER_PRIMARY,
    ticker_fallback: str = TICKER_FALLBACK,
    start: str = DEFAULT_START_DATE,
    end: Optional[str] = None,
    interval: str = DEFAULT_INTERVAL,
) -> pd.DataFrame:
    """Fetch, clean, and validate Bitcoin OHLCV data.

    Tries ``ticker_primary`` first; if that fails, falls back to
    ``ticker_fallback``. The returned DataFrame is guaranteed to contain
    the columns ``Open``, ``High``, ``Low``, ``Close``, and ``Volume``
    with no NaN values.

    Args:
        ticker_primary: Primary Yahoo Finance ticker symbol.
        ticker_fallback: Fallback ticker if the primary returns no data.
        start: Start date for the historical window (``"YYYY-MM-DD"``).
        end: End date (inclusive). Defaults to ``None`` (today).
        interval: Bar interval (``"1d"``, ``"1h"``, etc.).

    Returns:
        Cleaned ``pd.DataFrame`` with OHLCV columns indexed by date.

    Raises:
        ValueError: If both tickers fail to return usable data.
    """
    # --- 1. Download -------------------------------------------------------
    ticker_used = ticker_primary
    df = _download_ticker(ticker_primary, start, end, interval)

    if df is None or df.empty:
        logger.warning(
            "Primary ticker '%s' returned no data. "
            "Falling back to '%s'.",
            ticker_primary,
            ticker_fallback,
        )
        ticker_used = ticker_fallback
        df = _download_ticker(ticker_fallback, start, end, interval)

    if df is None or df.empty:
        raise ValueError(
            f"Both tickers ('{ticker_primary}', '{ticker_fallback}') "
            "failed to return any data. Check your network connection, "
            "ticker symbols, and date range."
        )

    # --- 2. Clean ----------------------------------------------------------
    # Drop rows where ALL OHLCV values are NaN.
    ohlcv_cols = [c for c in EXPECTED_COLUMNS if c in df.columns]
    df = df.dropna(subset=ohlcv_cols, how="all")

    # Forward-fill remaining NaN values.
    df = df.ffill()

    # Drop any rows that still contain NaN (e.g. leading rows before first
    # valid observation).
    df = df.dropna()

    # --- 3. Validate -------------------------------------------------------
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Downloaded data is missing expected columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # --- 4. Log summary ----------------------------------------------------
    logger.info(
        "Data retrieved — ticker: %s | shape: %s | "
        "date range: %s → %s",
        ticker_used,
        df.shape,
        df.index.min().strftime("%Y-%m-%d"),
        df.index.max().strftime("%Y-%m-%d"),
    )

    return df


def get_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the Open, High, Low, Close, Volume columns from *df*.

    Args:
        df: DataFrame that contains at least the five OHLCV columns.

    Returns:
        A new DataFrame with only the OHLCV columns.

    Raises:
        KeyError: If any of the expected OHLCV columns are missing.
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(
            f"DataFrame is missing OHLCV columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return df[EXPECTED_COLUMNS].copy()


# ---------------------------------------------------------------------------
# Demo / quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    print("Fetching Bitcoin OHLCV data …\n")
    data = fetch_bitcoin_data()

    print(f"Shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    print(f"Date range: {data.index.min()} → {data.index.max()}\n")

    print("— Head —")
    print(data.head())
    print("\n— Tail —")
    print(data.tail())

    ohlcv = get_ohlcv(data)
    print(f"\nOHLCV subset shape: {ohlcv.shape}")
