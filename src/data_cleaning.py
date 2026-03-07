"""
Data Cleaning & Preprocessing Pipeline
=======================================
Standalone script for Exercise 1 — cleans the PJM West hourly energy
consumption dataset.  Run via:

    uv run python src/data_cleaning.py
    uv run python src/data_cleaning.py --input data/PJMW_hourly.csv --output data/PJMW_hourly_cleaned.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def validate_raw(df: pd.DataFrame) -> None:
    """Fail fast if the raw dataframe is malformed."""
    assert "Datetime" in df.columns, "Missing 'Datetime' column"
    assert "PJMW_MW" in df.columns, "Missing 'PJMW_MW' column"
    assert len(df) > 0, "Dataframe is empty"
    assert df["PJMW_MW"].notna().any(), "All MW values are NaN"


def validate_cleaned(df: pd.DataFrame) -> None:
    """Assert postconditions on the cleaned dataframe."""
    assert df.index.is_monotonic_increasing, "Index is not sorted chronologically"
    assert not df.index.duplicated().any(), "Duplicate timestamps remain"
    assert df["PJMW_MW"].isnull().sum() == 0, "NaN values remain after cleaning"
    freq = pd.infer_freq(df.index)
    assert freq is not None and freq.upper() in ("H", "h"), (
        f"Expected hourly frequency, got '{freq}'"
    )


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load the raw CSV dataset."""
    logger.info("Loading data from %s …", filepath)
    df_raw = pd.read_csv(filepath)
    logger.info("Raw dataset shape: %s", df_raw.shape)
    validate_raw(df_raw)
    return df_raw


def detect_outliers(
    df: pd.DataFrame, column: str = "PJMW_MW", factor: float = 3.0
) -> pd.DataFrame:
    """Flag statistical outliers using the IQR method and replace with NaN.

    Values outside [Q1 − factor×IQR, Q3 + factor×IQR] are set to NaN so
    that the downstream interpolation step fills them smoothly.
    """
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - factor * iqr, q3 + factor * iqr
    mask = (df[column] < lower) | (df[column] > upper)
    n_outliers = mask.sum()
    logger.info(
        "Step 3b: Outlier detection (IQR×%.1f) — bounds: %.0f–%.0f MW",
        factor,
        lower,
        upper,
    )
    if n_outliers > 0:
        logger.info("  > Outliers detected: %d", n_outliers)
        for ts in df.index[mask]:
            logger.info(
                "    Flagged: %s → %.1f MW (replaced with NaN)",
                ts,
                df.loc[ts, column],
            )
        df.loc[mask, column] = np.nan
    else:
        logger.info("  > No outliers detected")
    return df, n_outliers


def audit_gaps(df: pd.DataFrame, full_range: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Identify and log the locations of missing hourly timestamps."""
    missing = full_range.difference(df.index)
    if len(missing) == 0:
        logger.info("  > No gaps found")
        return []

    # Identify contiguous gap blocks
    deltas = pd.Series(missing).diff()
    block_starts = deltas[deltas > pd.Timedelta(hours=1)].index.tolist()
    n_blocks = len(block_starts) + 1

    logger.info(
        "  > %d missing hour(s) across %d contiguous gap block(s):",
        len(missing),
        n_blocks,
    )
    for ts in missing:
        logger.debug("    Gap at: %s", ts)

    return missing.tolist()


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Perform the full data cleaning pipeline for Exercise 1.

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe with hourly frequency, no duplicates, no NaNs.
    summary : dict
        Cleaning statistics for programmatic access.
    """
    summary: dict = {"raw_rows": len(df)}

    # 1. Convert Datetime column ------------------------------------------------
    logger.info("Step 1: Converting 'Datetime' column to datetime objects …")
    df["Datetime"] = pd.to_datetime(df["Datetime"])

    # 2. Set index and sort chronologically ------------------------------------
    logger.info("Step 2: Setting index to 'Datetime' and sorting chronologically …")
    df = df.set_index("Datetime")
    df = df.sort_index()

    # 3a. Handle duplicate timestamps ------------------------------------------
    logger.info("Step 3a: Handling duplicate timestamps …")
    duplicates = df.index.duplicated(keep=False)
    n_dup_rows = int(duplicates.sum())
    summary["duplicate_rows"] = n_dup_rows
    logger.info("  > Rows involved in duplicate timestamps: %d", n_dup_rows)
    if n_dup_rows > 0:
        df = df.groupby(df.index).mean()
        summary["unique_timestamps_after_dedup"] = len(df)
        logger.info("  > Shape after deduplication: %s", df.shape)

    # 3b. Detect and remove outliers -------------------------------------------
    df, n_outliers = detect_outliers(df)
    summary["outliers_replaced"] = n_outliers

    # 4. Force hourly frequency and fill gaps ----------------------------------
    logger.info("Step 4: Forcing hourly frequency and filling gaps …")
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="h")
    n_gaps = len(full_range) - len(df)
    summary["expected_records"] = len(full_range)
    summary["actual_records_before_reindex"] = len(df)
    summary["gaps_filled"] = n_gaps

    logger.info("  > Expected hourly records: %d", len(full_range))
    logger.info("  > Actual records:          %d", len(df))
    logger.info("  > Missing hours (gaps):    %d", n_gaps)

    # Audit which hours are missing
    missing_timestamps = audit_gaps(df, full_range)
    summary["missing_timestamps"] = missing_timestamps

    # Reindex to enforce exact hourly frequency
    df = df.reindex(full_range)
    df.index.name = "Datetime"

    # Fill gaps + outlier NaNs via linear interpolation
    df["PJMW_MW"] = df["PJMW_MW"].interpolate(method="linear")

    nulls_remaining = int(df["PJMW_MW"].isnull().sum())
    summary["nulls_after_interpolation"] = nulls_remaining
    summary["final_rows"] = len(df)
    logger.info("  > Null values after interpolation: %d", nulls_remaining)
    logger.info("Final dataset shape: %s", df.shape)

    # Postcondition checks
    validate_cleaned(df)

    return df, summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Parse arguments, run pipeline, save results."""
    parser = argparse.ArgumentParser(
        description="Clean the PJM West hourly energy consumption dataset."
    )
    parser.add_argument(
        "--input",
        default="data/PJMW_hourly.csv",
        help="Path to the raw CSV (default: data/PJMW_hourly.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/PJMW_hourly_cleaned.csv",
        help="Path for the cleaned CSV (default: data/PJMW_hourly_cleaned.csv)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging (shows individual gap timestamps)",
    )
    args = parser.parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )

    df_raw = load_data(args.input)
    df_cleaned, summary = clean_data(df_raw.copy())

    # Save cleaned output
    df_cleaned.to_csv(args.output)
    logger.info("Cleaned dataset saved to %s", args.output)

    # Print summary
    logger.info("--- Cleaning Summary ---")
    for key, value in summary.items():
        if key == "missing_timestamps":
            continue  # already logged during audit
        logger.info("  %s: %s", key, value)

    return df_cleaned, summary


if __name__ == "__main__":
    main()
