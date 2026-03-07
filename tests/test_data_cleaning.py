"""
Unit tests for the data cleaning pipeline.

Run with:
    uv run python -m pytest tests/test_data_cleaning.py -v
"""

import numpy as np
import pandas as pd
import pytest

# Allow imports from src/
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_cleaning import (
    audit_gaps,
    clean_data,
    detect_outliers,
    validate_cleaned,
    validate_raw,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_df():
    """A tiny valid dataframe for basic tests."""
    return pd.DataFrame(
        {
            "Datetime": [
                "2024-01-01 01:00",
                "2024-01-01 02:00",
                "2024-01-01 03:00",
                "2024-01-01 04:00",
            ],
            "PJMW_MW": [5000.0, 5100.0, 5200.0, 5300.0],
        }
    )


@pytest.fixture
def df_with_duplicates():
    """Dataframe containing duplicate timestamps (simulates DST fall-back)."""
    return pd.DataFrame(
        {
            "Datetime": [
                "2024-01-01 01:00",
                "2024-01-01 02:00",
                "2024-01-01 02:00",
                "2024-01-01 03:00",
            ],
            "PJMW_MW": [5000.0, 5100.0, 5300.0, 5400.0],
        }
    )


@pytest.fixture
def df_with_gap():
    """Dataframe with a 2-hour gap to test reindexing + interpolation."""
    return pd.DataFrame(
        {
            "Datetime": [
                "2024-01-01 01:00",
                "2024-01-01 04:00",
            ],
            "PJMW_MW": [100.0, 400.0],
        }
    )


@pytest.fixture
def df_with_outlier():
    """Dataframe containing one extreme low outlier."""
    values = [5000.0] * 100
    values[50] = 10.0  # extreme outlier
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    return pd.DataFrame(
        {
            "Datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "PJMW_MW": values,
        }
    )


# ---------------------------------------------------------------------------
# validate_raw tests
# ---------------------------------------------------------------------------
class TestValidateRaw:
    def test_valid_input(self, simple_df):
        validate_raw(simple_df)  # should not raise

    def test_missing_datetime_column(self):
        df = pd.DataFrame({"Date": ["2024-01-01"], "PJMW_MW": [5000.0]})
        with pytest.raises(AssertionError, match="Missing 'Datetime'"):
            validate_raw(df)

    def test_missing_mw_column(self):
        df = pd.DataFrame({"Datetime": ["2024-01-01"], "Value": [5000.0]})
        with pytest.raises(AssertionError, match="Missing 'PJMW_MW'"):
            validate_raw(df)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"Datetime": [], "PJMW_MW": []})
        with pytest.raises(AssertionError, match="empty"):
            validate_raw(df)

    def test_all_nan_values(self):
        df = pd.DataFrame({"Datetime": ["2024-01-01"], "PJMW_MW": [np.nan]})
        with pytest.raises(AssertionError, match="NaN"):
            validate_raw(df)


# ---------------------------------------------------------------------------
# clean_data tests
# ---------------------------------------------------------------------------
class TestCleanData:
    def test_basic_clean(self, simple_df):
        result, summary = clean_data(simple_df)
        assert len(result) == 4
        assert result["PJMW_MW"].isnull().sum() == 0
        assert result.index.is_monotonic_increasing
        assert summary["final_rows"] == 4

    def test_duplicates_resolved(self, df_with_duplicates):
        result, summary = clean_data(df_with_duplicates)
        assert not result.index.duplicated().any()
        assert summary["duplicate_rows"] == 2
        # Duplicate values (5100 + 5300) / 2 = 5200
        assert result.loc["2024-01-01 02:00", "PJMW_MW"] == 5200.0

    def test_gaps_filled(self, df_with_gap):
        result, summary = clean_data(df_with_gap)
        assert len(result) == 4  # 01:00, 02:00, 03:00, 04:00
        assert result["PJMW_MW"].isnull().sum() == 0
        assert summary["gaps_filled"] == 2
        # Linear interpolation: 100, 200, 300, 400
        assert result.loc["2024-01-01 02:00", "PJMW_MW"] == pytest.approx(200.0)
        assert result.loc["2024-01-01 03:00", "PJMW_MW"] == pytest.approx(300.0)

    def test_outlier_replaced(self, df_with_outlier):
        result, summary = clean_data(df_with_outlier)
        assert summary["outliers_replaced"] >= 1
        # The extreme value (10.0) should have been interpolated away
        assert result["PJMW_MW"].min() > 10.0

    def test_summary_keys(self, simple_df):
        _, summary = clean_data(simple_df)
        expected_keys = {
            "raw_rows",
            "duplicate_rows",
            "outliers_replaced",
            "expected_records",
            "actual_records_before_reindex",
            "gaps_filled",
            "missing_timestamps",
            "nulls_after_interpolation",
            "final_rows",
        }
        assert expected_keys.issubset(summary.keys())

    def test_chronological_order(self):
        """Input in reverse order should still produce sorted output."""
        df = pd.DataFrame(
            {
                "Datetime": [
                    "2024-01-01 04:00",
                    "2024-01-01 01:00",
                    "2024-01-01 03:00",
                    "2024-01-01 02:00",
                ],
                "PJMW_MW": [5300.0, 5000.0, 5200.0, 5100.0],
            }
        )
        result, _ = clean_data(df)
        assert result.index.is_monotonic_increasing
        assert result.iloc[0]["PJMW_MW"] == 5000.0


# ---------------------------------------------------------------------------
# detect_outliers tests
# ---------------------------------------------------------------------------
class TestDetectOutliers:
    def test_no_outliers(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="h")
        df = pd.DataFrame({"PJMW_MW": np.random.normal(5000, 100, 100)}, index=idx)
        result, count = detect_outliers(df.copy())
        assert count == 0

    def test_flagged_outlier_becomes_nan(self):
        idx = pd.date_range("2024-01-01", periods=50, freq="h")
        values = [5000.0] * 50
        values[25] = 1.0  # extreme low
        df = pd.DataFrame({"PJMW_MW": values}, index=idx)
        result, count = detect_outliers(df.copy())
        assert count >= 1
        assert pd.isna(result.iloc[25]["PJMW_MW"])


# ---------------------------------------------------------------------------
# audit_gaps tests
# ---------------------------------------------------------------------------
class TestAuditGaps:
    def test_no_gaps(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame({"PJMW_MW": range(5)}, index=idx)
        full_range = pd.date_range(start=idx.min(), end=idx.max(), freq="h")
        missing = audit_gaps(df, full_range)
        assert len(missing) == 0

    def test_identifies_gaps(self):
        idx = pd.DatetimeIndex(["2024-01-01 01:00", "2024-01-01 04:00"])
        df = pd.DataFrame({"PJMW_MW": [100, 400]}, index=idx)
        full_range = pd.date_range(start=idx.min(), end=idx.max(), freq="h")
        missing = audit_gaps(df, full_range)
        assert len(missing) == 2  # 02:00 and 03:00
