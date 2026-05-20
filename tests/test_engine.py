"""Tests for the data-quality + anomaly-detection engine.

These tests exercise the three top-level functions of ``anomaly_engine``:

* :func:`generate_dirty_data` — deterministic synthetic transactions with a
  documented ~4% true-anomaly rate that the engine must recover.
* :func:`run_data_quality_checks` — rule-based DQ checks (null, negative
  amount, schema) on hand-crafted fixtures.
* :func:`detect_anomalies` — Isolation Forest training, prediction, and
  the recall floor against the known-anomaly fixture (the Day-8 brief
  literally asks for "assert detection rate, recall floor").

Heavy deps (scikit-learn) are gated with ``pytest.importorskip`` so the
file is safe to keep checked in even when sklearn isn't installed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from anomaly_engine import (  # noqa: E402
    detect_anomalies,
    generate_dirty_data,
    run_data_quality_checks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SMALL_N = 2000


@pytest.fixture(scope="module")
def dirty_df() -> pd.DataFrame:
    """Deterministic dirty data with ~4% injected anomalies."""
    return generate_dirty_data(n_samples=SMALL_N)


# ---------------------------------------------------------------------------
# generate_dirty_data — shape + injected anomaly bookkeeping
# ---------------------------------------------------------------------------


EXPECTED_COLUMNS = {
    "transaction_id", "user_id", "amount", "account_age_days",
    "is_true_anomaly",
}


def test_generate_dirty_data_shape(dirty_df: pd.DataFrame) -> None:
    """Row count matches the request and the column set matches the spec."""
    assert len(dirty_df) == SMALL_N
    assert set(dirty_df.columns) == EXPECTED_COLUMNS
    assert not dirty_df.isna().any().any()


def test_generate_dirty_data_is_deterministic() -> None:
    """The RNG seed (42) means two calls produce identical frames."""
    a = generate_dirty_data(n_samples=500)
    b = generate_dirty_data(n_samples=500)
    pd.testing.assert_frame_equal(a, b)


def test_injected_anomaly_rate(dirty_df: pd.DataFrame) -> None:
    """Roughly 4% of rows are flagged as true anomalies (per script docstring)."""
    rate = float(dirty_df["is_true_anomaly"].mean())
    # Allow a generous +/- 2pp tolerance: the script picks exactly 4% but
    # individual buckets can shift the proportion a tiny amount.
    assert 0.02 <= rate <= 0.06, f"true-anomaly rate {rate:.3%} outside [2%, 6%]"


def test_injected_anomaly_types_show_up(dirty_df: pd.DataFrame) -> None:
    """The three documented anomaly types (fat-finger, negative, fraud) are
    actually present in the dirty frame."""
    anomalies = dirty_df[dirty_df["is_true_anomaly"] == 1]

    # Fat-finger: at least one extreme high amount.
    assert (anomalies["amount"] > 50_000).any(), "no fat-finger anomalies"
    # Negative amounts: at least one anomaly row has a negative amount.
    assert (anomalies["amount"] < 0).any(), "no negative-amount anomalies"
    # Fraud-like: at least one anomaly is on a brand-new account (< 5 days).
    assert (anomalies["account_age_days"] < 5).any(), "no new-account anomalies"


# ---------------------------------------------------------------------------
# run_data_quality_checks — rule-based DQ on hand-crafted fixtures
# ---------------------------------------------------------------------------


def _clean_df() -> pd.DataFrame:
    """A handful of rows the DQ checks should flag as clean."""
    return pd.DataFrame(
        {
            "transaction_id": np.array([1, 2, 3], dtype=np.int64),
            "user_id": np.array([10, 11, 12], dtype=np.int64),
            "amount": np.array([1.5, 2.5, 3.5], dtype=np.float64),
            "account_age_days": np.array([100, 200, 300], dtype=np.int64),
        }
    )


def test_dq_clean_frame_has_no_issues() -> None:
    issues = run_data_quality_checks(_clean_df())
    assert issues == [], issues


def test_dq_detects_negative_amounts() -> None:
    df = _clean_df()
    df.loc[1, "amount"] = -5.0
    issues = run_data_quality_checks(df)
    assert any("negative" in i for i in issues), issues


def test_dq_detects_null_values() -> None:
    df = _clean_df()
    df.loc[0, "amount"] = np.nan
    issues = run_data_quality_checks(df)
    assert any("null" in i.lower() for i in issues), issues


# ---------------------------------------------------------------------------
# detect_anomalies — Isolation Forest detection rate + recall floor
# ---------------------------------------------------------------------------


def _scored(dirty_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    df = dirty_df.copy()
    return detect_anomalies(df)


def test_detect_anomalies_adds_detection_column(dirty_df: pd.DataFrame) -> None:
    scored, _ = _scored(dirty_df)
    assert "is_detected_anomaly" in scored.columns
    assert set(scored["is_detected_anomaly"].unique()).issubset({0, 1})


def test_detection_rate_within_expected_band(dirty_df: pd.DataFrame) -> None:
    """The Isolation Forest is configured at contamination=0.04, so the
    detected anomaly rate should land in a tight band around 4%."""
    scored, _ = _scored(dirty_df)
    rate = float(scored["is_detected_anomaly"].mean())
    assert 0.025 <= rate <= 0.055, f"detection rate {rate:.3%} outside [2.5%, 5.5%]"


def test_recall_floor_against_known_anomalies(dirty_df: pd.DataFrame) -> None:
    """Recall against the injected anomalies must beat the smoke floor.

    The script documents that pipeline-error reduction is 94% — that
    relies on the detector catching the majority of true anomalies.  We
    set a conservative floor of 0.50 so this is a meaningful regression
    guard (not a flaky tight floor) on the fixture size."""
    scored, _ = _scored(dirty_df)
    tp = int(((scored["is_detected_anomaly"] == 1) & (scored["is_true_anomaly"] == 1)).sum())
    total_true = int(scored["is_true_anomaly"].sum())
    assert total_true > 0
    recall = tp / total_true
    assert recall >= 0.50, f"recall floor breached: {recall:.3f}"


def test_error_reduction_returned(dirty_df: pd.DataFrame) -> None:
    """``detect_anomalies`` returns a non-negative error-reduction metric."""
    _, error_reduction = _scored(dirty_df)
    assert 0.0 <= error_reduction <= 1.0, error_reduction
