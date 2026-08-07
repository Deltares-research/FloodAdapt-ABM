"""
test_income_marginal.py
=======================
Tests for calibrating the **marginal** income distribution from published
ACS county totals, the axis of the income model that needs no building
locations (see ``income_utils`` module docstring).

Everything here is offline: the pure functions are exercised directly and
the two network helpers are driven through a monkeypatched transport, so
the suite never touches the Census API.
"""
from __future__ import annotations

import numpy as np
import pytest

from floodadapt_abm import income_utils
from floodadapt_abm.income_utils import (
    ACS_B19001_EDGES,
    ACS_B19001_LABELS,
    bracket_fit_distance,
    fetch_acs_county_income,
    fetch_acs_income_brackets,
    lognormal_bracket_shares,
)

# Charleston County, SC, ACS 2020-2024 5-year (verified against the live API).
CHARLESTON_MEDIAN = 88_494.0
CHARLESTON_AGGREGATE = 23_566_353_400.0
CHARLESTON_HOUSEHOLDS = 178_975.0
CHARLESTON_RATIO = CHARLESTON_AGGREGATE / CHARLESTON_HOUSEHOLDS / CHARLESTON_MEDIAN


# ---------------------------------------------------------------------------
# lognormal_bracket_shares — pure, deterministic
# ---------------------------------------------------------------------------
def test_bracket_shares_are_a_probability_distribution():
    """Edges spanning [0, inf) must yield shares summing to exactly 1."""
    shares = lognormal_bracket_shares(CHARLESTON_MEDIAN, CHARLESTON_RATIO)
    assert shares.shape == (len(ACS_B19001_EDGES) - 1,)
    assert shares.shape == (len(ACS_B19001_LABELS),)
    assert np.all(shares >= 0.0)
    assert shares.sum() == pytest.approx(1.0, abs=1e-12)


def test_bracket_shares_split_at_the_median():
    """Half the mass must fall below the median, by definition."""
    median = CHARLESTON_MEDIAN
    shares = lognormal_bracket_shares(
        median, CHARLESTON_RATIO, edges=(0.0, median, float("inf"))
    )
    assert shares[0] == pytest.approx(0.5, abs=1e-9)
    assert shares[1] == pytest.approx(0.5, abs=1e-9)


def test_wider_ratio_puts_more_mass_in_both_tails():
    """A larger mean/median ratio means a larger sd, so fatter tails.

    This is the property that made the Charleston recalibration matter:
    the ratio, not the median, controls dispersion.
    """
    edges = (0.0, 25_000.0, 200_000.0, float("inf"))
    narrow = lognormal_bracket_shares(CHARLESTON_MEDIAN, 1.15, edges=edges)
    wide = lognormal_bracket_shares(CHARLESTON_MEDIAN, CHARLESTON_RATIO, edges=edges)
    assert wide[0] > narrow[0]      # more poor households
    assert wide[2] > narrow[2]      # more rich households
    assert wide[1] < narrow[1]      # fewer in the middle


def test_bracket_shares_reject_impossible_parameters():
    """A lognormal always has mean > median, so ratio <= 1 has no solution."""
    with pytest.raises(ValueError, match="mean_median_ratio"):
        lognormal_bracket_shares(CHARLESTON_MEDIAN, 1.0)
    with pytest.raises(ValueError, match="mean_median_ratio"):
        lognormal_bracket_shares(CHARLESTON_MEDIAN, 0.8)
    with pytest.raises(ValueError, match="median_income"):
        lognormal_bracket_shares(0.0, 1.5)


def test_bracket_shares_are_deterministic():
    """No RNG anywhere: repeated calls must be bit-identical."""
    a = lognormal_bracket_shares(CHARLESTON_MEDIAN, CHARLESTON_RATIO)
    b = lognormal_bracket_shares(CHARLESTON_MEDIAN, CHARLESTON_RATIO)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# bracket_fit_distance
# ---------------------------------------------------------------------------
def test_fit_distance_endpoints():
    """0 for identical vectors, 1 for disjoint support."""
    p = np.array([0.25, 0.25, 0.5])
    assert bracket_fit_distance(p, p) == pytest.approx(0.0)
    assert bracket_fit_distance(
        np.array([1.0, 0.0]), np.array([0.0, 1.0])
    ) == pytest.approx(1.0)


def test_fit_distance_reads_as_misplaced_fraction():
    """Moving 10% of mass between brackets must give a distance of 0.10."""
    obs = np.array([0.5, 0.5])
    pred = np.array([0.6, 0.4])
    assert bracket_fit_distance(obs, pred) == pytest.approx(0.10)


def test_fit_distance_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        bracket_fit_distance(np.zeros(3), np.zeros(4))


def test_calibrated_charleston_beats_generic_defaults():
    """The substantive result: measured parameters fit far better.

    Observed shares are the real ACS 2020-2024 B19001 histogram for
    Charleston County; the assertion pins the direction and rough magnitude
    of the improvement, not the exact figure.
    """
    observed = np.array([
        0.054, 0.027, 0.025, 0.032, 0.026, 0.029, 0.034, 0.031,
        0.032, 0.065, 0.085, 0.113, 0.098, 0.082, 0.095, 0.172,
    ])
    observed = observed / observed.sum()
    generic = lognormal_bracket_shares(70_000.0, 1.15)
    calibrated = lognormal_bracket_shares(CHARLESTON_MEDIAN, CHARLESTON_RATIO)

    d_generic = bracket_fit_distance(observed, generic)
    d_calibrated = bracket_fit_distance(observed, calibrated)
    assert d_calibrated < d_generic / 3.0
    assert d_calibrated < 0.10


# ---------------------------------------------------------------------------
# Network helpers, driven through a monkeypatched transport
# ---------------------------------------------------------------------------
def _fake_acs(payload_by_marker: dict[str, list[list[str]]]):
    """Build an ``_acs_get_json`` stub that dispatches on the URL."""

    def _stub(url: str, timeout: float = 120.0) -> list[list[str]]:
        for marker, payload in payload_by_marker.items():
            if marker in url:
                return payload
        raise AssertionError(f"unexpected URL: {url}")

    return _stub


def test_fetch_county_income_computes_the_mean_from_aggregate(monkeypatch):
    """ACS publishes no mean, so it must come from aggregate/households."""
    payload = [
        ["NAME", "B19013_001E", "B19025_001E", "B11001_001E",
         "state", "county"],
        ["Charleston County, South Carolina",
         str(int(CHARLESTON_MEDIAN)),
         str(int(CHARLESTON_AGGREGATE)),
         str(int(CHARLESTON_HOUSEHOLDS)),
         "45", "019"],
    ]
    monkeypatch.setattr(
        income_utils, "_acs_get_json", _fake_acs({"B19013": payload})
    )
    stats = fetch_acs_county_income("45", "019", api_key="dummy")

    expected_mean = CHARLESTON_AGGREGATE / CHARLESTON_HOUSEHOLDS
    assert stats["median_income"] == pytest.approx(CHARLESTON_MEDIAN)
    assert stats["mean_income"] == pytest.approx(expected_mean)
    assert stats["mean_median_ratio"] == pytest.approx(
        expected_mean / CHARLESTON_MEDIAN
    )
    assert stats["n_households"] == pytest.approx(CHARLESTON_HOUSEHOLDS)
    # The ratio must be usable straight away by the fitter.
    assert lognormal_bracket_shares(
        stats["median_income"], stats["mean_median_ratio"]
    ).sum() == pytest.approx(1.0)


def test_fetch_income_brackets_normalises_counts(monkeypatch):
    counts = list(range(2, 18))          # 16 arbitrary but distinct counts
    header = ["NAME"] + [f"B19001_{i:03d}E" for i in range(2, 18)]
    row = ["Charleston County"] + [str(c) for c in counts]
    monkeypatch.setattr(
        income_utils, "_acs_get_json", _fake_acs({"B19001": [header, row]})
    )
    brackets = fetch_acs_income_brackets("45", "019", api_key="dummy")

    assert brackets["counts"].shape == (16,)
    assert np.array_equal(brackets["counts"], np.array(counts, dtype=float))
    assert brackets["shares"].sum() == pytest.approx(1.0)
    # Shares must line up with the published bracket labels.
    assert brackets["shares"].shape == (len(ACS_B19001_LABELS),)


def test_missing_api_key_raises_before_any_request(monkeypatch):
    """No key must fail fast with instructions, not a network error."""
    monkeypatch.delenv("CENSUS_API_KEY", raising=False)
    monkeypatch.setattr(
        income_utils,
        "_acs_get_json",
        lambda *a, **k: pytest.fail("network must not be touched"),
    )
    with pytest.raises(RuntimeError, match="key_signup"):
        fetch_acs_county_income("45", "019")
    with pytest.raises(RuntimeError, match="CENSUS_API_KEY"):
        fetch_acs_income_brackets("45", "019")


def test_api_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("CENSUS_API_KEY", "from-env")
    seen: list[str] = []

    def _stub(url: str, timeout: float = 120.0):
        seen.append(url)
        return [
            ["NAME", "B19013_001E", "B19025_001E", "B11001_001E"],
            ["x", "84320", "22267010600", "175499"],
        ]

    monkeypatch.setattr(income_utils, "_acs_get_json", _stub)
    fetch_acs_county_income("45", "019")
    assert "key=from-env" in seen[0]


def test_html_response_is_reported_as_a_key_problem(monkeypatch):
    """The API answers a bad key with HTTP 200 + HTML, not an error status.

    Without this check the caller sees a JSONDecodeError pointing at line 1
    column 1, which reads like a broken URL rather than a missing key.
    """

    class _Resp:
        headers = {"Content-Type": "text/html;charset=UTF-8"}

        def read(self):
            return b"<html>\n<head><title>Missing Key</title></head></html>"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    with pytest.raises(RuntimeError, match="API key is missing or invalid"):
        income_utils._acs_get_json("https://api.census.gov/data/whatever")
