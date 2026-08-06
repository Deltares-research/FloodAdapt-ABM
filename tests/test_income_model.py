"""
test_income_model.py
====================
Tests for the synthetic lognormal income/wealth port (native DYNAMO-M
pipeline), income percentiles, the fixed adaptation cost, and —
critically — that the affordability gate genuinely binds for part of the
population (under legacy defaults it is algebraically inert).
"""
from __future__ import annotations

import numpy as np
import pytest

from floodadapt_abm import CouplingConfig, SimulationEngine
from floodadapt_abm.income_utils import (
    percentiles_from_income_values,
    percentiles_from_value_proxy,
)
from tests.conftest import make_mock_dataset


def _synthetic_cfg(**overrides) -> CouplingConfig:
    cfg = CouplingConfig.legacy()
    cfg.decision.income_mode = "synthetic_lognormal"
    for key, value in overrides.items():
        setattr(cfg.decision, key, value)
    return cfg


def _engine(cfg: CouplingConfig, **kwargs) -> SimulationEngine:
    return SimulationEngine(
        ds=make_mock_dataset(n_objects=200), config=cfg, **kwargs
    )


def test_synthetic_income_reproducible_from_seed():
    a = _engine(_synthetic_cfg())
    b = _engine(_synthetic_cfg())
    assert np.array_equal(a._data.income, b._data.income)
    assert np.array_equal(a._data.wealth, b._data.wealth)
    assert np.array_equal(a._data.income_percentile, b._data.income_percentile)


def test_synthetic_income_independent_of_building_value():
    """The degeneracy fix: income must not be a function of max_pot_dmg."""
    # Heterogeneous building values (the default mock table is constant).
    ds = make_mock_dataset(n_objects=200)
    rng = np.random.default_rng(1)
    ds["object_id"].attrs["max_pot_dmg"] = rng.uniform(
        100_000.0, 300_000.0, 200
    )
    engine = SimulationEngine(ds=ds, config=_synthetic_cfg())
    income = engine._data.income.astype(np.float64)
    mpd = engine.max_pot_dmg.astype(np.float64)
    corr = np.corrcoef(income, mpd)[0, 1]
    assert abs(corr) < 0.2  # uncorrelated (uniform percentiles vs table values)
    # And wealth is no longer identically max_pot_dmg.
    assert not np.allclose(engine._data.wealth, mpd)
    # Legacy control on the same table: income IS max_pot_dmg / 4.14.
    legacy = SimulationEngine(ds=ds, config=CouplingConfig.legacy())
    legacy_corr = np.corrcoef(
        legacy._data.income.astype(np.float64), mpd
    )[0, 1]
    assert legacy_corr > 0.999


def test_wealth_follows_native_percentile_ratio_table():
    """wealth = interp(pct, [0..100], [0,1.06,4.14,4.19,5.24,6]) * income."""
    pct = np.array([20, 40, 60, 80], dtype=np.int64)
    cfg = _synthetic_cfg()
    ds = make_mock_dataset(n_objects=5)  # 4 residential @ 80 %
    engine = SimulationEngine(ds=ds, config=cfg, income_percentile_per_agent=pct)

    expected_ratio = np.interp(
        pct, [0, 20, 40, 60, 80, 100], [0.0, 1.06, 4.14, 4.19, 5.24, 6.0]
    )
    np.testing.assert_allclose(
        engine._data.wealth,
        (expected_ratio * engine._data.income).astype(np.float32),
    )
    # Richer percentiles earn more (lognormal is monotone in percentile).
    assert np.all(np.diff(engine._data.income) >= 0)


def test_explicit_percentiles_respected_and_clipped():
    ds = make_mock_dataset(n_objects=5)
    pct = np.array([0, 150, 50, 99])  # out-of-range values get clipped
    engine = SimulationEngine(
        ds=ds, config=_synthetic_cfg(), income_percentile_per_agent=pct
    )
    assert np.array_equal(engine._data.income_percentile, [1, 99, 50, 99])


def test_fixed_adaptation_cost_constant_across_agents():
    cfg = _synthetic_cfg(adaptation_total_cost=10_800.0)
    engine = _engine(cfg)
    annual = engine._annual_adapt_cost
    # One value for everyone (native country-scaled constant), annuitised.
    assert np.allclose(annual, annual[0])
    r, lp = 0.04, 16
    annuity = r * (1 + r) ** lp / ((1 + r) ** lp - 1)
    assert np.isclose(annual[0], 10_800.0 * annuity, rtol=1e-6)


def test_affordability_gate_binds_partially_under_new_economics():
    """
    Regression: with synthetic incomes + fixed cost, the fraction of
    households failing ``income * expenditure_cap <= annual_cost`` must be
    strictly between 0 and 1.  Under the legacy economics the ratio is a
    population-wide constant (1.689) and the fraction is exactly 0.
    """
    # Cost calibrated so the 6 % expenditure cap bisects the income
    # distribution: annual cost ~6.9k -> constrained iff income <= ~114k,
    # ~80 % of a lognormal with median 70k.  (At the DYNAMO-M-anchored
    # default of 10.8k total the gate binds only for the poorest ~0.2 % —
    # heterogeneous but rarely visible at n=160.)
    cfg = _synthetic_cfg(adaptation_total_cost=80_000.0)
    engine = _engine(cfg)
    d = engine._data
    constrained = d.income * cfg.decision.expenditure_cap <= engine._annual_adapt_cost
    fraction = constrained.mean()
    assert 0.0 < fraction < 1.0, (
        f"affordability gate should bind heterogeneously, got {fraction:.3f}"
    )

    # Legacy control: gate never binds (mpd cancels; ratio 1.689 for all).
    legacy = _engine(CouplingConfig.legacy())
    legacy_constrained = (
        legacy._data.income * 0.06 <= legacy._annual_adapt_cost
    ) & (legacy.max_pot_dmg > 0)
    assert legacy_constrained.sum() == 0


def test_legacy_income_mode_unchanged():
    """mpd_ratio fallback still reproduces the historical relation exactly."""
    engine = _engine(CouplingConfig.legacy())
    np.testing.assert_allclose(
        engine._data.income * 4.14, engine.max_pot_dmg, rtol=1e-5
    )
    np.testing.assert_allclose(
        engine._data.wealth, engine.max_pot_dmg, rtol=1e-5
    )
    assert engine._data.income_percentile is None


def test_explicit_income_and_wealth_arrays_override():
    ds = make_mock_dataset(n_objects=5)
    income = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    wealth = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    engine = SimulationEngine(
        ds=ds, config=_synthetic_cfg(),
        income_per_agent=income, wealth_per_agent=wealth,
    )
    np.testing.assert_array_equal(engine._data.income, income)
    np.testing.assert_array_equal(engine._data.wealth, wealth)


def test_unknown_income_mode_raises():
    cfg = CouplingConfig.legacy()
    cfg.decision.income_mode = "bogus"
    with pytest.raises(ValueError, match="income_mode"):
        _engine(cfg)


# ---------------------------------------------------------------------------
# income_utils: percentile derivation for a case study
# ---------------------------------------------------------------------------

def test_percentiles_from_income_values_rank_order():
    values = np.array([30_000.0, 90_000.0, 60_000.0, np.nan])
    pct = percentiles_from_income_values(values, jitter=0)
    assert pct[0] < pct[2] < pct[1]        # rank order preserved
    assert pct[3] == 50                    # NaN -> median percentile
    assert pct.min() >= 1 and pct.max() <= 99


def test_percentiles_jitter_reproducible_and_bounded():
    values = np.full(50, 60_000.0)
    a = percentiles_from_income_values(values, jitter=5, seed=3)
    b = percentiles_from_income_values(values, jitter=5, seed=3)
    assert np.array_equal(a, b)
    assert len(np.unique(a)) > 1           # ties broken within a block group
    assert np.all((a >= 45) & (a <= 55))   # +/- 5 around the tied percentile


# ---------------------------------------------------------------------------
# Value-proxy percentiles (no-geometry fallback, Gaussian copula)
# ---------------------------------------------------------------------------

def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation via Pearson on the rank vectors."""
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def test_value_proxy_achieves_target_rank_correlation():
    rng = np.random.default_rng(0)
    values = rng.lognormal(12.0, 0.8, 5_000)
    pct = percentiles_from_value_proxy(values, rank_correlation=0.5, seed=1)
    assert abs(_spearman(values, pct) - 0.5) < 0.1


def test_value_proxy_reproducible_and_bounded():
    rng = np.random.default_rng(2)
    values = rng.uniform(50_000.0, 500_000.0, 1_000)
    a = percentiles_from_value_proxy(values, rank_correlation=0.5, seed=7)
    b = percentiles_from_value_proxy(values, rank_correlation=0.5, seed=7)
    assert np.array_equal(a, b)
    assert a.dtype == np.int64
    assert a.min() >= 1 and a.max() <= 99


def test_value_proxy_rho_zero_is_independent():
    rng = np.random.default_rng(3)
    values = rng.uniform(50_000.0, 500_000.0, 5_000)
    pct = percentiles_from_value_proxy(values, rank_correlation=0.0, seed=4)
    assert abs(_spearman(values, pct)) < 0.1


def test_value_proxy_rho_one_is_monotone():
    rng = np.random.default_rng(5)
    values = rng.uniform(50_000.0, 500_000.0, 500)
    pct = percentiles_from_value_proxy(values, rank_correlation=1.0, seed=6)
    order = np.argsort(values)
    assert np.all(np.diff(pct[order]) >= 0)   # non-decreasing in value


def test_value_proxy_invalid_rho_raises():
    with pytest.raises(ValueError, match="rank_correlation"):
        percentiles_from_value_proxy(np.ones(10), rank_correlation=1.5)
