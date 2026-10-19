"""
test_analysis.py
================
Tests for :mod:`floodadapt_abm.analysis`: the stock-versus-cumulative uptake
helpers, the damage / cost aggregates, the summary tables and the insurance
affordability diagnostics.  Everything runs on the mock dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from floodadapt_abm import CouplingConfig, SimulationEngine, ThresholdRule
from floodadapt_abm import analysis as an
from tests.conftest import historical_modes_config, make_mock_dataset

SLR = np.linspace(0.0, 1.0, 8)
SEED = 123
NO_SEQ = 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _run(config: CouplingConfig, rule=None, ds=None) -> dict:
    """Run the mock table under ``config`` and return the results dict."""
    ds = make_mock_dataset() if ds is None else ds
    engine = SimulationEngine(ds=ds, config=config, decision_rule=rule)
    return engine.run(SLR, no_seq=NO_SEQ, seed=SEED)


@pytest.fixture(scope="module")
def adapt_results() -> dict:
    """Threshold-rule run with permanent floodproofing (no lifespan reset)."""
    cfg = historical_modes_config(lifespan_dryproof=None)
    return _run(cfg, ThresholdRule(cfg.decision, damage_threshold=0.05))


@pytest.fixture(scope="module")
def baseline_results() -> dict:
    """Never-adapt baseline on the same event draws as ``adapt_results``."""
    cfg = historical_modes_config(lifespan_dryproof=None)
    return _run(cfg, ThresholdRule(cfg.decision, damage_threshold=1.01))


@pytest.fixture(scope="module")
def insurance_results() -> dict:
    """Insurance-enabled SEU run (synthetic incomes, community pricing)."""
    cfg = historical_modes_config(include_insurance=True)
    return _run(cfg)


# ---------------------------------------------------------------------------
# Uptake views
# ---------------------------------------------------------------------------

def test_current_uptake_equals_adoption_fraction(adapt_results):
    np.testing.assert_array_equal(
        an.current_uptake(adapt_results["adapted_history"]),
        adapt_results["adoption_fraction"],
    )


def test_cumulative_uptake_shape_bounds_and_monotone(adapt_results):
    cum = an.cumulative_uptake(adapt_results["adapted_history"])
    no_seq, _, n_years = adapt_results["adapted_history"].shape
    assert cum.shape == (no_seq, n_years)
    assert cum.dtype == np.float64
    assert (cum >= 0).all() and (cum <= 1).all()
    assert (np.diff(cum, axis=1) >= -1e-15).all()


def test_cumulative_equals_stock_without_lifespan_reset(adapt_results):
    """With ``lifespan_dryproof=None`` floodproofing never clears."""
    np.testing.assert_array_equal(
        an.cumulative_uptake(adapt_results["adapted_history"]),
        adapt_results["adoption_fraction"],
    )


def test_cumulative_exceeds_stock_when_status_clears():
    """A 1 -> 0 transition keeps the agent in the cumulative count."""
    history = np.zeros((1, 4, 5), dtype=bool)
    history[0, 0, 1] = True            # agent 0 holds the status in year 1 only
    history[0, 1, 2:] = True           # agent 1 from year 2 onwards
    stock = an.current_uptake(history)
    cum = an.cumulative_uptake(history)
    np.testing.assert_allclose(stock[0], [0, 0.25, 0.25, 0.25, 0.25])
    np.testing.assert_allclose(cum[0], [0, 0.25, 0.5, 0.5, 0.5])
    assert (cum >= stock).all()


def test_insured_cumulative_never_below_insured_fraction(insurance_results):
    insured = insurance_results["insured_history"]
    cum = an.cumulative_uptake(insured)
    assert (cum >= insurance_results["insured_fraction"] - 1e-15).all()
    dropped = (insured[:, :, :-1] & ~insured[:, :, 1:]).any()
    if dropped:
        assert (cum > insurance_results["insured_fraction"]).any()


def test_count_history_and_statistics(adapt_results):
    counts = an.count_history(adapt_results["adapted_history"])
    assert counts.dtype.kind == "i"
    n_agents = adapt_results["adapted_history"].shape[1]
    np.testing.assert_allclose(
        counts / n_agents, adapt_results["adoption_fraction"]
    )
    stats = an.uptake_statistics(adapt_results["adapted_history"])
    assert set(stats) == {"final_current", "final_cumulative", "peak_current"}
    assert stats["final_cumulative"] >= stats["final_current"]
    assert stats["peak_current"] >= stats["final_current"]


def test_history_validation_rejects_wrong_rank():
    with pytest.raises(ValueError, match="no_seq, n_agents, n_years"):
        an.cumulative_uptake(np.zeros((3, 4), dtype=bool))


# ---------------------------------------------------------------------------
# Damage and cost aggregates
# ---------------------------------------------------------------------------

def test_damage_per_year_matches_notebook_formula(adapt_results):
    expected = adapt_results["damage_history"].astype(np.float64).sum(axis=1) / 1e6
    np.testing.assert_array_equal(
        an.damage_per_year(adapt_results["damage_history"]), expected
    )
    cum = an.cumulative_damage(adapt_results["damage_history"])
    np.testing.assert_allclose(cum[:, -1], expected.sum(axis=1))


def test_avoided_damage_clipped_and_cumulative(adapt_results, baseline_results):
    avoided = an.avoided_damage_per_year(
        adapt_results["damage_history"], baseline_results["damage_history"]
    )
    assert (avoided >= 0).all()
    base = baseline_results["damage_history"].astype(np.float64)
    run = adapt_results["damage_history"].astype(np.float64)
    expected = np.clip(base - run, 0, None).sum(axis=1) / 1e6
    np.testing.assert_array_equal(avoided, expected)
    cum = an.cumulative_avoided_damage(
        adapt_results["damage_history"], baseline_results["damage_history"]
    )
    np.testing.assert_allclose(cum[:, -1], avoided.sum(axis=1))
    unclipped = an.avoided_damage_per_year(
        adapt_results["damage_history"],
        baseline_results["damage_history"],
        clip=False,
    )
    assert (unclipped <= avoided + 1e-12).all()


def test_household_cost_requires_insurance(adapt_results, insurance_results):
    with pytest.raises(KeyError, match="include_insurance"):
        an.household_cost_per_year(adapt_results)
    cost = an.household_cost_per_year(insurance_results)
    expected = (
        insurance_results["out_of_pocket_history"].astype(np.float64)
        + insurance_results["premium_paid_history"].astype(np.float64)
    ).sum(axis=1) / 1e6
    np.testing.assert_allclose(cost, expected)
    cum = an.cumulative_household_cost(insurance_results)
    np.testing.assert_allclose(cum[:, -1], cost.sum(axis=1))


def test_mean_and_std_and_money_unit():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    mean, std = an.mean_and_std(arr)
    np.testing.assert_allclose(mean, [2.0, 3.0])
    np.testing.assert_allclose(std, [1.0, 1.0])
    assert an.money_unit(1e6) == "M$"
    assert an.money_unit(1.0) == "$"
    assert an.money_unit(2.5) == "$/2.5"


def test_functions_do_not_mutate_inputs(adapt_results, baseline_results):
    before = {k: v.copy() for k, v in adapt_results.items()}
    an.create_summary_table(adapt_results, baseline_results)
    an.cumulative_uptake(adapt_results["adapted_history"])
    an.scenario_summary({"a": adapt_results}, reference=baseline_results)
    for key, value in before.items():
        np.testing.assert_array_equal(adapt_results[key], value)


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def test_summary_table_layout(adapt_results, baseline_results):
    table = an.create_summary_table(adapt_results, baseline_results)
    assert isinstance(table, pd.DataFrame)
    assert len(table) == NO_SEQ + 2
    assert list(table["sequence"].iloc[-2:]) == ["MEAN", "STD"]
    expected_cols = {
        "sequence",
        "final floodproofed (#)",
        "final floodproofed (%)",
        "ever floodproofed (%)",
        "cum. damage (M$)",
        "cum. avoided (M$)",
    }
    assert set(table.columns) == expected_cols
    per_seq = table.iloc[:NO_SEQ]
    mean_row = table.iloc[-2]
    np.testing.assert_allclose(
        mean_row["cum. damage (M$)"], per_seq["cum. damage (M$)"].mean(), atol=0.06
    )


def test_summary_table_conditional_columns(adapt_results, insurance_results):
    plain = an.create_summary_table(adapt_results)
    assert "cum. avoided (M$)" not in plain.columns
    assert "final insured (%)" not in plain.columns
    with_ins = an.create_summary_table(insurance_results)
    for col in (
        "final insured (%)", "ever insured (%)", "peak insured (%)",
        "premium final year ($)", "cum. household cost (M$)",
    ):
        assert col in with_ins.columns
    seq_rows = with_ins.iloc[:NO_SEQ]
    assert (seq_rows["ever insured (%)"] >= seq_rows["final insured (%)"]).all()
    assert (seq_rows["peak insured (%)"] >= seq_rows["final insured (%)"]).all()


def test_scenario_summary_rows_and_reference(
    adapt_results, insurance_results, baseline_results
):
    table = an.scenario_summary(
        {"adapt": adapt_results, "insure": insurance_results},
        reference=adapt_results,
        baselines={"adapt": baseline_results},
    )
    assert list(table["scenario"]) == ["adapt", "insure"]
    self_row = table.set_index("scenario").loc["adapt"]
    assert self_row["cum. avoided vs reference (M$)"] == 0.0
    assert np.isnan(table.set_index("scenario").loc["insure", "cum. avoided (M$)"])
    assert np.isnan(self_row["peak insured (%)"])
    assert not np.isnan(table.set_index("scenario").loc["insure", "peak insured (%)"])


# ---------------------------------------------------------------------------
# Affordability diagnostics
# ---------------------------------------------------------------------------

def _pool() -> tuple[np.ndarray, np.ndarray]:
    """A skewed EAD pool and matching incomes."""
    rng = np.random.default_rng(0)
    ead = np.concatenate([np.zeros(40), rng.lognormal(5.0, 1.5, 60)])
    income = rng.lognormal(11.3, 0.9, ead.size)
    return ead, income


def test_premium_offers_mirror_engine_rules():
    ead, _ = _pool()
    offers = an.premium_offers(ead, deductible=0.1)
    np.testing.assert_allclose(offers[an.PRICING_COMMUNITY], ead.mean())
    np.testing.assert_allclose(offers[an.PRICING_RISK_BASED], 0.9 * ead)


def test_affordability_table_extremes():
    ead, income = _pool()
    rich = an.affordability_table(ead, np.full_like(income, 1e12), 0.06, 0.1)
    poor = an.affordability_table(ead, np.zeros_like(income), 0.06, 0.1)
    assert list(rich["pricing"]) == [an.PRICING_COMMUNITY, an.PRICING_RISK_BASED]
    assert (rich["affordable - all households (%)"] == 100.0).all()
    assert (rich["affordable - top-10% risk (%)"] == 100.0).all()
    assert (poor["affordable - all households (%)"] == 0.0).all()
    summary = an.ead_summary(ead)
    assert summary["share_zero"] == pytest.approx(0.4)
    assert summary["p99"] >= summary["median"] >= 0.0


def test_affordability_vs_subsidy_monotone():
    ead, income = _pool()
    subs = np.linspace(0.0, 0.99, 25)
    shares = an.affordability_vs_subsidy(ead, income, 0.06, 0.1, subs)
    for label in (an.PRICING_COMMUNITY, an.PRICING_RISK_BASED):
        share = shares[label]
        assert share.shape == subs.shape
        assert (np.diff(share) >= 0).all()
        assert 0.0 <= share[0] <= share[-1] <= 1.0
