"""
test_insurance.py
=================
Tests for the flood-insurance option: the ported ``_calc_eu_insure``
kernel, the three-way ``decide()`` contract, the engine-level
premium/coverage bookkeeping, and the triple-parity contract with
insurance enabled.
"""
from __future__ import annotations

import numpy as np
import pytest

from floodadapt_abm import (
    ACTION_ADAPT,
    ACTION_DO_NOTHING,
    ACTION_INSURE,
    CouplingConfig,
    SEURule,
    SimulationEngine,
    ThresholdRule,
    run_mesa_native,
    run_mesa_native_full,
    HONEYBEES_AVAILABLE,
)
from floodadapt_abm._core.dynamo_decision_bridge import _calc_eu_insure
from tests.conftest import make_mock_dataset, historical_modes_config

SLR = np.linspace(0.0, 1.0, 8)
SEED = 123


def _insurance_cfg(**overrides) -> CouplingConfig:
    """Insurance-enabled config on the alternative behaviour modes."""
    cfg = historical_modes_config()
    cfg.decision.include_insurance = True
    # Synthetic incomes so the premium affordability gate is meaningful.
    cfg.decision.income_mode = "synthetic_lognormal"
    for key, value in overrides.items():
        setattr(cfg.decision, key, value)
    return cfg


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def _kernel_inputs(n_agents: int = 4, n_floods: int = 3):
    rng = np.random.default_rng(0)
    return dict(
        n_agents=n_agents,
        wealth=np.full(n_agents, 200_000.0, dtype=np.float32),
        income=np.full(n_agents, 50_000.0, dtype=np.float32),
        expenditure_cap=0.06,
        amenity_value=np.zeros(n_agents, dtype=np.float32),
        amenity_weight=1.0,
        risk_perception=np.full(n_agents, 1.0, dtype=np.float32),
        expected_damages=rng.uniform(
            0, 50_000, (n_floods, n_agents)
        ).astype(np.float32),
        premium=np.full(n_agents, 500.0, dtype=np.float32),
        p_floods=np.array([0.5, 0.1, 0.01], dtype=np.float32),
        T=np.full(n_agents, 15, dtype=np.int32),
        r=0.032,
        sigma=1.0,
        error_terms=np.ones(n_agents, dtype=np.float32),
    )


def test_calc_eu_insure_shape_and_finiteness():
    eu = _calc_eu_insure(**_kernel_inputs())
    assert eu.shape == (4,)
    assert np.isfinite(eu).all()


def test_calc_eu_insure_affordability_neg_inf():
    """Premium above the expenditure cap -> EU = -inf for that agent."""
    inputs = _kernel_inputs()
    inputs["income"] = np.array([50_000.0, 50_000.0, 100.0, 50_000.0],
                                dtype=np.float32)
    eu = _calc_eu_insure(**inputs)
    assert eu[2] == -np.inf
    assert np.isfinite(eu[[0, 1, 3]]).all()


def test_deductible_reduces_damage_burden():
    """A smaller deductible (better coverage) yields higher insured EU."""
    full = _calc_eu_insure(**_kernel_inputs(), deductible=1.0)   # no coverage
    partial = _calc_eu_insure(**_kernel_inputs(), deductible=0.1)
    assert (partial >= full).all() and (partial > full).any()


def test_insure_beats_do_nothing_for_high_risk_cheap_premium():
    """
    With large perceived damages and a token premium, EU_insure must exceed
    EU_do_nothing (which bears the full damages).
    """
    from floodadapt_abm._core.dynamo_decision_bridge import _calc_eu_do_nothing

    inputs = _kernel_inputs()
    inputs["expected_damages"] = np.full((3, 4), 100_000.0, dtype=np.float32)
    inputs["premium"] = np.full(4, 100.0, dtype=np.float32)
    eu_insure = _calc_eu_insure(**inputs)

    dn = {k: v for k, v in inputs.items()
          if k not in ("premium", "expenditure_cap")}
    eu_do_nothing = _calc_eu_do_nothing(
        adapted=np.zeros(4, dtype=np.int32), **dn
    )
    assert (eu_insure > eu_do_nothing).all()


# ---------------------------------------------------------------------------
# decide() contract
# ---------------------------------------------------------------------------

def test_default_decide_delegates_to_should_adapt():
    """ThresholdRule (no decide override) never insures."""
    cfg = historical_modes_config()
    engine = SimulationEngine(
        ds=make_mock_dataset(),
        decision_rule=ThresholdRule(cfg.decision, damage_threshold=0.0),
        config=cfg,
    )
    dmg_no, dmg_fp = engine.prepare_damages(0.5)
    actions = engine.decision_rule.decide(
        agent_state=engine.state,
        damages_this_year=np.full(engine.n_agents, 1e9, dtype=np.float32),
        damages_no_adapt=dmg_no,
        damages_adapt=dmg_fp,
        event_freqs=engine._data.p_floods_seu,
        max_pot_dmg=engine.max_pot_dmg,
        adaptation_costs=engine._annual_adapt_cost,
        insurance_premium=np.full(engine.n_agents, 1.0, dtype=np.float32),
    )
    assert actions.dtype == np.int8
    assert set(np.unique(actions)) <= {ACTION_DO_NOTHING, ACTION_ADAPT}


def test_seurule_decide_without_premium_matches_should_adapt():
    """decide(premium=None) reduces bit-exactly to the two-way rule."""
    cfg = historical_modes_config()
    engine = SimulationEngine(ds=make_mock_dataset(), config=cfg)
    dmg_no, dmg_fp = engine.prepare_damages(1.0)
    kwargs = dict(
        agent_state=engine.state,
        damages_this_year=np.zeros(engine.n_agents, dtype=np.float32),
        damages_no_adapt=dmg_no,
        damages_adapt=dmg_fp,
        event_freqs=engine._data.p_floods_seu,
        max_pot_dmg=engine.max_pot_dmg,
        adaptation_costs=engine._annual_adapt_cost,
    )
    rule: SEURule = engine.decision_rule
    actions = rule.decide(**kwargs, insurance_premium=None)
    adapt_bool = rule.should_adapt(**kwargs)
    assert np.array_equal(actions == ACTION_ADAPT, adapt_bool)
    assert rule.last_eu_insure is None


def test_seurule_three_way_exclusive_and_masked():
    """Actions are exclusive; already-adapted agents never act again."""
    cfg = _insurance_cfg()
    engine = SimulationEngine(ds=make_mock_dataset(), config=cfg)
    engine.state.is_adapted[:3] = True
    dmg_no, dmg_fp = engine.prepare_damages(1.0)
    actions = engine.decision_rule.decide(
        agent_state=engine.state,
        damages_this_year=np.zeros(engine.n_agents, dtype=np.float32),
        damages_no_adapt=dmg_no,
        damages_adapt=dmg_fp,
        event_freqs=engine._data.p_floods_seu,
        max_pot_dmg=engine.max_pot_dmg,
        adaptation_costs=engine._annual_adapt_cost,
        insurance_premium=np.full(engine.n_agents, 100.0, dtype=np.float32),
    )
    assert np.all(actions[:3] == ACTION_DO_NOTHING)  # sticky floodproofing
    assert set(np.unique(actions)) <= {
        ACTION_DO_NOTHING, ACTION_ADAPT, ACTION_INSURE,
    }
    assert engine.decision_rule.last_eu_insure is not None


# ---------------------------------------------------------------------------
# Engine bookkeeping
# ---------------------------------------------------------------------------

def test_insurance_off_does_not_perturb_the_run():
    """
    ``include_insurance=False`` must leave the run bit-identical.

    Turning the option off must not consume RNG, add arrays, or shift any
    damage.  Compared against an otherwise-identical config rather than a
    stored golden file, so the invariant is self-contained.
    """
    base = SimulationEngine(ds=make_mock_dataset(), config=historical_modes_config())
    with_flag_off = SimulationEngine(
        ds=make_mock_dataset(),
        config=historical_modes_config(include_insurance=False),
    )
    res_base = base.run(SLR, no_seq=2, seed=SEED)
    res_off = with_flag_off.run(SLR, no_seq=2, seed=SEED)

    assert np.array_equal(res_base["damage_history"], res_off["damage_history"])
    assert np.array_equal(res_base["adapted_history"], res_off["adapted_history"])
    assert "insured_history" not in res_base
    assert "insured_history" not in res_off


def test_run_returns_insurance_arrays_with_expected_shapes():
    engine = SimulationEngine(ds=make_mock_dataset(), config=_insurance_cfg())
    res = engine.run(SLR, no_seq=2, seed=SEED)
    n, t = engine.n_agents, len(SLR)
    assert res["insured_history"].shape == (2, n, t)
    assert res["insured_history"].dtype == bool
    assert res["out_of_pocket_history"].shape == (2, n, t)
    assert res["premium_history"].shape == (2, t)
    assert res["insured_fraction"].shape == (2, t)
    # Gross damage history is unchanged in schema and never smaller than
    # the out-of-pocket damage.
    assert (res["out_of_pocket_history"] <= res["damage_history"] + 1e-4).all()


def test_year0_starts_uninsured_and_coverage_lags_one_year():
    """Coverage timing: decided at t covers t+1; year 0 pays gross."""
    engine = SimulationEngine(ds=make_mock_dataset(), config=_insurance_cfg())
    res0 = engine.step(0, 0.0, np.random.default_rng(SEED))
    # Year-0 out-of-pocket equals gross damage (nobody was insured yet).
    np.testing.assert_array_equal(res0["out_of_pocket"], res0["damages"])
    assert (res0["premium_paid"] == 0).all()


def test_out_of_pocket_applies_deductible_to_insured_agents():
    engine = SimulationEngine(ds=make_mock_dataset(), config=_insurance_cfg())
    rng = np.random.default_rng(SEED)
    engine.step(0, 0.0, rng)
    # Force agent 0 insured for the next year with a known premium.
    engine.state.is_insured[:] = False
    engine.state.is_insured[0] = True
    engine._premium_locked = 123.0
    res1 = engine.step(1, 1.0, rng)
    dmg = res1["damages"].astype(np.float64)
    oop = res1["out_of_pocket"].astype(np.float64)
    deductible = engine.config.decision.insurance_deductible
    np.testing.assert_allclose(oop[0], deductible * dmg[0], rtol=1e-5)
    np.testing.assert_allclose(oop[1:], dmg[1:], rtol=1e-5)
    assert res1["premium_paid"][0] == np.float32(123.0)
    assert (res1["premium_paid"][1:] == 0).all()


def test_premium_equals_mean_ead():
    engine = SimulationEngine(ds=make_mock_dataset(), config=_insurance_cfg())
    res = engine.step(0, 0.5, np.random.default_rng(SEED))
    expected = float(engine._data.compute_expected_annual_damages(False).mean())
    assert res["premium"] == pytest.approx(expected, rel=1e-6)


def test_insured_and_adapted_mutually_exclusive_over_run():
    engine = SimulationEngine(ds=make_mock_dataset(), config=_insurance_cfg())
    res = engine.run(SLR, no_seq=2, seed=SEED)
    overlap = res["insured_history"] & res["adapted_history"]
    assert not overlap.any()


def test_insurance_annual_reset():
    """is_insured is re-decided every step, not sticky."""
    engine = SimulationEngine(ds=make_mock_dataset(), config=_insurance_cfg())
    rng = np.random.default_rng(SEED)
    engine.step(0, 0.0, rng)
    engine.state.is_insured[:] = True  # pretend everyone insured last year
    res = engine.step(1, 0.5, rng)
    # Post-step status equals this year's decision, not last year's carry.
    assert np.array_equal(res["is_insured"], engine.state.is_insured)
    assert np.array_equal(
        res["is_insured"], res["newly_insured"]
    )


# ---------------------------------------------------------------------------
# Parity with insurance ON
# ---------------------------------------------------------------------------

def test_triple_parity_with_insurance_on():
    """engine.run == run_mesa_native (== run_mesa_native_full) with insurance."""
    ds = make_mock_dataset()
    engine = SimulationEngine(ds=ds, config=_insurance_cfg())
    ref = engine.run(SLR, no_seq=2, seed=SEED)

    engine2 = SimulationEngine(ds=make_mock_dataset(), config=_insurance_cfg())
    mirror = run_mesa_native(engine2, SLR, no_seq=2, seed=SEED)

    for key in (
        "damage_history", "adapted_history", "insured_history",
        "out_of_pocket_history", "premium_history", "premium_paid_history",
    ):
        assert np.array_equal(ref[key], mirror[key]), key

    if HONEYBEES_AVAILABLE:
        engine3 = SimulationEngine(
            ds=make_mock_dataset(), config=_insurance_cfg()
        )
        full = run_mesa_native_full(engine3, SLR, no_seq=2, seed=SEED)
        for key in (
            "damage_history", "adapted_history", "insured_history",
            "out_of_pocket_history", "premium_history", "premium_paid_history",
        ):
            assert np.array_equal(ref[key], full[key]), key


# ---------------------------------------------------------------------------
# Premium pricing policy (community vs risk-based, loading, subsidy)
# ---------------------------------------------------------------------------

def _skewed_ds():
    """Mock table with a heavily skewed risk pool (a few high-EAD agents)."""
    ds = make_mock_dataset(n_objects=100)
    rng = np.random.default_rng(4)
    dmg = ds["total_damage"].values
    # Multiply the no-measures damages of a 10 % tail by 20x.
    tail = rng.choice(100, size=10, replace=False)
    dmg[tail, :, 0, :] *= 20.0
    dmg[tail, :, 1, :] *= 20.0
    ds["total_damage"].values = dmg
    return ds


def test_community_pricing_is_flat_and_equals_mean_ead():
    engine = SimulationEngine(ds=_skewed_ds(), config=_insurance_cfg())
    engine.prepare_damages(0.5)
    premium = engine._compute_premium_offer()
    ead = engine._data.compute_expected_annual_damages(False)
    assert np.allclose(premium, premium[0])                  # flat
    assert premium[0] == pytest.approx(ead.mean(), rel=1e-5)


def test_risk_based_pricing_is_per_agent_expected_payout():
    """Fair price of the cover sold: (1 - deductible) * own EAD."""
    cfg = _insurance_cfg(insurance_pricing="risk_based")
    engine = SimulationEngine(ds=_skewed_ds(), config=cfg)
    engine.prepare_damages(0.5)
    premium = engine._compute_premium_offer()
    ead = engine._data.compute_expected_annual_damages(False)
    expected = (1.0 - cfg.decision.insurance_deductible) * ead
    np.testing.assert_allclose(premium, expected.astype(np.float32), rtol=1e-5)
    assert premium.std() > 0                                  # heterogeneous
    # On a skewed pool the median household pays far less than the flat rate.
    assert np.median(premium) < ead.mean()


def test_loading_and_subsidy_scale_the_premium():
    base = SimulationEngine(ds=_skewed_ds(), config=_insurance_cfg())
    base.prepare_damages(0.5)
    p0 = base._compute_premium_offer()

    loaded = SimulationEngine(
        ds=_skewed_ds(), config=_insurance_cfg(insurance_loading=1.3)
    )
    loaded.prepare_damages(0.5)
    np.testing.assert_allclose(loaded._compute_premium_offer(), p0 * 1.3, rtol=1e-5)

    subsidised = SimulationEngine(
        ds=_skewed_ds(), config=_insurance_cfg(insurance_subsidy=0.5)
    )
    subsidised.prepare_damages(0.5)
    np.testing.assert_allclose(
        subsidised._compute_premium_offer(), p0 * 0.5, rtol=1e-5
    )


def test_unknown_pricing_mode_raises():
    cfg = _insurance_cfg(insurance_pricing="bogus")
    engine = SimulationEngine(ds=make_mock_dataset(), config=cfg)
    engine.prepare_damages(0.5)
    with pytest.raises(ValueError, match="insurance_pricing"):
        engine._compute_premium_offer()


def test_risk_based_pricing_raises_uptake_on_skewed_pool():
    """
    The substantive finding: over a skewed risk pool a flat community premium
    prices out (nearly) everyone, while the actuarially-fair risk-based
    premium is affordable for the low-risk majority.
    """
    ds = _skewed_ds()
    community = SimulationEngine(ds=ds, config=_insurance_cfg()).run(
        SLR, no_seq=2, seed=SEED
    )
    risk_based = SimulationEngine(
        ds=_skewed_ds(), config=_insurance_cfg(insurance_pricing="risk_based")
    ).run(SLR, no_seq=2, seed=SEED)

    assert (
        risk_based["insured_fraction"].mean()
        >= community["insured_fraction"].mean()
    )


def test_premium_paid_history_matches_locked_offer():
    """Insured agents pay the premium locked at the previous decision."""
    cfg = _insurance_cfg(insurance_pricing="risk_based")
    engine = SimulationEngine(ds=_skewed_ds(), config=cfg)
    res = engine.run(SLR, no_seq=2, seed=SEED)
    paid = res["premium_paid_history"]
    insured = res["insured_history"]
    assert paid.shape == insured.shape
    assert (paid[~insured] == 0).all()      # uninsured pay nothing
    assert (paid[:, :, 0] == 0).all()       # nobody insured in year 0
    if insured.any():
        assert (paid[insured] > 0).all()    # insured pay a positive premium


def test_parallel_run_parity_with_insurance_on():
    """n_jobs parallel path reproduces the sequential insurance arrays."""
    seq = SimulationEngine(ds=make_mock_dataset(), config=_insurance_cfg()).run(
        SLR, no_seq=3, seed=SEED, n_jobs=1
    )
    par = SimulationEngine(ds=make_mock_dataset(), config=_insurance_cfg()).run(
        SLR, no_seq=3, seed=SEED, n_jobs=-1
    )
    for key in (
        "damage_history", "insured_history", "out_of_pocket_history",
        "premium_history", "premium_paid_history",
    ):
        assert np.array_equal(seq[key], par[key]), key
