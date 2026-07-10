"""
test_dynamo_live_rule.py
========================
Phase-4a tests for :class:`floodadapt_abm.dynamo_live_rule.DynamoLiveRule` - the
adapter that drives the **native** DYNAMO-M ``DecisionModule`` as a parity
oracle for the ported :class:`~floodadapt_abm.decision_rule.SEURule`.

Test groups
-----------
1. Availability / guarded import behaviour (runs even when DYNAMO-M is absent).
2. **Parity gate** (skipped when DYNAMO-M is unavailable): DynamoLiveRule and
   SEURule must produce identical adaptation decisions and matching expected
   utilities (to float32 tolerance) on a shared agent state, across several
   configurations (sigma=1 and sigma=2, adapted subsets, risk-perception
   spread).  This is the mechanism that guarantees the ported kernels have not
   drifted from upstream DYNAMO-M.
3. Integration: the live rule plugs straight into ``SimulationEngine``.
"""
from __future__ import annotations

import numpy as np
import pytest

from floodadapt_abm import (
    CouplingConfig,
    DecisionConfig,
    SimulationEngine,
    SEURule,
    DYNAMO_M_AVAILABLE,
    DynamoMNotAvailable,
)
from floodadapt_abm.agent_state import AgentState
from floodadapt_abm.dynamo_live_rule import (
    DynamoLiveRule,
    resolve_dynamo_path,
    _probe_availability,
)

# Tolerances for float32 SEU integrals (trapz over ~9 points, values ~1e1-1e2).
_RTOL = 1e-4
_ATOL = 1e-3

requires_dynamo = pytest.mark.skipif(
    not DYNAMO_M_AVAILABLE,
    reason="native DYNAMO-M DecisionModule not importable in this environment",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _finite_equal(a: np.ndarray, b: np.ndarray) -> None:
    """Assert two EU arrays agree, treating -inf entries as a matching mask."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    neg_a, neg_b = np.isneginf(a), np.isneginf(b)
    assert np.array_equal(neg_a, neg_b), "affordability (-inf) masks differ"
    finite = ~neg_a
    np.testing.assert_allclose(a[finite], b[finite], rtol=_RTOL, atol=_ATOL)


def _make_engine(mock_ds_factory, cfg, *, n_objects=40, n_events=6, seed=3):
    ds = mock_ds_factory(n_objects=n_objects, n_events=n_events, seed=seed)
    return SimulationEngine(ds=ds, config=cfg)


def _decision_kwargs(engine, state, slr=1.0):
    d_no, d_fp = engine.prepare_damages(slr)
    return dict(
        agent_state=state,
        damages_this_year=np.zeros(state.n_agents, dtype=np.float32),
        damages_no_adapt=d_no,
        damages_adapt=d_fp,
        event_freqs=engine._event_freqs,
        max_pot_dmg=engine.max_pot_dmg,
        adaptation_costs=engine._annual_adapt_cost,
    )


# ===========================================================================
# 1. Availability / guarded import
# ===========================================================================
class TestAvailability:
    def test_flag_is_bool(self):
        assert isinstance(DYNAMO_M_AVAILABLE, bool)

    def test_probe_matches_flag(self):
        # Lightweight probe should agree with the module-level flag.
        assert _probe_availability() == DYNAMO_M_AVAILABLE

    def test_resolve_path_prefers_argument(self):
        assert resolve_dynamo_path(r"X:\some\path") == r"X:\some\path"

    def test_resolve_path_uses_env(self, monkeypatch):
        monkeypatch.setenv("DYNAMO_M_PATH", r"Y:\env\path")
        assert resolve_dynamo_path() == r"Y:\env\path"

    def test_missing_path_raises_typed_error(self):
        cfg = DecisionConfig()
        with pytest.raises(DynamoMNotAvailable):
            DynamoLiveRule(cfg, dynamo_path=r"Z:\definitely\not\here")


# ===========================================================================
# 2. Parity gate (the Phase-1 cross-check)
# ===========================================================================
@requires_dynamo
class TestParityWithSEURule:
    @pytest.mark.parametrize("sigma", [1.0, 2.0])
    def test_decisions_and_eu_match(self, mock_ds_factory, sigma):
        cfg = CouplingConfig()
        cfg.decision.risk_aversion = sigma
        engine = _make_engine(mock_ds_factory, cfg)

        state = engine.state
        rng = np.random.default_rng(11)
        state.risk_perception[:] = rng.uniform(
            0.3, 2.0, state.n_agents
        ).astype(np.float32)

        amenity = engine._data.amenity_value
        seu = SEURule(cfg.decision, amenity_value=amenity)
        live = DynamoLiveRule(cfg.decision, amenity_value=amenity)

        kw = _decision_kwargs(engine, state)
        a_seu = seu.should_adapt(**kw)
        a_live = live.should_adapt(**kw)

        assert np.array_equal(a_seu, a_live), "adaptation decisions differ"
        _finite_equal(seu.last_eu_adapt, live.last_eu_adapt)
        _finite_equal(seu.last_eu_do_nothing, live.last_eu_do_nothing)

    def test_parity_with_some_agents_already_adapted(self, mock_ds_factory):
        cfg = CouplingConfig()
        engine = _make_engine(mock_ds_factory, cfg, seed=9)
        state = engine.state

        rng = np.random.default_rng(5)
        state.is_adapted[rng.random(state.n_agents) < 0.4] = True
        state.time_adapted[state.is_adapted] = 3
        state.risk_perception[:] = rng.uniform(
            0.2, 1.8, state.n_agents
        ).astype(np.float32)

        amenity = engine._data.amenity_value
        seu = SEURule(cfg.decision, amenity_value=amenity)
        live = DynamoLiveRule(cfg.decision, amenity_value=amenity)

        kw = _decision_kwargs(engine, state, slr=1.5)
        a_seu = seu.should_adapt(**kw)
        a_live = live.should_adapt(**kw)

        assert np.array_equal(a_seu, a_live)
        # Already-adapted agents never "newly adapt".
        assert not a_live[state.is_adapted].any()
        _finite_equal(seu.last_eu_do_nothing, live.last_eu_do_nothing)

    def test_parity_under_affordability_constraint(self, mock_ds_factory):
        # Tight expenditure cap -> many agents priced out (-inf) in BOTH rules.
        cfg = CouplingConfig()
        cfg.decision.expenditure_cap = 1e-6
        engine = _make_engine(mock_ds_factory, cfg, seed=2)
        state = engine.state

        amenity = engine._data.amenity_value
        seu = SEURule(cfg.decision, amenity_value=amenity)
        live = DynamoLiveRule(cfg.decision, amenity_value=amenity)

        kw = _decision_kwargs(engine, state)
        a_seu = seu.should_adapt(**kw)
        a_live = live.should_adapt(**kw)

        assert np.array_equal(a_seu, a_live)
        # With no affordable adaptation, nobody adapts.
        assert not a_live.any()
        _finite_equal(seu.last_eu_adapt, live.last_eu_adapt)


# ===========================================================================
# 3. SimulationEngine integration
# ===========================================================================
@requires_dynamo
class TestEngineIntegration:
    def test_live_rule_runs_in_engine(self, mock_ds_factory):
        cfg = CouplingConfig()
        ds = mock_ds_factory(n_objects=30, n_events=6, seed=4)
        live = DynamoLiveRule(cfg.decision)
        engine = SimulationEngine(ds=ds, config=cfg, decision_rule=live)

        slr = np.linspace(0.0, 1.5, 10)
        res = engine.run(slr, no_seq=2, seed=42, track_eu=True)

        assert res["damage_history"].shape == (2, engine.n_agents, 10)
        assert res["adapted_history"].dtype == bool
        # EU diagnostics recorded through the live rule.
        assert np.isfinite(res["eu_adapt_history"]).any()

    def test_engine_seu_vs_live_same_trajectory(self, mock_ds_factory):
        # Full-run parity: identical seeds/inputs -> identical adoption history.
        cfg = CouplingConfig()
        ds = mock_ds_factory(n_objects=30, n_events=6, seed=6)

        eng_seu = SimulationEngine(ds=ds, config=cfg)  # SEURule default
        eng_live = SimulationEngine(
            ds=ds, config=cfg, decision_rule=DynamoLiveRule(cfg.decision)
        )

        slr = np.linspace(0.0, 1.5, 12)
        r_seu = eng_seu.run(slr, no_seq=3, seed=123)
        r_live = eng_live.run(slr, no_seq=3, seed=123)

        assert np.array_equal(
            r_seu["adapted_history"], r_live["adapted_history"]
        )
        np.testing.assert_allclose(
            r_seu["adoption_fraction"], r_live["adoption_fraction"],
            rtol=0, atol=0,
        )
