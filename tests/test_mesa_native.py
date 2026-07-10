"""
test_mesa_native.py
===================
Phase-4b tests for the Mesa-native driving seam
(:mod:`floodadapt_abm.mesa_native`).

The defining property of Phase 4b is **time-ownership inversion**: the year
loop moves from ``SimulationEngine.run`` into ``FloodAdaptSLRModel.step`` ticks,
mirroring the native DYNAMO-M ``SLRModel.run_model``.  The central gate is that
this inversion is *non-breaking*: the tick-driven driver must reproduce
``engine.run`` **bit-for-bit**.
"""
from __future__ import annotations

import numpy as np
import pytest

from floodadapt_abm import (
    SimulationEngine,
    CouplingConfig,
    SEURule,
    ThresholdRule,
    FloodAdaptSLRModel,
    CoastalNodePopulation,
    run_mesa_native,
    DYNAMO_M_AVAILABLE,
)

from tests.conftest import make_mock_dataset


SLR = np.linspace(0.0, 1.5, 20)


def _engine(rule_factory=None, seed=42):
    ds = make_mock_dataset(n_objects=120, n_events=6, seed=3)
    cfg = CouplingConfig()
    cfg.random_seed = seed
    rule = None if rule_factory is None else rule_factory(cfg)
    return SimulationEngine(ds=ds, config=cfg, decision_rule=rule)


# ---------------------------------------------------------------------------
# The Phase-4b gate: tick driver == engine.run(), bit for bit.
# ---------------------------------------------------------------------------
class TestBitParityWithEngineRun:
    @pytest.mark.parametrize("no_seq", [1, 3])
    @pytest.mark.parametrize("seed", [0, 42, 123])
    def test_seurule_bit_parity(self, no_seq, seed):
        engine = _engine()
        native = run_mesa_native(engine, SLR, no_seq=no_seq, seed=seed)
        # Fresh engine so state is not shared between the two runs.
        engine2 = _engine()
        loop = engine2.run(SLR, no_seq=no_seq, seed=seed)

        assert np.array_equal(native["damage_history"], loop["damage_history"])
        assert np.array_equal(native["adapted_history"], loop["adapted_history"])
        assert np.array_equal(native["adoption_fraction"], loop["adoption_fraction"])

    def test_thresholdrule_bit_parity(self):
        rf = lambda cfg: ThresholdRule(cfg.decision, damage_threshold=0.30)
        engine = _engine(rf)
        native = run_mesa_native(engine, SLR, no_seq=3, seed=7)
        engine2 = _engine(rf)
        loop = engine2.run(SLR, no_seq=3, seed=7)
        assert np.array_equal(native["damage_history"], loop["damage_history"])
        assert np.array_equal(native["adapted_history"], loop["adapted_history"])

    def test_track_eu_bit_parity(self):
        engine = _engine()
        native = run_mesa_native(engine, SLR, no_seq=2, seed=5, track_eu=True)
        engine2 = _engine()
        loop = engine2.run(SLR, no_seq=2, seed=5, track_eu=True)
        # NaN-aware comparison (unreached years / masked agents are NaN).
        assert np.array_equal(
            np.nan_to_num(native["eu_adapt_history"]),
            np.nan_to_num(loop["eu_adapt_history"]),
        )
        assert np.array_equal(
            np.nan_to_num(native["eu_do_nothing_history"]),
            np.nan_to_num(loop["eu_do_nothing_history"]),
        )


# ---------------------------------------------------------------------------
# Time ownership: the model owns the clock and advances one year per tick.
# ---------------------------------------------------------------------------
class TestTimeOwnership:
    def test_step_advances_one_year(self):
        engine = _engine()
        model = FloodAdaptSLRModel(engine, SLR, seed=1)
        assert model.timestep == 0
        model.step()
        assert model.timestep == 1
        model.step()
        assert model.timestep == 2

    def test_run_model_reaches_horizon(self):
        engine = _engine()
        model = FloodAdaptSLRModel(engine, SLR, seed=1)
        model.run_model()
        assert model.timestep == model.n_timesteps == len(SLR)

    def test_result_shapes(self):
        engine = _engine()
        res = run_mesa_native(engine, SLR, no_seq=4, seed=2)
        n = engine.n_agents
        assert res["damage_history"].shape == (4, n, len(SLR))
        assert res["adapted_history"].shape == (4, n, len(SLR))
        assert res["adoption_fraction"].shape == (4, len(SLR))


# ---------------------------------------------------------------------------
# Object graph mirrors DYNAMO-M: SLRModel -> Agents -> CoastalNode(Population).
# ---------------------------------------------------------------------------
class TestObjectGraph:
    def test_agent_graph_mirrors_dynamo(self):
        engine = _engine()
        model = FloodAdaptSLRModel(engine, SLR, seed=1)
        assert hasattr(model, "agents")
        assert isinstance(model.agents.regions, CoastalNodePopulation)
        assert model.agents.regions.n == engine.n_agents
        # The population is a view over the engine's live AgentState.
        assert model.agents.regions.state is engine.state

    def test_population_step_mutates_state(self):
        engine = _engine()
        model = FloodAdaptSLRModel(engine, SLR, seed=1)
        before = engine.state.is_adapted.copy()
        for _ in range(len(SLR)):
            model.step()
        # Over 20 years at least some households should have adapted.
        assert engine.state.is_adapted.sum() >= before.sum()


# ---------------------------------------------------------------------------
# Guarded: the live DYNAMO-M rule also drives correctly under Mesa-native time.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not DYNAMO_M_AVAILABLE, reason="DYNAMO-M not importable")
class TestLiveRuleUnderMesaNative:
    def test_live_rule_bit_parity(self):
        from floodadapt_abm import DynamoLiveRule

        ds = make_mock_dataset(n_objects=120, n_events=6, seed=3)
        cfg = CouplingConfig()
        amenity = SimulationEngine(ds=ds, config=cfg)._data.amenity_value

        eng_a = SimulationEngine(
            ds=ds, config=cfg,
            decision_rule=DynamoLiveRule(cfg.decision, amenity_value=amenity),
        )
        native = run_mesa_native(eng_a, SLR, no_seq=2, seed=9)

        eng_b = SimulationEngine(
            ds=ds, config=cfg,
            decision_rule=DynamoLiveRule(cfg.decision, amenity_value=amenity),
        )
        loop = eng_b.run(SLR, no_seq=2, seed=9)
        assert np.array_equal(native["adapted_history"], loop["adapted_history"])


# ---------------------------------------------------------------------------
# PRE.3 (2026-07-09 review): shared-engine staleness guard.
# ---------------------------------------------------------------------------
class TestSharedEngineStalenessGuard:
    """Constructing a second model on the same engine must not let the first
    model silently mutate the second model's state (mesa_native.py review
    finding, task PRE.3)."""

    def test_stale_model_step_raises(self):
        engine = _engine()
        model_a = FloodAdaptSLRModel(engine, SLR, seed=1)
        model_b = FloodAdaptSLRModel(engine, SLR, seed=2)  # resets shared state

        with pytest.raises(RuntimeError, match="stale"):
            model_a.step()
        # The newer model owns the state and keeps working.
        model_b.step()
        assert model_b.timestep == 1

    def test_engine_run_invalidates_manual_model(self):
        engine = _engine()
        model = FloodAdaptSLRModel(engine, SLR, seed=1)
        model.step()
        engine.run(SLR, no_seq=1, seed=0)  # reset_state() per sequence
        with pytest.raises(RuntimeError, match="stale"):
            model.step()

    def test_fresh_model_unaffected(self):
        engine = _engine()
        model = FloodAdaptSLRModel(engine, SLR, seed=1)
        model.run_model()
        assert model.timestep == model.n_timesteps

    def test_state_epoch_increments(self):
        engine = _engine()
        e0 = engine.state_epoch
        engine.reset_state()
        assert engine.state_epoch == e0 + 1
