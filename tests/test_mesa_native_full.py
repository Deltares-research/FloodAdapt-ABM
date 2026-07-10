"""
test_mesa_native_full.py
========================
Phase **4b-full** tests for the native-class integration
(:mod:`floodadapt_abm.mesa_native_full`).

Phase 4b-full binds the **real honeybees ``Model``** as the time-owning base
class and routes decisions through the native DYNAMO-M ``DecisionModule`` (via
:class:`DynamoLiveRule`), feeding a deterministic coastal-node population from
the FloodAdapt lookup table through the PRE.4 adapter.  The central gate is the
**triple bit-parity**::

    run_mesa_native_full == run_mesa_native == engine.run

so that swapping the framework-free 4b scaffold for a genuine honeybees-driven
model is provably non-breaking.  Tests that need the honeybees base class skip
cleanly when it is unavailable; the native-rule test additionally requires
DYNAMO-M.
"""
from __future__ import annotations

import numpy as np
import pytest

from floodadapt_abm import (
    SimulationEngine,
    CouplingConfig,
    SEURule,
    ThresholdRule,
    run_mesa_native,
    run_mesa_native_full,
    FloodAdaptSLRModelFull,
    CoastalNodePopulationFull,
    HoneybeesNotAvailable,
    HONEYBEES_AVAILABLE,
    DYNAMO_M_AVAILABLE,
)

from tests.conftest import make_mock_dataset


SLR = np.linspace(0.0, 1.5, 20)

pytestmark = pytest.mark.skipif(
    not HONEYBEES_AVAILABLE, reason="honeybees not importable"
)


def _engine(rule_factory=None, seed=42):
    ds = make_mock_dataset(n_objects=120, n_events=6, seed=3)
    cfg = CouplingConfig()
    cfg.random_seed = seed
    rule = None if rule_factory is None else rule_factory(cfg)
    return SimulationEngine(ds=ds, config=cfg, decision_rule=rule)


# ---------------------------------------------------------------------------
# The 4b-full gate: honeybees-driven == scaffold == engine.run, bit for bit.
# ---------------------------------------------------------------------------
class TestTripleBitParity:
    @pytest.mark.parametrize("no_seq", [1, 3])
    @pytest.mark.parametrize("seed", [0, 42, 123])
    def test_full_equals_engine_run(self, no_seq, seed):
        full = run_mesa_native_full(_engine(), SLR, no_seq=no_seq, seed=seed)
        loop = _engine().run(SLR, no_seq=no_seq, seed=seed)
        assert np.array_equal(full["damage_history"], loop["damage_history"])
        assert np.array_equal(full["adapted_history"], loop["adapted_history"])
        assert np.array_equal(full["adoption_fraction"], loop["adoption_fraction"])

    @pytest.mark.parametrize("seed", [0, 7, 99])
    def test_full_equals_scaffold(self, seed):
        full = run_mesa_native_full(_engine(), SLR, no_seq=3, seed=seed)
        scaf = run_mesa_native(_engine(), SLR, no_seq=3, seed=seed)
        assert np.array_equal(full["damage_history"], scaf["damage_history"])
        assert np.array_equal(full["adapted_history"], scaf["adapted_history"])
        assert np.array_equal(full["adoption_fraction"], scaf["adoption_fraction"])

    def test_thresholdrule_bit_parity(self):
        rf = lambda cfg: ThresholdRule(cfg.decision, damage_threshold=0.30)
        full = run_mesa_native_full(_engine(rf), SLR, no_seq=3, seed=7)
        loop = _engine(rf).run(SLR, no_seq=3, seed=7)
        assert np.array_equal(full["damage_history"], loop["damage_history"])
        assert np.array_equal(full["adapted_history"], loop["adapted_history"])

    def test_track_eu_bit_parity(self):
        full = run_mesa_native_full(_engine(), SLR, no_seq=2, seed=5, track_eu=True)
        loop = _engine().run(SLR, no_seq=2, seed=5, track_eu=True)
        assert np.array_equal(
            np.nan_to_num(full["eu_adapt_history"]),
            np.nan_to_num(loop["eu_adapt_history"]),
        )
        assert np.array_equal(
            np.nan_to_num(full["eu_do_nothing_history"]),
            np.nan_to_num(loop["eu_do_nothing_history"]),
        )


# ---------------------------------------------------------------------------
# Genuine honeybees Model: the clock is owned by the framework base class.
# ---------------------------------------------------------------------------
class TestHoneybeesClock:
    def test_subclasses_real_honeybees_model(self):
        from honeybees.model import Model

        model = FloodAdaptSLRModelFull(_engine(), SLR, seed=1)
        assert isinstance(model, Model)

    def test_step_advances_one_year(self):
        model = FloodAdaptSLRModelFull(_engine(), SLR, seed=1)
        assert model.timestep == model.current_timestep == 0
        model.step()
        assert model.timestep == model.current_timestep == 1

    def test_run_model_reaches_horizon(self):
        model = FloodAdaptSLRModelFull(_engine(), SLR, seed=1)
        model.run_model()
        assert model.current_timestep == model.n_timesteps == len(SLR)

    def test_current_time_advances_yearly(self):
        model = FloodAdaptSLRModelFull(_engine(), SLR, seed=1, start_year=2020)
        y0 = model.current_time.year
        model.step()
        assert model.current_time.year == y0 + 1

    def test_result_shapes(self):
        engine = _engine()
        res = run_mesa_native_full(engine, SLR, no_seq=4, seed=2)
        n = engine.n_agents
        assert res["damage_history"].shape == (4, n, len(SLR))
        assert res["adapted_history"].shape == (4, n, len(SLR))
        assert res["adoption_fraction"].shape == (4, len(SLR))


# ---------------------------------------------------------------------------
# Object graph mirrors DYNAMO-M and drives the live state through the adapter.
# ---------------------------------------------------------------------------
class TestObjectGraphAndAdapter:
    def test_agent_graph(self):
        engine = _engine()
        model = FloodAdaptSLRModelFull(engine, SLR, seed=1)
        assert isinstance(model.agents.regions, CoastalNodePopulationFull)
        assert model.agents.regions.n == engine.n_agents
        assert model.agents.regions.state is engine.state

    def test_population_step_mutates_state(self):
        engine = _engine()
        model = FloodAdaptSLRModelFull(engine, SLR, seed=1)
        before = engine.state.is_adapted.sum()
        model.run_model()
        assert engine.state.is_adapted.sum() >= before

    def test_adapter_round_trips_each_tick(self):
        # The population owns a LookupTableAdapter; a manual populate/write_back
        # round trip must reproduce the live state exactly (PRE.4 contract).
        engine = _engine()
        model = FloodAdaptSLRModelFull(engine, SLR, seed=1)
        adapter = model.agents.regions.adapter
        node = adapter.populate(0.5)
        assert np.array_equal(node.adapt.astype(bool), engine.state.is_adapted)
        adapter.write_back(node)
        assert np.array_equal(engine.state.is_adapted, node.adapt.astype(bool))


# ---------------------------------------------------------------------------
# PRE.3 shared-engine staleness guard also protects the honeybees driver.
# ---------------------------------------------------------------------------
class TestSharedEngineStalenessGuard:
    def test_stale_model_step_raises(self):
        engine = _engine()
        model_a = FloodAdaptSLRModelFull(engine, SLR, seed=1)
        model_b = FloodAdaptSLRModelFull(engine, SLR, seed=2)  # resets shared state
        with pytest.raises(RuntimeError, match="stale"):
            model_a.step()
        model_b.step()
        assert model_b.timestep == 1

    def test_engine_run_invalidates_manual_model(self):
        engine = _engine()
        model = FloodAdaptSLRModelFull(engine, SLR, seed=1)
        model.step()
        engine.run(SLR, no_seq=1, seed=0)
        with pytest.raises(RuntimeError, match="stale"):
            model.step()


# ---------------------------------------------------------------------------
# Native-class integration: the native DYNAMO-M DecisionModule drives 4b-full.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not DYNAMO_M_AVAILABLE, reason="DYNAMO-M not importable")
class TestNativeDecisionModuleUnderFull:
    def test_live_rule_bit_parity(self):
        from floodadapt_abm import DynamoLiveRule

        ds = make_mock_dataset(n_objects=120, n_events=6, seed=3)
        cfg = CouplingConfig()
        amenity = SimulationEngine(ds=ds, config=cfg)._data.amenity_value

        def eng():
            return SimulationEngine(
                ds=ds, config=cfg,
                decision_rule=DynamoLiveRule(cfg.decision, amenity_value=amenity),
            )

        full = run_mesa_native_full(eng(), SLR, no_seq=2, seed=9)
        loop = eng().run(SLR, no_seq=2, seed=9)
        scaf = run_mesa_native(eng(), SLR, no_seq=2, seed=9)
        assert np.array_equal(full["adapted_history"], loop["adapted_history"])
        assert np.array_equal(full["adapted_history"], scaf["adapted_history"])
        assert np.array_equal(full["damage_history"], loop["damage_history"])
