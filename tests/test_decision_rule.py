"""
test_decision_rule.py
=====================
Unit tests for the pluggable decision rules (Phase-3 Strategy Pattern):
``ThresholdRule`` (legacy heuristic) and ``SEURule`` (validated DYNAMO-M SEU).

The key correctness guarantees verified here are:

* ``ThresholdRule`` reproduces the legacy ``ABMSimulator`` decision formula
  (adapt iff realised/max_pot_dmg > threshold, masked on non-adapted &
  max_pot_dmg > 0).
* ``SEURule`` reproduces ``DynamoDecisionBridge.evaluate_decisions`` *exactly*
  on a shared state — i.e. the refactor preserves the Phase-1-validated SEU
  science bit-for-bit.
"""
from __future__ import annotations

import numpy as np
import pytest

from floodadapt_abm import (
    CouplingConfig,
    DecisionConfig,
    DynamoDecisionBridge,
)
from floodadapt_abm.agent_state import AgentState
from floodadapt_abm.decision_rule import ThresholdRule, SEURule


# ===========================================================================
# ThresholdRule
# ===========================================================================
def _state_from_bridge(bridge: DynamoDecisionBridge) -> AgentState:
    """Snapshot a bridge's per-agent state into an AgentState."""
    return AgentState(
        wealth=bridge.wealth.copy(),
        income=bridge.income.copy(),
        risk_perception=bridge.risk_perception.copy(),
        flood_timer=bridge.flood_timer.copy(),
        is_adapted=bridge.is_adapted.copy(),
        time_adapted=np.zeros(bridge.n_agents, dtype=np.int32),
    )


def test_threshold_rule_matches_legacy_formula(mock_ds):
    """ThresholdRule reproduces the exact ABMSimulator masking + comparison."""
    bridge = DynamoDecisionBridge(ds=mock_ds)
    n = bridge.n_agents
    rule = ThresholdRule(DecisionConfig(), damage_threshold=0.30)

    max_pot = bridge.max_pot_dmg.copy()
    # Realised damages: a mix below/above the 0.3 threshold, some zero max_pot.
    realised = np.array(
        [0.1, 0.5, 0.29, 0.31, 0.0][: n] + [0.4] * max(0, n - 5),
        dtype=np.float32,
    ) * max_pot

    state = AgentState.initial(n, bridge.income, bridge.wealth, 0.01)
    newly = rule.should_adapt(
        agent_state=state,
        damages_this_year=realised,
        damages_no_adapt=np.zeros((n, 3), dtype=np.float32),
        damages_adapt=np.zeros((n, 3), dtype=np.float32),
        event_freqs=np.array([0.5, 0.2, 0.1]),
        max_pot_dmg=max_pot,
        adaptation_costs=np.zeros(n, dtype=np.float32),
    )
    expected = (realised / max_pot) > 0.30
    assert np.array_equal(newly, expected)


def test_threshold_rule_ignores_already_adapted(mock_ds):
    bridge = DynamoDecisionBridge(ds=mock_ds)
    n = bridge.n_agents
    rule = ThresholdRule(DecisionConfig())
    state = AgentState.initial(n, bridge.income, bridge.wealth, 0.01)
    state.is_adapted[:] = True  # everyone already adapted

    realised = bridge.max_pot_dmg.copy()  # huge damage
    newly = rule.should_adapt(
        state, realised,
        np.zeros((n, 2), np.float32), np.zeros((n, 2), np.float32),
        np.array([0.5, 0.1]), bridge.max_pot_dmg, np.zeros(n, np.float32),
    )
    assert not newly.any()  # no NEW adopters (already adapted)


def test_threshold_rule_zero_max_pot_dmg_never_adapts():
    cfg = DecisionConfig()
    rule = ThresholdRule(cfg)
    n = 4
    state = AgentState.initial(
        n, np.ones(n, np.float32), np.ones(n, np.float32), 0.01
    )
    max_pot = np.zeros(n, dtype=np.float32)  # no potential damage
    newly = rule.should_adapt(
        state, np.ones(n, np.float32),
        np.zeros((n, 2), np.float32), np.zeros((n, 2), np.float32),
        np.array([0.5, 0.1]), max_pot, np.zeros(n, np.float32),
    )
    assert not newly.any()


# ===========================================================================
# SEURule ↔ bridge parity (preserves validated science)
# ===========================================================================
def test_seurule_matches_bridge_evaluate_decisions(mock_ds_factory):
    """
    SEURule.should_adapt must equal DynamoDecisionBridge.evaluate_decisions on
    a shared fresh state (error_interval=0 → deterministic).
    """
    ds = mock_ds_factory(n_objects=40, n_events=6, n_slr=5, seed=11)
    cfg = CouplingConfig()  # error_interval=0 by default → deterministic
    bridge = DynamoDecisionBridge(ds=ds, config=cfg)
    bridge.prepare_damage_arrays(slr_value=1.0)

    # Give agents a fresh-flood risk perception so some actually adapt.
    flooded = np.ones(bridge.n_agents, dtype=bool)
    bridge.update_flood_experience(flooded)

    # Build a rule + matching state.
    rule = SEURule(cfg.decision, rng=np.random.default_rng(cfg.random_seed),
                   amenity_value=bridge.amenity_value)
    state = _state_from_bridge(bridge)

    newly_rule = rule.should_adapt(
        agent_state=state,
        damages_this_year=np.zeros(bridge.n_agents, dtype=np.float32),
        damages_no_adapt=bridge._damage_no_measures,
        damages_adapt=bridge._damage_floodproof,
        event_freqs=bridge._event_freqs,
        max_pot_dmg=bridge.max_pot_dmg,
        adaptation_costs=bridge._annual_adapt_cost,
    )
    newly_bridge = bridge.evaluate_decisions(year_index=0)
    assert np.array_equal(newly_rule, newly_bridge)


def test_seurule_matches_bridge_power_utility(mock_ds_factory):
    """Parity also holds for sigma != 1 (power utility)."""
    ds = mock_ds_factory(n_objects=30, n_events=6, seed=21)
    cfg = CouplingConfig(decision=DecisionConfig(risk_aversion=2.0))
    bridge = DynamoDecisionBridge(ds=ds, config=cfg)
    bridge.prepare_damage_arrays(slr_value=1.5)
    bridge.update_flood_experience(np.ones(bridge.n_agents, dtype=bool))

    rule = SEURule(cfg.decision, amenity_value=bridge.amenity_value)
    state = _state_from_bridge(bridge)
    newly_rule = rule.should_adapt(
        state, np.zeros(bridge.n_agents, np.float32),
        bridge._damage_no_measures, bridge._damage_floodproof,
        bridge._event_freqs, bridge.max_pot_dmg, bridge._annual_adapt_cost,
    )
    newly_bridge = bridge.evaluate_decisions(0)
    assert np.array_equal(newly_rule, newly_bridge)


def test_seurule_exposes_eu_diagnostics(mock_ds):
    cfg = CouplingConfig()
    bridge = DynamoDecisionBridge(ds=mock_ds, config=cfg)
    bridge.prepare_damage_arrays(slr_value=1.0)
    rule = SEURule(cfg.decision, amenity_value=bridge.amenity_value)
    state = _state_from_bridge(bridge)
    rule.should_adapt(
        state, np.zeros(bridge.n_agents, np.float32),
        bridge._damage_no_measures, bridge._damage_floodproof,
        bridge._event_freqs, bridge.max_pot_dmg, bridge._annual_adapt_cost,
    )
    assert rule.last_eu_adapt is not None
    assert rule.last_eu_do_nothing is not None
    assert rule.last_eu_adapt.shape == (bridge.n_agents,)


def test_seurule_degenerate_zero_risk_no_adoption(mock_ds_factory):
    """risk_perception ≡ 0 → nobody adapts (V1 degenerate, rule-level)."""
    ds = mock_ds_factory(n_objects=25, seed=4)
    cfg = CouplingConfig(decision=DecisionConfig(risk_perc_min=0.0, risk_perc_max=0.0))
    bridge = DynamoDecisionBridge(ds=ds, config=cfg)
    bridge.prepare_damage_arrays(slr_value=1.0)
    rule = SEURule(cfg.decision, amenity_value=bridge.amenity_value)
    state = _state_from_bridge(bridge)
    state.risk_perception[:] = 0.0
    newly = rule.should_adapt(
        state, np.zeros(bridge.n_agents, np.float32),
        bridge._damage_no_measures, bridge._damage_floodproof,
        bridge._event_freqs, bridge.max_pot_dmg, bridge._annual_adapt_cost,
    )
    assert not newly.any()
