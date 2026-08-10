"""
Tests for the rule/driver status tags and the preferred-rule factory.

The status vocabulary is part of the public contract: docs, examples and the
notebook all state that the native DYNAMO-M coupling is the preferred path and
that the port is a parity-verified stand-in.  These tests pin that.
"""
import numpy as np
import pytest

from floodadapt_abm import (
    STATUS_EXPERIMENT,
    STATUS_PREFERRED,
    STATUS_REFERENCE,
    STATUS_VERIFICATION,
    CouplingConfig,
    DecisionRule,
    SEURule,
    ThresholdRule,
    preferred_decision_rule,
)
from floodadapt_abm import mesa_native, mesa_native_full
from floodadapt_abm.dynamo_live_rule import DYNAMO_M_AVAILABLE, DynamoLiveRule


def test_rule_status_tags():
    """Each shipped rule carries its documented status."""
    assert ThresholdRule.STATUS == STATUS_EXPERIMENT
    assert SEURule.STATUS == STATUS_REFERENCE
    assert DynamoLiveRule.STATUS == STATUS_PREFERRED


def test_third_party_rule_defaults_to_experiment():
    """A subclass that does not opt in is not silently promoted."""

    class MyRule(DecisionRule):
        def should_adapt(self, *args, **kwargs):  # pragma: no cover - unused
            return np.zeros(0, dtype=bool)

    assert MyRule.STATUS == STATUS_EXPERIMENT


def test_driver_status_tags():
    """The honeybees driver is preferred; the mirror is verification-only."""
    assert mesa_native_full.STATUS == STATUS_PREFERRED
    assert mesa_native.STATUS == STATUS_VERIFICATION


def test_preferred_rule_falls_back_for_risk_based_pricing():
    """Per-agent premiums are outside the native kernel's capability.

    Native calcEU_insure discounts premium.mean() (decision_module.py:337),
    so risk-based pricing must route to the port.
    """
    from dataclasses import replace

    cfg = CouplingConfig()
    risk_cfg = replace(
        cfg.decision, include_insurance=True, insurance_pricing="risk_based"
    )
    rule = preferred_decision_rule(risk_cfg)
    assert isinstance(rule, SEURule)
    assert rule.STATUS == STATUS_REFERENCE


@pytest.mark.skipif(not DYNAMO_M_AVAILABLE, reason="DYNAMO-M not installed")
def test_preferred_rule_is_native_for_community_insurance():
    """The flat community premium IS expressible natively -> native rule."""
    from dataclasses import replace

    cfg = CouplingConfig()
    comm_cfg = replace(
        cfg.decision, include_insurance=True, insurance_pricing="community"
    )
    rule = preferred_decision_rule(comm_cfg)
    assert isinstance(rule, DynamoLiveRule)


@pytest.mark.skipif(not DYNAMO_M_AVAILABLE, reason="DYNAMO-M not installed")
def test_dynamo_live_rule_clone_has_independent_native_module():
    """clone() must build a fresh DecisionModule per worker (no shared _dm).

    should_adapt assigns self._dm.error_terms_stay on every call; a shared
    module would let parallel sequences overwrite each other's error terms.
    """
    cfg = CouplingConfig()
    rule = DynamoLiveRule(cfg.decision)
    c1 = rule.clone(rng_seed=1)
    c2 = rule.clone(rng_seed=2)
    assert c1._dm is not rule._dm
    assert c2._dm is not rule._dm
    assert c1._dm is not c2._dm
    # And the clone still carries the preferred status.
    assert c1.STATUS == STATUS_PREFERRED


@pytest.mark.skipif(not DYNAMO_M_AVAILABLE, reason="DYNAMO-M not installed")
def test_dynamo_live_rule_rejects_per_agent_premium():
    """decide() must raise on a varying premium instead of silently averaging."""
    import numpy as np

    from floodadapt_abm import AgentState

    cfg = CouplingConfig()
    rule = DynamoLiveRule(cfg.decision)
    n = 4
    state = AgentState.initial(
        n_agents=n,
        income=np.full(n, 50_000.0),
        wealth=np.full(n, 200_000.0),
        risk_perc_min=cfg.decision.risk_perc_min,
    )
    args = dict(
        agent_state=state,
        damages_this_year=np.zeros(n),
        damages_no_adapt=np.full((n, 2), 100.0),
        damages_adapt=np.full((n, 2), 10.0),
        event_freqs=np.array([0.1, 0.01]),
        max_pot_dmg=np.full(n, 300_000.0),
        adaptation_costs=np.full(n, 1_000.0),
    )
    # Flat premium: fine.
    rule.decide(**args, insurance_premium=np.full(n, 500.0))
    # Varying premium: must raise, not average.
    with pytest.raises(ValueError, match="premium.mean"):
        rule.decide(**args, insurance_premium=np.array([1.0, 2.0, 3.0, 4.0]))


def test_preferred_rule_status_is_always_usable():
    """Whatever the environment, the factory returns a documented status."""
    cfg = CouplingConfig()
    rule = preferred_decision_rule(cfg.decision)
    assert rule.STATUS in (STATUS_PREFERRED, STATUS_REFERENCE)


@pytest.mark.skipif(not DYNAMO_M_AVAILABLE, reason="DYNAMO-M not installed")
def test_preferred_rule_is_native_when_available():
    """With DYNAMO-M present and a serial run, the native rule is chosen."""
    cfg = CouplingConfig()
    rule = preferred_decision_rule(cfg.decision)
    assert isinstance(rule, DynamoLiveRule)
    assert rule.STATUS == STATUS_PREFERRED


def test_preferred_rule_falls_back_when_dynamo_missing(tmp_path):
    """A bad DYNAMO-M path degrades to the port instead of raising."""
    cfg = CouplingConfig()
    rule = preferred_decision_rule(cfg.decision, dynamo_path=str(tmp_path / "nope"))
    assert isinstance(rule, SEURule)
