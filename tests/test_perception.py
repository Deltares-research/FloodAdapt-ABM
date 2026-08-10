"""
test_perception.py
==================
Tests for the magnitude-aware flood-perception update: severity-scaled
risk-perception peaks, the flood-significance threshold, and exact legacy
equivalence of the binary mode.

The severity response is a **single** one-parameter form, the power law
``peak = risk_perc_max * s ** gamma``.  The tests here pin what
gamma covers rather than comparing forms.
"""
from __future__ import annotations

import numpy as np
import pytest

from floodadapt_abm import CouplingConfig, SimulationEngine
from tests.conftest import make_mock_dataset, historical_modes_config


def _engine(perception_mode: str = "binary", **decision_overrides) -> SimulationEngine:
    cfg = historical_modes_config()
    cfg.decision.perception_mode = perception_mode
    for key, value in decision_overrides.items():
        setattr(cfg.decision, key, value)
    return SimulationEngine(ds=make_mock_dataset(), config=cfg)


def test_binary_mode_matches_legacy_formula():
    """Binary mode: full-scale peak for any flooded agent (legacy/native)."""
    engine = _engine("binary")
    dec = engine.config.decision
    flooded = np.zeros(engine.n_agents, dtype=bool)
    flooded[0] = True

    engine.update_flood_experience(flooded)

    # Flooded agent: timer 0 -> rp = rp_max * 1.6^0 + rp_min.
    assert np.isclose(
        engine.state.risk_perception[0], dec.risk_perc_max + dec.risk_perc_min
    )
    # Unflooded agents (timer 100 after increment): at the floor.
    assert np.allclose(
        engine.state.risk_perception[1:], dec.risk_perc_min, atol=1e-6
    )


def test_severity_scales_peak_with_power_law():
    """Severity mode: peak = rp_max * severity**gamma (gamma = 0.5)."""
    engine = _engine("severity", perception_severity_exponent=0.5)
    dec = engine.config.decision
    n = engine.n_agents

    flooded = np.zeros(n, dtype=bool)
    flooded[:3] = True
    severity = np.zeros(n)
    severity[0] = 1.0    # total loss
    severity[1] = 0.25   # a quarter of max potential damage
    severity[2] = 0.05   # minor flood

    engine.update_flood_experience(flooded, severity)

    rp = engine.state.risk_perception
    # Total loss reproduces the full legacy peak exactly.
    assert np.isclose(rp[0], dec.risk_perc_max + dec.risk_perc_min)
    # Concave scaling: 25 % damage -> sqrt(0.25) = 50 % of rp_max.
    assert np.isclose(rp[1], dec.risk_perc_max * 0.5 + dec.risk_perc_min)
    # 5 % damage -> sqrt(0.05) ~ 22 % of rp_max.
    assert np.isclose(
        rp[2], dec.risk_perc_max * np.sqrt(0.05) + dec.risk_perc_min, rtol=1e-5
    )
    # Ordering: worse floods -> higher perception.
    assert rp[0] > rp[1] > rp[2] > rp[3]
    # Unflooded agents stay at the floor.
    assert np.allclose(rp[3:], dec.risk_perc_min, atol=1e-6)


def test_severity_peak_decays_from_scaled_level():
    """The decay in later years starts from the severity-scaled peak."""
    engine = _engine("severity", perception_severity_exponent=1.0)
    dec = engine.config.decision
    n = engine.n_agents

    flooded = np.zeros(n, dtype=bool)
    flooded[0] = True
    severity = np.zeros(n)
    severity[0] = 0.5

    engine.update_flood_experience(flooded, severity)
    year0 = engine.state.risk_perception[0]
    # One quiet year later: same peak, decayed by 1.6^coef.
    engine.update_flood_experience(np.zeros(n, dtype=bool), np.zeros(n))
    year1 = engine.state.risk_perception[0]

    expected0 = dec.risk_perc_max * 0.5 + dec.risk_perc_min
    expected1 = (
        dec.risk_perc_max * 0.5 * 1.6 ** dec.risk_perc_coef + dec.risk_perc_min
    )
    assert np.isclose(year0, expected0)
    assert np.isclose(year1, expected1, rtol=1e-5)


def test_significance_threshold_creates_deadband():
    """
    Damage below the significance threshold must not register as a flood:
    the flood timer keeps counting and perception stays at the floor.  This
    is the fix for float-noise floods and for adapted agents resetting
    their timer on tiny residual damages.
    """
    cfg = historical_modes_config()
    cfg.decision.perception_mode = "severity"
    cfg.decision.flood_significance_threshold = 0.01
    engine = SimulationEngine(ds=make_mock_dataset(), config=cfg)

    # Severity below threshold for agent 0, above for agent 1.
    realised = np.zeros(engine.n_agents)
    realised[0] = 0.001 * engine.max_pot_dmg[0]   # 0.1 % — insignificant
    realised[1] = 0.20 * engine.max_pot_dmg[1]    # 20 % — significant
    severity = realised / engine.max_pot_dmg
    was_flooded = severity > cfg.decision.flood_significance_threshold

    assert not was_flooded[0] and was_flooded[1]

    engine.update_flood_experience(was_flooded, severity)
    # Insignificant flood: timer NOT reset (99 + 1 = 100), perception floor.
    assert engine.state.flood_timer[0] == 100
    assert np.isclose(
        engine.state.risk_perception[0], cfg.decision.risk_perc_min, atol=1e-6
    )
    # Significant flood: timer reset, elevated perception.
    assert engine.state.flood_timer[1] == 0
    assert engine.state.risk_perception[1] > 0.5


# ---------------------------------------------------------------------------
# The severity exponent (perception_severity_exponent) and its guards
# ---------------------------------------------------------------------------

def _peak(engine: SimulationEngine, severities: np.ndarray) -> np.ndarray:
    """Run one perception update and return the risk-perception array."""
    n = engine.n_agents
    flooded = np.zeros(n, dtype=bool)
    flooded[: severities.size] = severities > 0
    sev = np.zeros(n)
    sev[: severities.size] = severities
    engine.update_flood_experience(flooded, sev)
    return engine.state.risk_perception


def test_default_form_is_power_and_unchanged():
    """perception_severity_form='power' (default) equals the pre-form code."""
    engine = _engine("severity")
    assert engine.config.decision.perception_severity_form == "power"
    dec = engine.config.decision
    rp = _peak(engine, np.array([1.0, 0.25, 0.05]))
    # Pinned expectations of the original single-form implementation.
    assert np.isclose(rp[0], dec.risk_perc_max + dec.risk_perc_min)
    assert np.isclose(rp[1], dec.risk_perc_max * 0.5 + dec.risk_perc_min)
    assert np.isclose(
        rp[2], dec.risk_perc_max * np.sqrt(0.05) + dec.risk_perc_min, rtol=1e-5
    )


@pytest.mark.parametrize("gamma", [0.2, 0.5, 1.0, 1.35, 2.0, 5.0])
def test_gamma_spans_concave_linear_and_near_miss(gamma):
    """One exponent covers the whole hypothesis range.

    Whatever gamma is, the response stays monotone and pinned at both ends;
    only the small-flood behaviour changes.  This is the property that made
    the two extra forms redundant.
    """
    engine = _engine("severity", perception_severity_exponent=gamma)
    dec = engine.config.decision
    sev = np.array([1.0, 0.5, 0.25, 0.05])
    rp = _peak(engine, sev)

    # Pinned at s = 1 for every gamma: a total loss is the full spike.
    assert np.isclose(rp[0], dec.risk_perc_max + dec.risk_perc_min)
    # Monotone in severity, and unflooded agents stay at the floor.
    assert rp[0] > rp[1] > rp[2] > rp[3] > dec.risk_perc_min
    assert np.allclose(rp[sev.size:], dec.risk_perc_min, atol=1e-6)
    # Exact formula.
    assert np.allclose(
        rp[: sev.size],
        dec.risk_perc_max * sev ** gamma + dec.risk_perc_min,
        rtol=1e-5,
    )


def test_gamma_above_one_suppresses_small_floods():
    """gamma > 1 is the near-miss arm: small floods barely register.

    This is what replaced the retired ``threshold_linear`` form, so it has
    to actually behave like a soft deadband.
    """
    convex = _peak(_engine("severity", perception_severity_exponent=2.0),
                   np.array([0.1]))[0]
    concave = _peak(_engine("severity", perception_severity_exponent=0.5),
                    np.array([0.1]))[0]
    floor = _engine("severity").config.decision.risk_perc_min
    rp_max = _engine("severity").config.decision.risk_perc_max

    # 10 % damage: 1 % of the spike at gamma=2 against ~32 % at gamma=0.5.
    assert np.isclose(convex - floor, rp_max * 0.01, rtol=1e-4)
    assert (convex - floor) < 0.05 * (concave - floor)


def test_zero_exponent_is_rejected():
    """gamma = 0 must raise rather than spike every agent.

    ``0.0 ** 0.0 == 1.0`` in NumPy, so without the guard even agents that
    never flooded would get the full peak.
    """
    engine = _engine("severity", perception_severity_exponent=0.0)
    with pytest.raises(ValueError, match="perception_severity_exponent"):
        engine.update_flood_experience(
            np.zeros(engine.n_agents, dtype=bool), np.zeros(engine.n_agents)
        )
    engine = _engine("severity", perception_severity_exponent=-1.0)
    with pytest.raises(ValueError, match="perception_severity_exponent"):
        engine.update_flood_experience(
            np.zeros(engine.n_agents, dtype=bool), np.zeros(engine.n_agents)
        )


@pytest.mark.parametrize(
    "form, gamma_hint",
    [("saturating_exp", "0.5"), ("threshold_linear", "1.3")],
)
def test_retired_forms_raise_a_directed_error(form, gamma_hint):
    """The removed forms must name their measured gamma equivalent."""
    engine = _engine("severity", perception_severity_form=form)
    with pytest.raises(ValueError, match="removed") as excinfo:
        engine.update_flood_experience(
            np.zeros(engine.n_agents, dtype=bool), np.zeros(engine.n_agents)
        )
    message = str(excinfo.value)
    assert "perception_severity_exponent=" + gamma_hint in message


def test_unknown_form_raises():
    """An unrecognised form is reported as such, not as a retired one."""
    engine = _engine("severity", perception_severity_form="nope")
    with pytest.raises(ValueError, match="only supported form"):
        engine.update_flood_experience(
            np.zeros(engine.n_agents, dtype=bool), np.zeros(engine.n_agents)
        )


def test_binary_mode_ignores_the_severity_fields():
    """Binary mode must not touch the power-law branch at all."""
    ref = _engine("binary")
    alt = _engine("binary", perception_severity_exponent=3.0)
    flooded = np.zeros(ref.n_agents, dtype=bool)
    flooded[0] = True
    ref.update_flood_experience(flooded)
    alt.update_flood_experience(flooded)
    assert np.array_equal(ref.state.risk_perception, alt.state.risk_perception)


def test_step_reports_flood_severity():
    """step() exposes the per-agent severity for diagnostics."""
    engine = _engine("binary")
    res = engine.step(0, 0.5, np.random.default_rng(0))
    assert res["flood_severity"].shape == (engine.n_agents,)
    assert res["flood_severity"].dtype == np.float32
    assert (res["flood_severity"] >= 0).all()
