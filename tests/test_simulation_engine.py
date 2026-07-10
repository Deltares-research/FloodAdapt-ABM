"""
test_simulation_engine.py
========================
Tests for the unified ``SimulationEngine`` (Phase 2 + Phase 3 refactor):

* shapes / plumbing of ``step`` and ``run``,
* the lifespan-dryproof reset (Phase-3 gap closure),
* the degenerate ``risk_perception ≡ 0`` guarantee at engine level,
* ``ThresholdRule`` irreversibility (legacy behaviour),
* sequence independence via ``reset_state``.

Bit-for-bit SEURule↔bridge parity is covered in ``test_decision_rule.py``;
here we exercise the engine that *drives* those rules.
"""
from __future__ import annotations

import numpy as np
import pytest

from floodadapt_abm import CouplingConfig, DecisionConfig
from floodadapt_abm.agent_state import AgentState
from floodadapt_abm.decision_rule import DecisionRule, ThresholdRule, SEURule
from floodadapt_abm.simulation_engine import SimulationEngine


# ---------------------------------------------------------------------------
# Test-only stub rules
# ---------------------------------------------------------------------------
class AlwaysAdaptRule(DecisionRule):
    """Adapt every currently non-adapted agent (deterministic driver)."""

    def should_adapt(self, agent_state, damages_this_year, damages_no_adapt,
                     damages_adapt, event_freqs, max_pot_dmg, adaptation_costs):
        return ~agent_state.is_adapted


class NeverAdaptRule(DecisionRule):
    def should_adapt(self, agent_state, damages_this_year, damages_no_adapt,
                     damages_adapt, event_freqs, max_pot_dmg, adaptation_costs):
        return np.zeros(agent_state.n_agents, dtype=bool)


# ---------------------------------------------------------------------------
# Construction & plumbing
# ---------------------------------------------------------------------------
def test_engine_defaults_to_seurule(mock_ds):
    eng = SimulationEngine(ds=mock_ds)
    assert isinstance(eng.decision_rule, SEURule)
    assert eng.n_agents == int(eng.is_residential.sum())


def test_engine_only_residential_agents(mock_ds_factory):
    ds = mock_ds_factory(n_objects=20, residential_fraction=0.5)
    eng = SimulationEngine(ds=ds)
    assert eng.n_agents == 10  # 50 % of 20


def test_step_returns_expected_keys(mock_ds):
    eng = SimulationEngine(ds=mock_ds, decision_rule=NeverAdaptRule(DecisionConfig()))
    res = eng.step(0, slr_value=1.0, rng=np.random.default_rng(0))
    for key in ("occurred_events", "damages", "was_flooded", "newly_adapted",
                "expired", "is_adapted"):
        assert key in res
    assert res["damages"].shape == (eng.n_agents,)
    assert res["is_adapted"].shape == (eng.n_agents,)


def test_run_output_shapes(mock_ds):
    eng = SimulationEngine(ds=mock_ds)
    slr = np.linspace(0, 1.5, 8)
    out = eng.run(slr, no_seq=3, seed=1)
    assert out["damage_history"].shape == (3, eng.n_agents, 8)
    assert out["adapted_history"].shape == (3, eng.n_agents, 8)
    assert out["adoption_fraction"].shape == (3, 8)


def test_run_reproducible(mock_ds):
    eng = SimulationEngine(ds=mock_ds)
    slr = np.linspace(0, 1.0, 6)
    a = eng.run(slr, no_seq=2, seed=42)
    b = eng.run(slr, no_seq=2, seed=42)
    assert np.array_equal(a["damage_history"], b["damage_history"])
    assert np.array_equal(a["adapted_history"], b["adapted_history"])


def test_track_eu_populates_history(mock_ds):
    eng = SimulationEngine(ds=mock_ds)
    out = eng.run(np.linspace(0, 1, 4), no_seq=1, seed=0, track_eu=True)
    assert "eu_adapt_history" in out
    assert out["eu_adapt_history"].shape == (1, eng.n_agents, 4)
    assert np.isfinite(out["eu_adapt_history"]).any()


# ---------------------------------------------------------------------------
# Lifespan-dryproof reset (Phase-3 gap closure)
# ---------------------------------------------------------------------------
def test_lifespan_reset_expires_old_adaptation(mock_ds):
    cfg = CouplingConfig(decision=DecisionConfig(lifespan_dryproof=5))
    eng = SimulationEngine(ds=mock_ds, config=cfg,
                           decision_rule=NeverAdaptRule(cfg.decision))
    eng.state.is_adapted[:] = True
    eng.state.time_adapted[:] = 4
    expired = eng._apply_lifespan_reset()  # ages to 5 → expires
    assert expired.all()
    assert not eng.state.is_adapted.any()
    assert (eng.state.time_adapted == 0).all()


def test_lifespan_reset_disabled_when_zero(mock_ds):
    cfg = CouplingConfig(decision=DecisionConfig(lifespan_dryproof=0))
    eng = SimulationEngine(ds=mock_ds, config=cfg,
                           decision_rule=NeverAdaptRule(cfg.decision))
    eng.state.is_adapted[:] = True
    eng.state.time_adapted[:] = 500
    expired = eng._apply_lifespan_reset()
    assert not expired.any()
    assert eng.state.is_adapted.all()


def test_lifespan_turnover_over_long_run(mock_ds):
    """
    With an always-adapt driver and a short lifespan, adaptation must turn over:
    a cohort adapted early un-adapts on schedule and re-adapts.
    """
    lifespan = 5
    cfg = CouplingConfig(decision=DecisionConfig(lifespan_dryproof=lifespan))
    eng = SimulationEngine(ds=mock_ds, config=cfg,
                           decision_rule=AlwaysAdaptRule(cfg.decision))
    rng = np.random.default_rng(0)
    expired_by_year = []
    max_age_by_year = []
    for t in range(14):
        res = eng.step(t, 1.0, rng)
        expired_by_year.append(res["expired"].copy())
        max_age_by_year.append(int(eng.state.time_adapted.max()))
    expired_by_year = np.array(expired_by_year)  # (years, n_agents)

    # Turnover happens: adaptations expire on the lifespan schedule.  With an
    # always-adapt driver the re-adaptation is immediate, so the signal is the
    # `expired` flag (year 5, then again every 5 years), and the fact that the
    # adaptation age never exceeds the lifespan.
    assert expired_by_year[lifespan].all()      # cohort expires at year == lifespan
    assert expired_by_year.any(axis=0).all()    # every agent expired at least once
    assert max(max_age_by_year) < lifespan + 1  # age is bounded by the lifespan


def test_permanent_adaptation_without_lifespan(mock_ds):
    """lifespan_dryproof=None → adaptation is permanent (never expires)."""
    cfg = CouplingConfig(decision=DecisionConfig(lifespan_dryproof=None))
    eng = SimulationEngine(ds=mock_ds, config=cfg,
                           decision_rule=AlwaysAdaptRule(cfg.decision))
    rng = np.random.default_rng(0)
    for t in range(20):
        res = eng.step(t, 1.0, rng)
    assert res["is_adapted"].all()  # never expired


# ---------------------------------------------------------------------------
# Degenerate & legacy guarantees
# ---------------------------------------------------------------------------
def test_degenerate_zero_risk_no_adoption(mock_ds_factory):
    """SEURule with risk_perception ≡ 0 ⇒ no adaptation over a full run (V1)."""
    ds = mock_ds_factory(n_objects=30, seed=3)
    cfg = CouplingConfig(
        decision=DecisionConfig(risk_perc_min=0.0, risk_perc_max=0.0)
    )
    eng = SimulationEngine(ds=ds, config=cfg)
    out = eng.run(np.linspace(0, 2.0, 10), no_seq=2, seed=5)
    assert not out["adapted_history"].any()


def test_threshold_rule_adaptation_is_irreversible(mock_ds):
    """
    With ThresholdRule and no lifespan, once an agent adapts it stays adapted
    (legacy irreversible behaviour) — adoption fraction is non-decreasing.
    """
    cfg = CouplingConfig(decision=DecisionConfig(lifespan_dryproof=None))
    eng = SimulationEngine(
        ds=mock_ds, config=cfg,
        decision_rule=ThresholdRule(cfg.decision, damage_threshold=0.0),
    )
    rng = np.random.default_rng(7)
    frac = []
    for t in range(15):
        res = eng.step(t, 1.5, rng)
        frac.append(res["is_adapted"].mean())
    frac = np.array(frac)
    assert np.all(np.diff(frac) >= 0)  # monotonic non-decreasing


# ---------------------------------------------------------------------------
# reset_state
# ---------------------------------------------------------------------------
def test_reset_state_clears_adaptation(mock_ds):
    eng = SimulationEngine(ds=mock_ds,
                           decision_rule=AlwaysAdaptRule(DecisionConfig()))
    eng.step(0, 1.0, np.random.default_rng(0))
    assert eng.state.is_adapted.any()
    eng.reset_state()
    assert not eng.state.is_adapted.any()
    assert (eng.state.time_adapted == 0).all()


# ---------------------------------------------------------------------------
# Per-SLR interpolation cache
# ---------------------------------------------------------------------------
def test_interp_cache_is_bit_identical_to_fresh(mock_ds_factory):
    """A cached prepare_damages must return the exact arrays a fresh (uncached)
    interpolation produces."""
    ds = mock_ds_factory(n_objects=40, n_events=6, seed=9)
    eng = SimulationEngine(ds=ds)

    # Fresh (cache disabled).
    eng._data._interp_cache_enabled = False
    fresh_no, fresh_fp = eng.prepare_damages(0.7)
    fresh_no = fresh_no.copy()
    fresh_fp = fresh_fp.copy()

    # Cached (cache enabled) — first call populates, second call hits cache.
    eng._data._interp_cache_enabled = True
    eng._data.clear_interp_cache()
    eng.prepare_damages(0.7)
    cached_no, cached_fp = eng.prepare_damages(0.7)  # cache hit

    assert (0.7, "linear") in eng._data._interp_cache
    assert np.array_equal(fresh_no, cached_no)
    assert np.array_equal(fresh_fp, cached_fp)


def test_interp_cache_reused_across_run(mock_ds_factory):
    """A full run over a repeating SLR trajectory caches only the unique SLR
    levels, not one entry per tick."""
    ds = mock_ds_factory(n_objects=30, seed=2)
    eng = SimulationEngine(ds=ds)
    slr = np.array([0.0, 0.5, 1.0, 0.5, 0.0])  # 3 unique values, 5 ticks
    eng.run(slr, no_seq=2, seed=1)
    assert len(eng._data._interp_cache) == 3


# ---------------------------------------------------------------------------
# Parallel Monte-Carlo sequences (n_jobs)
# ---------------------------------------------------------------------------
def test_parallel_run_bit_identical_to_sequential(mock_ds_factory):
    """run(n_jobs=4) must reproduce run(n_jobs=1) bit-for-bit (default
    deterministic SEURule, error_interval == 0)."""
    ds = mock_ds_factory(n_objects=50, n_events=6, seed=13)
    slr = np.linspace(0.0, 1.5, 8)

    seq = SimulationEngine(ds=ds).run(slr, no_seq=5, seed=7, track_eu=True, n_jobs=1)
    par = SimulationEngine(ds=ds).run(slr, no_seq=5, seed=7, track_eu=True, n_jobs=4)

    assert np.array_equal(seq["damage_history"], par["damage_history"])
    assert np.array_equal(seq["adapted_history"], par["adapted_history"])
    assert np.allclose(
        seq["eu_adapt_history"], par["eu_adapt_history"],
        equal_nan=True, atol=0.0, rtol=0.0,
    )


def test_parallel_run_threshold_rule_bit_identical(mock_ds_factory):
    """Parallel parity also holds for the legacy ThresholdRule."""
    ds = mock_ds_factory(n_objects=40, seed=3)
    slr = np.linspace(0.0, 2.0, 6)
    cfg = CouplingConfig(decision=DecisionConfig(lifespan_dryproof=None))

    def _eng():
        return SimulationEngine(
            ds=ds, config=cfg,
            decision_rule=ThresholdRule(cfg.decision, damage_threshold=0.1),
        )

    seq = _eng().run(slr, no_seq=4, seed=11, n_jobs=1)
    par = _eng().run(slr, no_seq=4, seed=11, n_jobs=3)
    assert np.array_equal(seq["damage_history"], par["damage_history"])
    assert np.array_equal(seq["adapted_history"], par["adapted_history"])


def test_parallel_run_does_not_mutate_shared_state(mock_ds_factory):
    """Worker clones must isolate state — the parent engine's live state is
    untouched by a parallel run."""
    ds = mock_ds_factory(n_objects=25, seed=6)
    eng = SimulationEngine(ds=ds)
    before = eng.state.is_adapted.copy()
    eng.run(np.linspace(0, 1.5, 5), no_seq=4, seed=2, n_jobs=4)
    assert np.array_equal(eng.state.is_adapted, before)
