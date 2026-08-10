"""
test_hazard_semantics.py
========================
Bridge/engine-level tests for the hazard-sampling fix:

* the nuisance-event filter (``nuisance_freq_threshold``) slices the event
  catalogue AND the interpolated damage matrices consistently,
* ``p_floods_seu`` implements both probability modes correctly,
* the engine runs end-to-end (reproducibly) under the Poisson draw mode.
"""
from __future__ import annotations

import numpy as np
import pytest

from floodadapt_abm import CouplingConfig, SimulationEngine
from tests.conftest import make_mock_dataset, historical_modes_config


def _ds_with_freqs(freqs: np.ndarray):
    """Mock dataset whose event frequencies are overridden with ``freqs``."""
    ds = make_mock_dataset(n_events=len(freqs))
    ds["event"].attrs["freq"] = np.asarray(freqs, dtype=np.float64)
    return ds


# ---------------------------------------------------------------------------
# Nuisance filter
# ---------------------------------------------------------------------------

def test_nuisance_filter_slices_catalogue_and_damage_matrices():
    """Names, freqs, and damage-matrix columns all shrink together."""
    freqs = np.array([0.01, 0.2, 1.0, 1.5, 3.0, 12.0])
    ds = _ds_with_freqs(freqs)

    cfg = historical_modes_config()
    cfg.decision.nuisance_freq_threshold = 1.0
    engine = SimulationEngine(ds=ds, config=cfg)

    kept = freqs <= 1.0
    assert engine._event_freqs.shape == (kept.sum(),)
    assert np.array_equal(engine._event_freqs, freqs[kept])
    assert len(engine._event_names) == kept.sum()
    # The event index only knows the surviving events.
    assert set(engine._event_index) == set(str(n) for n in engine._event_names)

    # Interpolated damage matrices are sliced consistently with the catalogue.
    dmg_no, dmg_fp = engine.prepare_damages(0.5)
    assert dmg_no.shape == (engine.n_agents, kept.sum())
    assert dmg_fp.shape == (engine.n_agents, kept.sum())

    # The retained columns match the unfiltered engine's matching columns.
    cfg_all = historical_modes_config()
    engine_all = SimulationEngine(ds=_ds_with_freqs(freqs), config=cfg_all)
    dmg_no_all, _ = engine_all.prepare_damages(0.5)
    assert np.array_equal(dmg_no, dmg_no_all[:, kept])


def test_nuisance_filter_none_keeps_everything():
    freqs = np.array([0.01, 0.2, 1.0, 1.5, 3.0, 12.0])
    engine = SimulationEngine(
        ds=_ds_with_freqs(freqs), config=historical_modes_config()
    )
    assert engine._event_freqs.shape == (6,)


# ---------------------------------------------------------------------------
# SEU probability modes
# ---------------------------------------------------------------------------

def test_p_floods_seu_exceedance_formula():
    """exceedance mode: p = 1 - exp(-freq); stays < 1 even for freq > 1."""
    freqs = np.array([0.01, 0.5, 1.0, 3.0, 12.0])
    cfg = historical_modes_config()
    cfg.decision.seu_prob_mode = "exceedance"
    engine = SimulationEngine(ds=_ds_with_freqs(freqs), config=cfg)

    p = engine._data.p_floods_seu
    assert np.allclose(p, 1.0 - np.exp(-freqs))
    assert np.all(p < 1.0)
    # Rare-event limit: p ~= freq for small rates.
    assert abs(p[0] - freqs[0]) < 1e-4


def test_p_floods_seu_raw_matches_legacy():
    freqs = np.array([0.01, 0.5, 1.0, 3.0])
    engine = SimulationEngine(
        ds=_ds_with_freqs(freqs), config=historical_modes_config()
    )
    assert np.array_equal(engine._data.p_floods_seu, freqs)


def test_p_floods_seu_unknown_mode_raises():
    cfg = historical_modes_config()
    cfg.decision.seu_prob_mode = "bogus"
    engine = SimulationEngine(ds=make_mock_dataset(), config=cfg)
    with pytest.raises(ValueError, match="seu_prob_mode"):
        _ = engine._data.p_floods_seu


# ---------------------------------------------------------------------------
# End-to-end Poisson engine run
# ---------------------------------------------------------------------------

def _new_mode_config() -> CouplingConfig:
    cfg = historical_modes_config()
    cfg.decision.event_draw_mode = "poisson"
    cfg.decision.max_events_per_year = None
    cfg.decision.cap_policy = "largest_damage"
    cfg.decision.seu_prob_mode = "exceedance"
    return cfg


def test_engine_runs_end_to_end_under_poisson_mode():
    slr = np.linspace(0.0, 1.0, 5)
    engine = SimulationEngine(ds=make_mock_dataset(), config=_new_mode_config())
    res = engine.run(slr, no_seq=2, seed=11)
    assert res["damage_history"].shape == (2, engine.n_agents, 5)
    assert np.isfinite(res["adoption_fraction"]).all()


def test_engine_poisson_mode_reproducible():
    slr = np.linspace(0.0, 1.0, 5)
    a = SimulationEngine(ds=make_mock_dataset(), config=_new_mode_config()).run(
        slr, no_seq=2, seed=11
    )
    b = SimulationEngine(ds=make_mock_dataset(), config=_new_mode_config()).run(
        slr, no_seq=2, seed=11
    )
    assert np.array_equal(a["damage_history"], b["damage_history"])
    assert np.array_equal(a["adapted_history"], b["adapted_history"])


def test_engine_poisson_cap_binds_with_severity_ranking():
    """A binding cap under largest_damage runs without extra severity input."""
    freqs = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    cfg = _new_mode_config()
    cfg.decision.max_events_per_year = 2
    engine = SimulationEngine(ds=_ds_with_freqs(freqs), config=cfg)
    res = engine.run(np.linspace(0.0, 1.0, 4), no_seq=1, seed=3)
    assert res["damage_history"].shape[2] == 4
