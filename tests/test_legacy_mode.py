"""
test_legacy_mode.py
===================
The master parity wall for the event/perception/income/insurance refactor.

``tests/data/golden_legacy_mock.npz`` was captured from the UNMODIFIED
kernels (before the Poisson event draw, severity-aware perception,
synthetic-income, and insurance changes landed) running under
``CouplingConfig.legacy()``.  These tests assert that the legacy preset
keeps reproducing those outputs **bit-exactly** — every subsequent phase
of the refactor must leave them green.

If one of these tests fails, a behaviour change has leaked into the
legacy code path; fix the leak rather than re-capturing the golden file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from floodadapt_abm import (
    CouplingConfig,
    DecisionConfig,
    SEURule,
    SimulationEngine,
    ThresholdRule,
)
from tests.conftest import make_mock_dataset

GOLDEN_PATH = Path(__file__).parent / "data" / "golden_legacy_mock.npz"

SEED = 123
N_YEARS = 10
NO_SEQ = 2


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    """Load the golden reference arrays captured from the pre-change kernels."""
    with np.load(GOLDEN_PATH) as data:
        return {key: data[key] for key in data.files}


def _slr(golden: dict[str, np.ndarray]) -> np.ndarray:
    return golden["slr"]


def test_legacy_preset_pins_all_mode_switches() -> None:
    """legacy() must pin every behaviour switch to its historical value."""
    dec = DecisionConfig.legacy()
    assert dec.event_draw_mode == "bernoulli_clip"
    assert dec.nuisance_freq_threshold is None
    assert dec.max_events_per_year == 4
    assert dec.cap_policy == "random"
    assert dec.seu_prob_mode == "raw_freq"
    assert dec.perception_mode == "binary"
    assert dec.flood_significance_threshold == 0.0
    assert dec.income_mode == "mpd_ratio"
    assert dec.adaptation_total_cost is None
    assert dec.include_insurance is False
    # Composite wrapper keeps the historical seed default.
    assert CouplingConfig.legacy().random_seed == 42


def test_seu_run_bit_matches_golden(golden: dict[str, np.ndarray]) -> None:
    """SimulationEngine + SEURule under legacy() reproduces the golden run."""
    engine = SimulationEngine(ds=make_mock_dataset(), config=CouplingConfig.legacy())
    assert isinstance(engine.decision_rule, SEURule)
    res = engine.run(_slr(golden), no_seq=NO_SEQ, seed=SEED, track_eu=True)

    assert np.array_equal(res["damage_history"], golden["seu_damage_history"])
    assert np.array_equal(res["adapted_history"], golden["seu_adapted_history"])
    assert np.array_equal(res["adoption_fraction"], golden["seu_adoption_fraction"])
    assert np.array_equal(
        res["eu_adapt_history"], golden["seu_eu_adapt_history"], equal_nan=True
    )
    assert np.array_equal(
        res["eu_do_nothing_history"],
        golden["seu_eu_do_nothing_history"],
        equal_nan=True,
    )


def test_threshold_run_bit_matches_golden(golden: dict[str, np.ndarray]) -> None:
    """SimulationEngine + ThresholdRule under legacy() reproduces the golden run."""
    cfg = CouplingConfig.legacy()
    engine = SimulationEngine(
        ds=make_mock_dataset(),
        decision_rule=ThresholdRule(cfg.decision, damage_threshold=0.3),
        config=cfg,
    )
    res = engine.run(_slr(golden), no_seq=NO_SEQ, seed=SEED)

    assert np.array_equal(res["damage_history"], golden["thr_damage_history"])
    assert np.array_equal(res["adapted_history"], golden["thr_adapted_history"])
    assert np.array_equal(res["adoption_fraction"], golden["thr_adoption_fraction"])
