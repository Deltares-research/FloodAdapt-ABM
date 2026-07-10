"""
test_event_utils.py
===================
Unit tests for the unified stochastic event generator
(``floodadapt_abm.event_utils``), the Phase-2 single source of truth for the
Bernoulli draw + ``max_events_per_year`` cap.
"""
from __future__ import annotations

import numpy as np
import pytest

from floodadapt_abm.event_utils import draw_year_events, generate_event_sequences


EVENT_NAMES = np.array([f"RP{rp:04d}" for rp in (2, 5, 10, 25, 50, 100, 500)])
EVENT_FREQS = 1.0 / np.array([2, 5, 10, 25, 50, 100, 500], dtype=float)


def test_draw_is_reproducible():
    """Same seed → identical draw."""
    a = draw_year_events(EVENT_NAMES, EVENT_FREQS, np.random.default_rng(1))
    b = draw_year_events(EVENT_NAMES, EVENT_FREQS, np.random.default_rng(1))
    assert a == b


def test_draw_returns_subset_of_catalogue():
    rng = np.random.default_rng(3)
    for _ in range(50):
        occ = draw_year_events(EVENT_NAMES, EVENT_FREQS, rng)
        assert set(occ).issubset(set(EVENT_NAMES.astype(str)))


def test_cap_limits_event_count():
    """With a certain-occurrence catalogue, the cap bounds the count exactly."""
    freqs = np.ones(7)  # every event occurs with prob 1
    rng = np.random.default_rng(0)
    occ = draw_year_events(EVENT_NAMES, freqs, rng, max_events_per_year=3)
    assert len(occ) == 3


def test_no_cap_returns_all_when_certain():
    freqs = np.ones(7)
    rng = np.random.default_rng(0)
    occ = draw_year_events(EVENT_NAMES, freqs, rng, max_events_per_year=None)
    assert len(occ) == 7


def test_cap_selection_is_random_not_frequency_ordered():
    """
    The cap must use RANDOM selection from the drawn pool, not 'keep the most
    frequent'.  Over many draws with all-certain events and cap=1, every event
    index should be selected at least once (a frequency-ordered policy would
    only ever keep the first).
    """
    freqs = np.ones(7)
    seen: set[str] = set()
    for s in range(200):
        occ = draw_year_events(
            EVENT_NAMES, freqs, np.random.default_rng(s), max_events_per_year=1
        )
        assert len(occ) == 1
        seen.update(occ)
    # Expect many distinct events retained (random), not a single fixed one.
    assert len(seen) >= 5


def test_zero_frequencies_never_occur():
    occ = draw_year_events(
        EVENT_NAMES, np.zeros(7), np.random.default_rng(5)
    )
    assert occ == []


def test_dt_scales_probability():
    """dt=0 → probabilities collapse to 0 → no events."""
    occ = draw_year_events(
        EVENT_NAMES, EVENT_FREQS, np.random.default_rng(9), dt=0.0
    )
    assert occ == []


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        draw_year_events(EVENT_NAMES, EVENT_FREQS[:-1], np.random.default_rng(0))


def test_generate_sequences_shape():
    seqs = generate_event_sequences(
        EVENT_NAMES, EVENT_FREQS, n_seq=4, n_years=10,
        rng=np.random.default_rng(2), max_events_per_year=4,
    )
    assert len(seqs) == 4
    assert all(len(s) == 10 for s in seqs)


def test_generate_sequences_reproducible():
    a = generate_event_sequences(
        EVENT_NAMES, EVENT_FREQS, 3, 5, np.random.default_rng(7)
    )
    b = generate_event_sequences(
        EVENT_NAMES, EVENT_FREQS, 3, 5, np.random.default_rng(7)
    )
    assert a == b


def test_cap_respected_across_sequences():
    freqs = np.ones(7)
    seqs = generate_event_sequences(
        EVENT_NAMES, freqs, 5, 8, np.random.default_rng(0),
        max_events_per_year=2,
    )
    for seq in seqs:
        for year in seq:
            assert len(year) <= 2
