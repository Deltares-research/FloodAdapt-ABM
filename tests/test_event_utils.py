"""
test_event_utils.py
===================
Unit tests for the unified stochastic event generator
(``floodadapt_abm.event_utils``).

Two families of tests:

* **Clipped-draw spec** — the Bernoulli-clip tests, pinned
  explicitly to ``mode="bernoulli_clip"`` / ``cap_policy="random"``.  They
  define the frozen clip semantics that ``historical_modes_config()`` and
  the golden regression rely on.
* **Poisson-mode tests** — the statistically-correct draw, including the
  rate-recovery regression that fails by construction under the clipped
  clip+cap behaviour.
"""
from __future__ import annotations

import numpy as np
import pytest

from floodadapt_abm.event_utils import draw_year_events, generate_event_sequences


EVENT_NAMES = np.array([f"RP{rp:04d}" for rp in (2, 5, 10, 25, 50, 100, 500)])
EVENT_FREQS = 1.0 / np.array([2, 5, 10, 25, 50, 100, 500], dtype=float)

#: kwargs pinning the frozen clip-and-cap draw semantics.
CLIP_CAP = dict(mode="bernoulli_clip", cap_policy="random")


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
    """Clipped draw: with a certain-occurrence catalogue, the cap binds exactly."""
    freqs = np.ones(7)  # every event occurs with prob 1 under bernoulli_clip
    rng = np.random.default_rng(0)
    occ = draw_year_events(
        EVENT_NAMES, freqs, rng, max_events_per_year=3, **CLIP_CAP
    )
    assert len(occ) == 3


def test_no_cap_returns_all_when_certain():
    """Clipped draw: freq >= 1 becomes certainty."""
    freqs = np.ones(7)
    rng = np.random.default_rng(0)
    occ = draw_year_events(
        EVENT_NAMES, freqs, rng, max_events_per_year=None, **CLIP_CAP
    )
    assert len(occ) == 7


def test_cap_selection_is_random_not_frequency_ordered():
    """
    Uniform cap policy: RANDOM selection from the drawn pool, not 'keep the
    most frequent'.  Over many draws with all-certain events and cap=1, many
    distinct event indices should be retained (a frequency-ordered policy
    would only ever keep the first).
    """
    freqs = np.ones(7)
    seen: set[str] = set()
    for s in range(200):
        occ = draw_year_events(
            EVENT_NAMES, freqs, np.random.default_rng(s),
            max_events_per_year=1, **CLIP_CAP,
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
        rng=np.random.default_rng(2), max_events_per_year=4, **CLIP_CAP,
    )
    assert len(seqs) == 4
    assert all(len(s) == 10 for s in seqs)


def test_generate_sequences_reproducible():
    a = generate_event_sequences(
        EVENT_NAMES, EVENT_FREQS, 3, 5, np.random.default_rng(7), **CLIP_CAP
    )
    b = generate_event_sequences(
        EVENT_NAMES, EVENT_FREQS, 3, 5, np.random.default_rng(7), **CLIP_CAP
    )
    assert a == b


def test_cap_respected_across_sequences():
    freqs = np.ones(7)
    seqs = generate_event_sequences(
        EVENT_NAMES, freqs, 5, 8, np.random.default_rng(0),
        max_events_per_year=2, **CLIP_CAP,
    )
    for seq in seqs:
        for year in seq:
            assert len(year) <= 2


# ---------------------------------------------------------------------------
# Poisson mode
# ---------------------------------------------------------------------------

def test_poisson_reproducible():
    """Same seed -> identical Poisson draw."""
    a = draw_year_events(
        EVENT_NAMES, EVENT_FREQS, np.random.default_rng(1), mode="poisson"
    )
    b = draw_year_events(
        EVENT_NAMES, EVENT_FREQS, np.random.default_rng(1), mode="poisson"
    )
    assert a == b


def test_poisson_allows_multiple_occurrences_per_year():
    """A sub-annual event (freq=3) legitimately occurs >1 time in a year."""
    names = np.array(["nuisance"])
    freqs = np.array([3.0])
    rng = np.random.default_rng(0)
    counts = [
        len(draw_year_events(names, freqs, rng, mode="poisson"))
        for _ in range(50)
    ]
    assert max(counts) >= 2          # duplicates happen
    assert 2.0 < np.mean(counts) < 4.0   # mean count ~ freq


def test_poisson_rate_recovery_vs_clipped_draw():
    """
    Regression: with Poisson + no cap, every event's realised occurrence
    rate matches its nominal frequency — including a rare extreme sharing
    the catalogue with many sub-annual nuisance events.  Under the clipped
    clip+cap draw the same extreme is silently discarded most of the years
    it occurs (its realised rate collapses).
    """
    n_years = 5_000
    # 1 rare extreme (freq 0.01) + 11 sub-annual nuisance events (freq 1.5),
    # mirroring the real Charleston catalogue's structure.
    names = np.array(["extreme"] + [f"nuis_{i}" for i in range(11)])
    freqs = np.array([0.01] + [1.5] * 11)

    def realised_extreme_rate(mode: str, cap: int | None) -> float:
        rng = np.random.default_rng(2026)
        hits = 0
        for _ in range(n_years):
            occ = draw_year_events(
                names, freqs, rng, max_events_per_year=cap,
                mode=mode, cap_policy="random",
            )
            hits += occ.count("extreme")
        return hits / n_years

    # Poisson, no cap: rate ~= nominal 0.01 (4-sigma band: 50 +/- 28 hits).
    poisson_rate = realised_extreme_rate("poisson", cap=None)
    assert 22 / n_years <= poisson_rate <= 78 / n_years

    # Clip + cap of 5: 11 events are guaranteed every year, so when
    # the extreme is drawn it survives the uniform 5-of-12 subsample with
    # probability ~5/12 -> realised rate collapses well below nominal.
    clipped_rate = realised_extreme_rate("bernoulli_clip", cap=5)
    assert clipped_rate < 0.7 * 0.01

    # And the nuisance events' mean count per year is preserved by Poisson.
    rng = np.random.default_rng(7)
    total = sum(
        len(draw_year_events(names, freqs, rng, mode="poisson"))
        for _ in range(2_000)
    )
    expected = freqs.sum() * 2_000
    assert abs(total - expected) / expected < 0.05


def test_largest_damage_cap_is_deterministic_and_keeps_extremes():
    """
    The severity cap keeps exactly the top-damage occurrences of the drawn
    pool and consumes NO extra RNG: the capped draw with a given seed equals
    the top-K (by severity, ties by index) of the uncapped draw with the
    same seed.
    """
    names = np.array(["small", "medium", "huge"])
    freqs = np.array([5.0, 5.0, 5.0])
    severity = np.array([1.0, 10.0, 100.0])
    cap = 2

    # Uncapped reference draw: same seed -> identical Poisson occurrences,
    # because the severity cap performs no RNG call.
    uncapped = draw_year_events(
        names, freqs, np.random.default_rng(3), max_events_per_year=None,
        mode="poisson",
    )
    assert len(uncapped) > cap  # the cap must actually bind in this test

    capped_a = draw_year_events(
        names, freqs, np.random.default_rng(3), max_events_per_year=cap,
        mode="poisson", cap_policy="largest_damage", event_severity=severity,
    )
    capped_b = draw_year_events(
        names, freqs, np.random.default_rng(3), max_events_per_year=cap,
        mode="poisson", cap_policy="largest_damage", event_severity=severity,
    )
    assert capped_a == capped_b  # deterministic

    # Expected: top-`cap` of the uncapped pool by (severity desc, index asc),
    # returned in dataset order.
    sev_by_name = dict(zip(names.tolist(), severity.tolist()))
    idx_by_name = {n: i for i, n in enumerate(names.tolist())}
    ranked = sorted(
        uncapped, key=lambda n: (-sev_by_name[n], idx_by_name[n])
    )[:cap]
    expected = sorted(ranked, key=lambda n: idx_by_name[n])
    assert capped_a == expected
    # The most damaging drawn event is never discarded.
    assert "huge" in capped_a


def test_poisson_random_cap_policy_available():
    freqs = np.full(7, 2.0)
    occ = draw_year_events(
        EVENT_NAMES, freqs, np.random.default_rng(0),
        max_events_per_year=3, mode="poisson", cap_policy="random",
    )
    assert len(occ) == 3


def test_unknown_mode_and_policy_raise():
    with pytest.raises(ValueError, match="draw mode"):
        draw_year_events(
            EVENT_NAMES, EVENT_FREQS, np.random.default_rng(0), mode="bogus"
        )
    with pytest.raises(ValueError, match="cap policy"):
        draw_year_events(
            EVENT_NAMES, EVENT_FREQS, np.random.default_rng(0),
            cap_policy="bogus",
        )


def test_largest_damage_cap_requires_severity():
    freqs = np.full(7, 2.0)
    with pytest.raises(ValueError, match="event_severity"):
        draw_year_events(
            EVENT_NAMES, freqs, np.random.default_rng(0),
            max_events_per_year=2, mode="poisson",
            cap_policy="largest_damage", event_severity=None,
        )
