"""
event_utils.py
==============
Single source of truth for stochastic flood-event generation in the
FloodAdapt-ABM x DYNAMO-M coupling (Phase 2 consolidation).

Historically the Bernoulli event draw plus the ``max_events_per_year`` cap was
duplicated in three places:

* ``ABMSimulator.generate_event_sequences`` (no cap),
* ``example/run_coupled_example.py::_simulate_year_events`` (magnitude/random cap),
* ``example/run_trace_manual_check.py`` (its own inline copy).

This module provides one vectorised implementation used by ``SimulationEngine``
and re-exported for the example scripts, so the drawing semantics live in
exactly one place.

Cap-selection
--------------------
When the independent Bernoulli trials yield more than ``max_events_per_year``
events in a single year, the surplus is resolved by **random selection without
replacement** from the drawn pool.  This preserves the Monte-Carlo distribution
of event magnitudes (it does not bias toward the most frequent or the most
damaging events).  See ``coupling_config.DecisionConfig.max_events_per_year``.
"""
from __future__ import annotations

import numpy as np


def draw_year_events(
    event_names: np.ndarray,
    event_freqs: np.ndarray,
    rng: np.random.Generator,
    max_events_per_year: int | None = None,
    dt: float = 1.0,
) -> list[str]:
    """
    Draw the flood events that occur in a single simulation year.

    Each event ``i`` occurs independently with probability
    ``min(freq_i * dt, 1.0)`` (a Bernoulli trial).  If more than
    ``max_events_per_year`` events are drawn, a random subset of exactly
    ``max_events_per_year`` events is retained (random selection without
    replacement).

    Parameters
    ----------
    event_names : np.ndarray
        1-D array of event names (any dtype convertible to ``str``).
    event_freqs : np.ndarray
        1-D array of annual exceedance frequencies (events/year), aligned with
        ``event_names``.
    rng : np.random.Generator
        Seeded generator, used for both the Bernoulli trials and the cap
        subsampling so runs are reproducible.
    max_events_per_year : int or None
        Maximum number of events retained per year.  ``None`` (default)
        disables the cap.
    dt : float
        Timestep length in years used to convert per-year frequencies into
        per-step occurrence probabilities.  Default ``1.0``.

    Returns
    -------
    occurred_events : list[str]
        Names of the events that occurred this year, in dataset order (the cap
        subsample is returned sorted by original index for determinism).
    """
    names = np.asarray(event_names)
    freqs = np.asarray(event_freqs, dtype=np.float64)
    if names.shape[0] != freqs.shape[0]:
        raise ValueError(
            f"event_names ({names.shape[0]}) and event_freqs "
            f"({freqs.shape[0]}) must have the same length."
        )

    probs = np.clip(freqs * dt, 0.0, 1.0)
    occurred_mask = rng.random(probs.shape[0]) < probs
    occurred_idx = np.flatnonzero(occurred_mask)

    if max_events_per_year is not None and occurred_idx.size > max_events_per_year:
        chosen = rng.choice(
            occurred_idx, size=int(max_events_per_year), replace=False
        )
        occurred_idx = np.sort(chosen)

    return [str(names[i]) for i in occurred_idx]


def generate_event_sequences(
    event_names: np.ndarray,
    event_freqs: np.ndarray,
    n_seq: int,
    n_years: int,
    rng: np.random.Generator,
    max_events_per_year: int | None = None,
    dt: float = 1.0,
) -> list[list[list[str]]]:
    """
    Generate ``n_seq`` independent Monte-Carlo event sequences.

    Each sequence is a list of ``n_years`` per-year event lists, produced by
    repeated calls to :func:`draw_year_events` with a shared ``rng`` (so the
    whole batch is reproducible from a single seed).

    Parameters
    ----------
    event_names, event_freqs : np.ndarray
        Event catalogue, as in :func:`draw_year_events`.
    n_seq : int
        Number of Monte-Carlo sequences.
    n_years : int
        Number of years per sequence.
    rng : np.random.Generator
        Seeded generator.
    max_events_per_year : int or None
        Per-year cap (see :func:`draw_year_events`).
    dt : float
        Timestep length in years.

    Returns
    -------
    sequences : list[list[list[str]]]
        ``sequences[s][y]`` is the list of event names occurring in sequence
        ``s``, year ``y``.
    """
    sequences: list[list[list[str]]] = []
    for _ in range(int(n_seq)):
        seq: list[list[str]] = [
            draw_year_events(
                event_names, event_freqs, rng, max_events_per_year, dt
            )
            for _ in range(int(n_years))
        ]
        sequences.append(seq)
    return sequences
