"""
event_utils.py
==============
Single source of truth for stochastic flood-event generation in the
FloodAdapt-ABM x DYNAMO-M coupling.

Two draw modes are provided (selected via
``DecisionConfig.event_draw_mode``):

``"poisson"`` (recommended)
    Each event ``i`` occurs ``n_i ~ Poisson(freq_i * dt)`` times per year.
    Event frequencies are occurrence *rates* (events/year), so the Poisson
    distribution is the statistically exact model: the realised long-run
    rate of every event — rare extremes and sub-annual ``freq > 1`` events
    alike — equals its nominal frequency, and one event may occur several
    times within a year.

``"bernoulli_clip"`` (legacy)
    One Bernoulli trial per event with ``p = min(freq_i * dt, 1)``.
    Frequencies above ``1/dt`` are clipped to certainty, so sub-annual
    events occur every single year and — combined with the
    ``max_events_per_year`` cap — crowd out the rare extremes.  Kept as an
    ordinary option (its RNG call order is frozen) so the Poisson draw can be
    compared against it; do not use it for a real study.

Cap policies
------------
When ``max_events_per_year`` binds, the surplus occurrences are discarded
according to ``cap_policy``:

* ``"largest_damage"`` — keep the most damaging occurrences (requires
  ``event_severity``).  Deterministic: no extra RNG draw, so sequential
  and parallel runs remain bit-identical, and extremes are never discarded
  in favour of nuisance events.
* ``"random"`` — legacy: uniform random selection without replacement.
  Preserves the magnitude distribution *conditional on the drawn pool*
  but biases the marginal occurrence rate of every event downwards.

With the Poisson mode the cap is best left disabled
(``max_events_per_year=None``): realised damage is already bounded by
``max_pot_dmg`` per event, and any discard distorts the hazard statistics.
"""
from __future__ import annotations

import numpy as np

#: Valid values for the ``mode`` argument of :func:`draw_year_events`.
EVENT_DRAW_MODES: tuple[str, ...] = ("poisson", "bernoulli_clip")

#: Valid values for the ``cap_policy`` argument of :func:`draw_year_events`.
CAP_POLICIES: tuple[str, ...] = ("largest_damage", "random")


def draw_year_events(
    event_names: np.ndarray,
    event_freqs: np.ndarray,
    rng: np.random.Generator,
    max_events_per_year: int | None = None,
    dt: float = 1.0,
    mode: str = "poisson",
    cap_policy: str = "largest_damage",
    event_severity: np.ndarray | None = None,
) -> list[str]:
    """
    Draw the flood-event occurrences for a single simulation year.

    Parameters
    ----------
    event_names : np.ndarray
        1-D array of event names (any dtype convertible to ``str``).
    event_freqs : np.ndarray
        1-D array of annual occurrence frequencies (events/year), aligned
        with ``event_names``.
    rng : np.random.Generator
        Seeded generator, used for the stochastic draw (and, under
        ``cap_policy="random"``, the cap subsampling) so runs are
        reproducible.
    max_events_per_year : int or None
        Maximum number of occurrences retained per year.  ``None``
        (default) disables the cap.
    dt : float
        Timestep length in years used to convert per-year frequencies into
        per-step rates/probabilities.  Default ``1.0``.
    mode : str
        ``"poisson"`` or ``"bernoulli_clip"`` (see module docstring).
        Default ``"poisson"``.
    cap_policy : str
        ``"largest_damage"`` or ``"random"`` (see module docstring).
        Default ``"largest_damage"``.
    event_severity : np.ndarray or None
        1-D array (aligned with ``event_names``) ranking events by damage
        (e.g. community gross damage at the current SLR).  Required when
        the cap binds under ``cap_policy="largest_damage"``.

    Returns
    -------
    occurred_events : list[str]
        Names of the event occurrences this year, in dataset order.  Under
        ``mode="poisson"`` the list may contain the same name more than
        once (multiple occurrences of one event within a year); realised
        damages are summed per occurrence downstream.
    """
    names = np.asarray(event_names)
    freqs = np.asarray(event_freqs, dtype=np.float64)
    if names.shape[0] != freqs.shape[0]:
        raise ValueError(
            f"event_names ({names.shape[0]}) and event_freqs "
            f"({freqs.shape[0]}) must have the same length."
        )
    if mode not in EVENT_DRAW_MODES:
        raise ValueError(
            f"Unknown event draw mode {mode!r}; expected one of {EVENT_DRAW_MODES}."
        )
    if cap_policy not in CAP_POLICIES:
        raise ValueError(
            f"Unknown cap policy {cap_policy!r}; expected one of {CAP_POLICIES}."
        )

    if mode == "bernoulli_clip":
        # PRE-REVIEW branch — byte-identical RNG call order (rng.random ->
        # rng.choice -> np.sort).  Keep the stream frozen: notebook 2 relies on
        # it to give R2 and R3 bit-identical flood histories, which is what
        # makes that comparison a controlled one.
        probs = np.clip(freqs * dt, 0.0, 1.0)
        occurred_mask = rng.random(probs.shape[0]) < probs
        occurred_idx = np.flatnonzero(occurred_mask)

        if max_events_per_year is not None and occurred_idx.size > max_events_per_year:
            if cap_policy == "random":
                chosen = rng.choice(
                    occurred_idx, size=int(max_events_per_year), replace=False
                )
                occurred_idx = np.sort(chosen)
            else:
                occurred_idx = _cap_by_severity(
                    occurred_idx, int(max_events_per_year), event_severity
                )
        return [str(names[i]) for i in occurred_idx]

    # -- Poisson branch ----------------------------------------------------
    counts = rng.poisson(freqs * dt)
    # Expand to one entry per occurrence; np.repeat keeps ascending event
    # order, and duplicates encode multiple occurrences of the same event.
    occurred_idx = np.repeat(np.arange(freqs.shape[0]), counts)

    if max_events_per_year is not None and occurred_idx.size > max_events_per_year:
        if cap_policy == "largest_damage":
            occurred_idx = _cap_by_severity(
                occurred_idx, int(max_events_per_year), event_severity
            )
        else:
            pos = rng.choice(
                occurred_idx.size, size=int(max_events_per_year), replace=False
            )
            occurred_idx = np.sort(occurred_idx[pos])

    return [str(names[i]) for i in occurred_idx]


def _cap_by_severity(
    occurred_idx: np.ndarray,
    cap: int,
    event_severity: np.ndarray | None,
) -> np.ndarray:
    """
    Retain the ``cap`` most damaging occurrences, deterministically.

    Ranking is by descending ``event_severity`` with ties broken by
    ascending event index (stable, no RNG involved — sequential and
    parallel executions therefore agree bit-for-bit).  The retained
    occurrences are returned sorted ascending (dataset order).

    Parameters
    ----------
    occurred_idx : np.ndarray[int]
        Drawn occurrence indices into the event catalogue (may contain
        duplicates under the Poisson mode).
    cap : int
        Number of occurrences to keep.
    event_severity : np.ndarray or None
        Per-event damage ranking key; must be provided when this policy is
        invoked.

    Returns
    -------
    kept_idx : np.ndarray[int]
        The retained occurrence indices, sorted ascending.
    """
    if event_severity is None:
        raise ValueError(
            "cap_policy='largest_damage' requires event_severity when the "
            "max_events_per_year cap binds."
        )
    severity = np.asarray(event_severity, dtype=np.float64)[occurred_idx]
    # np.lexsort: LAST key is the primary sort key -> order by descending
    # severity, ties by ascending event index.
    order = np.lexsort((occurred_idx, -severity))
    return np.sort(occurred_idx[order[:cap]])


def generate_event_sequences(
    event_names: np.ndarray,
    event_freqs: np.ndarray,
    n_seq: int,
    n_years: int,
    rng: np.random.Generator,
    max_events_per_year: int | None = None,
    dt: float = 1.0,
    mode: str = "poisson",
    cap_policy: str = "largest_damage",
    event_severity: np.ndarray | None = None,
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
    mode, cap_policy : str
        Draw mode and cap policy (see :func:`draw_year_events`).
    event_severity : np.ndarray or None
        Per-event damage ranking key for ``cap_policy="largest_damage"``.

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
                event_names,
                event_freqs,
                rng,
                max_events_per_year,
                dt,
                mode=mode,
                cap_policy=cap_policy,
                event_severity=event_severity,
            )
            for _ in range(int(n_years))
        ]
        sequences.append(seq)
    return sequences
