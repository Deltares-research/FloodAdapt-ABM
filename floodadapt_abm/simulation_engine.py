"""
simulation_engine.py
====================
Unified agent-based flood-adaptation simulation engine (Phase 2 + Phase 3 of
the step-wise refactoring).

``SimulationEngine`` is the single owner of *time and data*:

* NetCDF loading, residential filtering, per-agent economic arrays and
  per-event damage interpolation (delegated to a composed
  :class:`~floodadapt_abm.dynamo_decision_bridge.DynamoDecisionBridge`, so the
  interpolation kernel is not duplicated),
* the unified stochastic event draw (Bernoulli + ``max_events_per_year`` cap
  with random pool selection, from :mod:`floodadapt_abm.event_utils`),
* per-agent state tracking via :class:`~floodadapt_abm.agent_state.AgentState`
  (``is_adapted``, ``flood_timer``, ``risk_perception``, ``time_adapted``),
* the adaptation-lifespan (dry-proofing) reset, and
* the year loop (``run`` / ``step``).

*Behaviour* — whether each household adapts — is delegated to a pluggable
:class:`~floodadapt_abm.decision_rule.DecisionRule` (``ThresholdRule`` for the
legacy heuristic, ``SEURule`` for the DYNAMO-M science).  Switching rules is the
only change required to move between the legacy 0.3-threshold behaviour and the
validated SEU decision engine.
"""
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import xarray as xr
from numpy import ndarray

from floodadapt_abm.agent_state import AgentState
from floodadapt_abm.coupling_config import CouplingConfig, DecisionConfig
from floodadapt_abm.decision_rule import DecisionRule, SEURule
from floodadapt_abm._core.dynamo_decision_bridge import DynamoDecisionBridge
from floodadapt_abm.event_utils import draw_year_events


class SimulationEngine:
    """
    Single simulation engine with a pluggable decision rule.

    Parameters
    ----------
    ds : xarray.Dataset
        Precomputed damage lookup table (see ``NetCDFMappingConfig``).
    decision_rule : DecisionRule or None
        The behaviour plug-in.  When ``None`` a :class:`SEURule` built from the
        configuration is used (the MVP default).
    config : CouplingConfig or None
        Coupling configuration.  Defaults to ``CouplingConfig()``.
    income_per_agent, amenity_value_per_agent : np.ndarray or None
        Optional per-*residential*-agent economic arrays, forwarded to the data
        layer.  ``None`` triggers the same fallbacks as ``DynamoDecisionBridge``.
    damage_dtype : np.dtype
        Storage dtype for the damage history (default ``np.float32``).  Integer
        dtypes trigger ``np.rint`` rounding, matching ``ABMSimulator``.

    Attributes
    ----------
    n_agents : int
        Number of residential agents.
    state : AgentState
        Live per-agent state (mutated in place by :meth:`step`).
    """

    _RISK_PERC_BASE: float = 1.6  # matches DynamoDecisionBridge / DYNAMO-M

    def __init__(
        self,
        ds: xr.Dataset,
        decision_rule: DecisionRule | None = None,
        config: CouplingConfig | None = None,
        income_per_agent: np.ndarray | None = None,
        amenity_value_per_agent: np.ndarray | None = None,
        damage_dtype: np.dtype = np.float32,
    ) -> None:
        self.config: CouplingConfig = config if config is not None else CouplingConfig()
        self._dec: DecisionConfig = self.config.decision
        self.damage_dtype = damage_dtype

        # -- Data plumbing (composed bridge; interpolation kernel reused) -----
        self._data = DynamoDecisionBridge(
            ds=ds,
            config=self.config,
            income_per_agent=income_per_agent,
            amenity_value_per_agent=amenity_value_per_agent,
        )
        self.n_agents: int = self._data.n_agents
        self.object_ids: np.ndarray = self._data.object_ids
        self.max_pot_dmg: np.ndarray = self._data.max_pot_dmg
        self._event_names: np.ndarray = self._data._event_names
        self._event_freqs: np.ndarray = self._data._event_freqs
        self._annual_adapt_cost: np.ndarray = self._data._annual_adapt_cost
        self._event_index: dict[str, int] = {
            str(name): i for i, name in enumerate(self._event_names)
        }

        # -- Decision rule (behaviour plug-in) --------------------------------
        if decision_rule is None:
            decision_rule = SEURule(
                self._dec,
                rng=np.random.default_rng(self.config.random_seed),
                amenity_value=self._data.amenity_value,
            )
        self.decision_rule: DecisionRule = decision_rule

        # -- Per-agent state --------------------------------------------------
        self.state: AgentState = AgentState.initial(
            n_agents=self.n_agents,
            income=self._data.income,
            wealth=self._data.wealth,
            risk_perc_min=self._dec.risk_perc_min,
        )
        # Monotonic counter bumped by every reset_state(); lets external
        # drivers (e.g. FloodAdaptSLRModel) detect that their view of
        # ``self.state`` has been invalidated by a later reset.
        self.state_epoch: int = 0

    # -----------------------------------------------------------------------
    # Event generation
    # -----------------------------------------------------------------------
    def draw_year_events(self, rng: np.random.Generator, dt: float = 1.0) -> list[str]:
        """
        Draw the flood events for a single year (unified generator).

        Delegates to :func:`floodadapt_abm.event_utils.draw_year_events`, using
        this engine's event catalogue and the configured
        ``max_events_per_year`` cap.
        """
        return draw_year_events(
            self._event_names,
            self._event_freqs,
            rng,
            max_events_per_year=self._dec.max_events_per_year,
            dt=dt,
        )

    # -----------------------------------------------------------------------
    # Damage preparation & realised damage
    # -----------------------------------------------------------------------
    def prepare_damages(
        self, slr_value: float, interp_method: str = "linear"
    ) -> tuple[ndarray | None, ndarray | None]:
        """
        Interpolate the per-event damage catalogues at ``slr_value``.

        Returns
        -------
        damages_no_adapt, damages_adapt : np.ndarray
            Each of shape ``(n_agents, n_events)``.
        """
        self._data.prepare_damage_arrays(slr_value, interp_method)
        return self._data._damage_no_measures, self._data._damage_floodproof

    def _realised_damage(
        self,
        occurred_events: list[str],
        damages_no_adapt: np.ndarray,
        damages_adapt: np.ndarray,
    ) -> np.ndarray:
        """
        Total realised damage per agent for the events that occurred, honouring
        each agent's current adaptation status and capping at ``max_pot_dmg``.
        """
        total = np.zeros(self.n_agents, dtype=np.float64)
        is_adapted = self.state.is_adapted
        for evt in occurred_events:
            idx = self._event_index[evt]
            dmg = np.where(
                is_adapted, damages_adapt[:, idx], damages_no_adapt[:, idx]
            )
            total += np.clip(dmg, 0.0, self.max_pot_dmg)
        return total

    # -----------------------------------------------------------------------
    # State updates
    # -----------------------------------------------------------------------
    def update_flood_experience(self, flooded_agents: np.ndarray) -> None:
        """
        Update ``flood_timer`` and ``risk_perception`` after a year.

        Replicates the DYNAMO-M decay formula
        ``rp = rp_max * 1.6^(coef * flood_timer) + rp_min`` (identical to
        ``DynamoDecisionBridge.update_flood_experience``).
        """
        self.state.flood_timer += 1
        self.state.flood_timer[flooded_agents] = 0
        self.state.risk_perception = (
            self._dec.risk_perc_max
            * (self._RISK_PERC_BASE ** (self._dec.risk_perc_coef * self.state.flood_timer))
            + self._dec.risk_perc_min
        ).astype(np.float32)

    def _apply_lifespan_reset(self) -> np.ndarray:
        """
        Age adaptations by one year and expire those reaching the dry-proofing
        lifespan.

        Increments ``time_adapted`` for currently-adapted agents, then resets
        agents whose age has reached ``lifespan_dryproof`` (they un-adapt and
        become eligible to re-decide this year).  Ported from DYNAMO-M
        ``coastal_nodes.py`` lines 2221-2227.

        Returns
        -------
        expired : np.ndarray[bool]
            Agents whose adaptation expired this year.
        """
        state = self.state
        state.time_adapted[state.is_adapted] += 1

        lifespan = self._dec.lifespan_dryproof
        expired = np.zeros(self.n_agents, dtype=bool)
        if lifespan is not None and lifespan > 0:
            expired = state.is_adapted & (state.time_adapted >= lifespan)
            state.is_adapted[expired] = False
            state.time_adapted[expired] = 0
        return expired

    # -----------------------------------------------------------------------
    # Year loop
    # -----------------------------------------------------------------------
    def step(
        self,
        year_index: int,
        slr_value: float,
        rng: np.random.Generator,
        interp_method: str = "linear",
    ) -> dict:
        """
        Advance the simulation by one year for the live ``self.state``.

        Sequence (mirrors architecture Sec.3.2):
        draw events → realised damage → flood-experience update →
        lifespan reset → rule decision → state bookkeeping.

        Parameters
        ----------
        year_index : int
            0-based year index (passed to the rule for diagnostics).
        slr_value : float
            Sea-level rise for this year (lookup-table SLR unit).
        rng : np.random.Generator
            Generator for this year's event draw.
        interp_method : str
            Damage interpolation kind.

        Returns
        -------
        result : dict
            Per-year outputs: ``occurred_events``, ``damages`` (realised, per
            agent), ``was_flooded``, ``newly_adapted``, ``expired``,
            ``is_adapted`` (snapshot), ``eu_adapt`` / ``eu_do_nothing``
            (``None`` for rules that do not compute them).
        """
        damages_no_adapt, damages_adapt = self.prepare_damages(slr_value, interp_method)

        occurred = self.draw_year_events(rng)
        realised = self._realised_damage(occurred, damages_no_adapt, damages_adapt)
        was_flooded = realised > 0

        # Flood experience (risk-perception update) before the decision.
        self.update_flood_experience(was_flooded)

        # Age & expire adaptations (lifespan-dryproof reset).
        expired = self._apply_lifespan_reset()

        # Decision.
        newly_adapted = self.decision_rule.should_adapt(
            agent_state=self.state,
            damages_this_year=realised.astype(np.float32),
            damages_no_adapt=damages_no_adapt,
            damages_adapt=damages_adapt,
            event_freqs=self._event_freqs,
            max_pot_dmg=self.max_pot_dmg,
            adaptation_costs=self._annual_adapt_cost,
        )

        # Bookkeeping: newly adapted agents start a fresh adaptation age.
        self.state.is_adapted[newly_adapted] = True
        self.state.time_adapted[newly_adapted] = 0

        if np.issubdtype(self.damage_dtype, np.integer):
            realised_store = np.rint(realised).astype(self.damage_dtype)
        else:
            realised_store = realised.astype(self.damage_dtype)

        return {
            "year_index": year_index,
            "occurred_events": occurred,
            "damages": realised_store,
            "was_flooded": was_flooded,
            "newly_adapted": newly_adapted,
            "expired": expired,
            "is_adapted": self.state.is_adapted.copy(),
            "eu_adapt": getattr(self.decision_rule, "last_eu_adapt", None),
            "eu_do_nothing": getattr(self.decision_rule, "last_eu_do_nothing", None),
        }

    def reset_state(self) -> None:
        """
        Reset per-agent state to the initial condition (fresh sequence).

        Increments :attr:`state_epoch` so that any driver holding a view of the
        previous state (e.g. an earlier ``FloodAdaptSLRModel``) can detect it
        has gone stale instead of silently stepping the wrong state.
        """
        self.state = AgentState.initial(
            n_agents=self.n_agents,
            income=self._data.income,
            wealth=self._data.wealth,
            risk_perc_min=self._dec.risk_perc_min,
        )
        self.state_epoch += 1

    def run(
        self,
        slr_values,
        no_seq: int = 1,
        seed: int | None = None,
        interp_method: str = "linear",
        track_eu: bool = False,
        n_jobs: int = 1,
    ) -> dict:
        """
        Run the engine over ``no_seq`` Monte-Carlo sequences.

        Time progression is owned here (not by an external demo loop).  Each
        sequence starts from a fresh :class:`AgentState`.

        Parameters
        ----------
        slr_values : array-like
            Per-year SLR trajectory (length ``n_years``).  Time is driven by
            the length of this array — derive it from the FloodAdapt SLR
            projection metadata rather than hard-coding a horizon.
        no_seq : int
            Number of Monte-Carlo sequences.
        seed : int or None
            Base RNG seed.  When ``None`` uses ``config.random_seed``.
        interp_method : str
            Damage interpolation kind.
        track_eu : bool
            When ``True`` and the rule exposes them, records the per-year
            expected utilities into ``eu_history`` (memory: ``2 * no_seq *
            n_agents * n_years``).
        n_jobs : int
            Number of parallel workers for the Monte-Carlo sequences.  ``1``
            (default) uses the sequential path unchanged.  ``>1`` (or ``-1`` for
            "all cores") runs the independent sequences across a thread pool of
            per-worker engine clones that share the read-only, pre-warmed SLR
            interpolation cache.  Because each sequence uses an independently
            seeded event RNG (``base_seed + s``) and writes its own slice, the
            parallel result is **bit-identical** to ``n_jobs=1`` for decision
            rules whose output depends only on the agent state and the event
            draw (the default ``SEURule`` with ``error_interval == 0`` and
            ``ThresholdRule``).

        Returns
        -------
        results : dict
            ``damage_history`` (``no_seq, n_agents, n_years``),
            ``adapted_history`` (bool, same shape),
            ``adoption_fraction`` (``no_seq, n_years``),
            optionally ``eu_adapt_history`` / ``eu_do_nothing_history``.
        """
        slr_values = np.asarray(slr_values, dtype=float)
        n_years = slr_values.shape[0]
        base_seed = seed if seed is not None else self.config.random_seed

        damage_history = np.zeros(
            (no_seq, self.n_agents, n_years), dtype=self.damage_dtype
        )
        adapted_history = np.zeros(
            (no_seq, self.n_agents, n_years), dtype=bool
        )
        eu_adapt_history = (
            np.full((no_seq, self.n_agents, n_years), np.nan, dtype=np.float32)
            if track_eu else None
        )
        eu_do_nothing_history = (
            np.full((no_seq, self.n_agents, n_years), np.nan, dtype=np.float32)
            if track_eu else None
        )

        def _store(s: int, out: dict) -> None:
            damage_history[s] = out["damages"]
            adapted_history[s] = out["adapted"]
            if track_eu and out["eu_adapt"] is not None:
                eu_adapt_history[s] = out["eu_adapt"]
                eu_do_nothing_history[s] = out["eu_do_nothing"]

        if n_jobs == 1:
            # Sequential path (unchanged): one engine, rule RNG threads across
            # sequences exactly as before.
            for s in range(no_seq):
                self.reset_state()
                rng = np.random.default_rng(base_seed + s)
                for t in range(n_years):
                    res = self.step(t, float(slr_values[t]), rng, interp_method)
                    damage_history[s, :, t] = res["damages"]
                    adapted_history[s, :, t] = res["is_adapted"]
                    if track_eu and res["eu_adapt"] is not None:
                        eu_adapt_history[s, :, t] = res["eu_adapt"]
                        eu_do_nothing_history[s, :, t] = res["eu_do_nothing"]
        else:
            # Parallel path: independent sequences on per-worker engine clones
            # that share a pre-warmed, read-only SLR interpolation cache.
            self._prewarm_interp_cache(slr_values, interp_method)
            workers = no_seq if n_jobs < 0 else min(n_jobs, no_seq)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        self._simulate_one_sequence,
                        s, slr_values, base_seed, n_years,
                        interp_method, track_eu,
                    ): s
                    for s in range(no_seq)
                }
                for fut in futures:
                    s = futures[fut]
                    _store(s, fut.result())

        adoption_fraction = adapted_history.mean(axis=1)
        results = {
            "damage_history": damage_history,
            "adapted_history": adapted_history,
            "adoption_fraction": adoption_fraction,
        }
        if track_eu:
            results["eu_adapt_history"] = eu_adapt_history
            results["eu_do_nothing_history"] = eu_do_nothing_history
        return results

    # -----------------------------------------------------------------------
    # Parallel-sequence helpers
    # -----------------------------------------------------------------------
    def _prewarm_interp_cache(self, slr_values, interp_method: str) -> None:
        """
        Interpolate each unique SLR value once (single-threaded) so the shared
        bridge cache is fully populated before parallel workers read from it.
        """
        for slr in np.unique(np.asarray(slr_values, dtype=float)):
            self.prepare_damages(float(slr), interp_method)

    def _clone_for_worker(self, seq_index: int, base_seed: int) -> "SimulationEngine":
        """
        Shallow-copy this engine for a single parallel sequence.

        The clone shares the read-only dataset, event catalogue and (pre-warmed)
        SLR interpolation cache, but gets its own bridge ``__dict__`` (so the
        per-tick ``_damage_*`` slots don't race), its own fresh
        :class:`AgentState`, and an independently seeded clone of the decision
        rule.
        """
        eng = copy.copy(self)
        eng._data = copy.copy(self._data)  # separate _damage_* slots; shared cache
        eng.decision_rule = self.decision_rule.clone(rng_seed=base_seed + seq_index)
        eng.state = AgentState.initial(
            n_agents=self.n_agents,
            income=self._data.income,
            wealth=self._data.wealth,
            risk_perc_min=self._dec.risk_perc_min,
        )
        eng.state_epoch = 0
        return eng

    def _simulate_one_sequence(
        self,
        seq_index: int,
        slr_values: np.ndarray,
        base_seed: int,
        n_years: int,
        interp_method: str,
        track_eu: bool,
    ) -> dict:
        """Run one Monte-Carlo sequence on an isolated engine clone."""
        eng = self._clone_for_worker(seq_index, base_seed)
        rng = np.random.default_rng(base_seed + seq_index)
        damages = np.zeros((self.n_agents, n_years), dtype=self.damage_dtype)
        adapted = np.zeros((self.n_agents, n_years), dtype=bool)
        eu_adapt = (
            np.full((self.n_agents, n_years), np.nan, dtype=np.float32)
            if track_eu else None
        )
        eu_do_nothing = (
            np.full((self.n_agents, n_years), np.nan, dtype=np.float32)
            if track_eu else None
        )
        for t in range(n_years):
            res = eng.step(t, float(slr_values[t]), rng, interp_method)
            damages[:, t] = res["damages"]
            adapted[:, t] = res["is_adapted"]
            if track_eu and res["eu_adapt"] is not None:
                eu_adapt[:, t] = res["eu_adapt"]
                eu_do_nothing[:, t] = res["eu_do_nothing"]
        return {
            "damages": damages,
            "adapted": adapted,
            "eu_adapt": eu_adapt,
            "eu_do_nothing": eu_do_nothing,
        }


    # -----------------------------------------------------------------------
    # Property
    # -----------------------------------------------------------------------
    @property
    def is_residential(self) -> np.ndarray:
        """
        Boolean mask (all ``True``) — the engine already operates on the
        residential subset.  Exposed for parity with the architecture's
        ``is_residential`` engine-state field and for downstream filtering.
        """
        return np.ones(self.n_agents, dtype=bool)
