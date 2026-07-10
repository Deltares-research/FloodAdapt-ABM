"""
mesa_native.py
==============
Phase 4b of the FloodAdapt-ABM x DYNAMO-M coupling: **Mesa-native driving**.

Where Phase 3 / 4a let :class:`~floodadapt_abm.simulation_engine.SimulationEngine`
own the year loop (``engine.run()`` iterates the sequences and the years), Phase
4b **inverts time ownership**: a small model object advances one tick at a time
via :meth:`FloodAdaptSLRModel.step`, mirroring how the native DYNAMO-M
``SLRModel.run_model()`` drives ticks with ``while True: self.step()``.

Why a framework-free mirror (and not the real ``honeybees`` ``SLRModel``)
------------------------------------------------------------------------
The upstream ``SLRModel`` subclasses ``honeybees.model.Model`` and its
``Agents`` constructor eagerly builds the full DYNAMO-M object graph (coastal /
inland nodes, beaches, government + insurer agents, gravity models, GLOFRIS
flood risk, geojson study areas, spin-up, pickling, low-memory scratch folders).
Instantiating it requires the entire DYNAMO-M **data + geodata ecosystem** plus
``honeybees`` / ``mesa`` — none of which need be present to *use* FloodAdapt-ABM.
That full binding ("4b-full") is a documented follow-up: see the Phase-4b model
documentation ``20260708_phase_4b_model_documentaiton_phase.docx`` (Sections 2 and
8, incl. the effort/risk assessment for 4b-full) and the "Phase 4b" section of
``AGENTS.md``.

What this module provides instead is a faithful, dependency-free **mirror of the
native control flow** that reuses the already-validated FloodAdapt-ABM kernels:

======================  ======================================================
Native DYNAMO-M         FloodAdapt-ABM Mesa-native mirror (this module)
======================  ======================================================
``SLRModel``            :class:`FloodAdaptSLRModel`  (owns time)
``SLRModel.run_model``  :meth:`FloodAdaptSLRModel.run_model` (``while`` tick loop)
``SLRModel.step``       :meth:`FloodAdaptSLRModel.step`      (one native tick)
``Agents``              :class:`Agents`                     (has ``step()``)
``Agents.regions``      :class:`CoastalNodePopulation`       (vectorised households)
``CoastalNode`` arrays  :class:`~floodadapt_abm.agent_state.AgentState`
decision                shared :class:`~floodadapt_abm.decision_rule.DecisionRule`
======================  ======================================================

Because both paths call the *same* :meth:`SimulationEngine.step` kernel with the
*same* RNG stream, the Mesa-native driver reproduces ``engine.run()``
**bit-for-bit** — that equivalence is the Phase-4b gate and the proof that the
time-ownership inversion is non-breaking (see ``test_mesa_native.py`` and the
``examples_engine/06_mesa_native_driving.py`` demo).

The seam that stays stable across 4a -> 4b is the ``DecisionRule.should_adapt``
contract: in 4a it is called inside ``engine.run``; in 4b the very same call is
made inside ``model.step`` — and ultimately, in 4b-full, inside the native
``honeybees`` ``model.step()``.
"""
from __future__ import annotations

import numpy as np

from floodadapt_abm.simulation_engine import SimulationEngine


class CoastalNodePopulation:
    """
    Vectorised household population — the FloodAdapt-ABM mirror of a DYNAMO-M
    ``CoastalNode`` (which holds per-household arrays for one region).

    It owns no state of its own: the household arrays live in the engine's
    :class:`~floodadapt_abm.agent_state.AgentState`, and its :meth:`step`
    advances the population by exactly one year by delegating to the shared
    :meth:`SimulationEngine.step` kernel (event draw -> realised damage ->
    flood-experience update -> lifespan reset -> decision -> bookkeeping).

    Parameters
    ----------
    model : FloodAdaptSLRModel
        The owning model (provides the engine, clock, RNG and history buffers).
    """

    def __init__(self, model: "FloodAdaptSLRModel") -> None:
        self.model = model

    @property
    def state(self):
        """Live per-agent :class:`AgentState` (mutated in place each tick)."""
        return self.model.engine.state

    @property
    def n(self) -> int:
        """Number of households (mirrors ``CoastalNode.n``)."""
        return self.model.engine.n_agents

    def step(self) -> None:
        """Advance the population by one year (mirrors ``CoastalNode.step``)."""
        m = self.model
        m._check_not_stale()
        t = m.timestep
        res = m.engine.step(
            t, float(m.slr_values[t]), m._rng, m.interp_method
        )
        m._record(t, res)


class Agents:
    """
    Container that advances every agent group each tick — the FloodAdapt-ABM
    mirror of DYNAMO-M's ``Agents`` (which steps regions, beaches, government,
    …).  Here the only group is the coastal household population.

    Parameters
    ----------
    model : FloodAdaptSLRModel
        The owning model.
    """

    def __init__(self, model: "FloodAdaptSLRModel") -> None:
        self.model = model
        self.regions = CoastalNodePopulation(model)

    def step(self) -> None:
        """Step each agent group once (mirrors ``Agents.step``)."""
        self.regions.step()


class FloodAdaptSLRModel:
    """
    Minimal, framework-free Mesa-native model that **owns time** — the
    FloodAdapt-ABM mirror of DYNAMO-M's ``SLRModel``.

    One :meth:`step` is one native tick (one year); :meth:`run_model` is the
    ``while`` tick loop.  A single model instance simulates **one** Monte-Carlo
    sequence; :func:`run_mesa_native` orchestrates ``no_seq`` sequences with
    fresh state + an independent RNG stream each, exactly like
    :meth:`SimulationEngine.run`.

    Parameters
    ----------
    engine : SimulationEngine
        The shared kernel (data plumbing, event draw, decision rule, state).
    slr_values : np.ndarray
        Per-year SLR trajectory; its length is the horizon (time is driven by
        this, mirroring the config-driven ``start_time``/``end_time`` of the
        native model).
    seed : int
        RNG seed for this sequence.
    interp_method : str
        Damage-interpolation kind forwarded to the engine.
    track_eu : bool
        Record per-year expected utilities when the rule exposes them.

    Attributes
    ----------
    timestep : int
        Current 0-based year index (the model's clock).
    n_timesteps : int
        Number of years (``len(slr_values)``).
    agents : Agents
        The agent container (``agents.regions`` is the household population).
    """

    def __init__(
        self,
        engine: SimulationEngine,
        slr_values: np.ndarray,
        seed: int,
        interp_method: str = "linear",
        track_eu: bool = False,
    ) -> None:
        self.engine = engine
        self.slr_values = np.asarray(slr_values, dtype=float)
        self.n_timesteps = int(self.slr_values.shape[0])
        self.interp_method = interp_method
        self.track_eu = track_eu

        # Clock + RNG (owned by the model, mirroring SLRModel).
        self.timestep = 0
        self._rng = np.random.default_rng(seed)

        # Fresh per-agent state for this sequence.  The engine's state is
        # shared: constructing a model resets it and claims the new state
        # epoch, so any *earlier* model on the same engine becomes stale (its
        # step() will raise rather than silently mutate this model's state).
        self.engine.reset_state()
        self._state_epoch = engine.state_epoch

        # Agent object graph (mirrors SLRModel.agents = Agents(self)).
        self.agents = Agents(self)

        # Per-year history buffers (filled in place by _record).
        n = self.engine.n_agents
        self.damage_history = np.zeros((n, self.n_timesteps), dtype=engine.damage_dtype)
        self.adapted_history = np.zeros((n, self.n_timesteps), dtype=bool)
        self.eu_adapt_history = (
            np.full((n, self.n_timesteps), np.nan, dtype=np.float32)
            if track_eu else None
        )
        self.eu_do_nothing_history = (
            np.full((n, self.n_timesteps), np.nan, dtype=np.float32)
            if track_eu else None
        )

    # -- staleness guard ----------------------------------------------------
    def _check_not_stale(self) -> None:
        """
        Raise if the engine's state has been reset since this model claimed it.

        The engine's :class:`AgentState` is shared; constructing another
        ``FloodAdaptSLRModel`` (or calling ``engine.run()`` / ``reset_state()``)
        replaces it.  Without this guard a stale model would silently step the
        *new* owner's state — a hard-to-attribute corruption bug.
        """
        if self.engine.state_epoch != self._state_epoch:
            raise RuntimeError(
                "This FloodAdaptSLRModel is stale: engine.reset_state() has "
                "been called since it was constructed (e.g. by a newer model "
                "or engine.run()). Construct a new model, or use a dedicated "
                "SimulationEngine per concurrently-driven model."
            )

    # -- recording ----------------------------------------------------------
    def _record(self, t: int, res: dict) -> None:
        self.damage_history[:, t] = res["damages"]
        self.adapted_history[:, t] = res["is_adapted"]
        if self.track_eu and res["eu_adapt"] is not None:
            self.eu_adapt_history[:, t] = res["eu_adapt"]
            self.eu_do_nothing_history[:, t] = res["eu_do_nothing"]

    # -- ticking ------------------------------------------------------------
    def step(self) -> None:
        """
        Advance the model by one native tick (mirrors ``SLRModel.step``):
        step the agents for the current year, then advance the clock.
        """
        self.agents.step()
        self.timestep += 1

    def run_model(self) -> None:
        """
        Drive ticks until the horizon is reached — the FloodAdapt-ABM mirror of
        ``SLRModel.run_model`` (``while True: self.step(); if done: break``).
        """
        while self.timestep < self.n_timesteps:
            self.step()


def run_mesa_native(
    engine: SimulationEngine,
    slr_values,
    no_seq: int = 1,
    seed: int | None = None,
    interp_method: str = "linear",
    track_eu: bool = False,
) -> dict:
    """
    Run the simulation with **Mesa-native time driving** (Phase 4b).

    Drop-in analogue of :meth:`SimulationEngine.run`, but the year loop is owned
    by :class:`FloodAdaptSLRModel` ticks rather than by ``engine.run``.  Because
    both paths call the identical per-year kernel with the identical RNG stream,
    the returned arrays are **bit-for-bit identical** to ``engine.run(...)`` for
    the same arguments — that equivalence is the Phase-4b gate.

    Parameters
    ----------
    engine : SimulationEngine
        The shared engine (its decision rule selects the behaviour).
    slr_values : array-like
        Per-year SLR trajectory (length ``n_years``); drives the horizon.
    no_seq : int
        Number of Monte-Carlo sequences.
    seed : int or None
        Base RNG seed; sequence ``s`` uses ``seed + s`` (matching
        :meth:`SimulationEngine.run`).  ``None`` uses ``config.random_seed``.
    interp_method : str
        Damage-interpolation kind.
    track_eu : bool
        Record per-year expected utilities when the rule exposes them.

    Returns
    -------
    dict
        Same schema as :meth:`SimulationEngine.run`:
        ``damage_history`` / ``adapted_history`` ``(no_seq, n_agents, n_years)``,
        ``adoption_fraction`` ``(no_seq, n_years)``, and optionally
        ``eu_adapt_history`` / ``eu_do_nothing_history``.
    """
    slr_values = np.asarray(slr_values, dtype=float)
    n_years = int(slr_values.shape[0])
    base_seed = seed if seed is not None else engine.config.random_seed
    n = engine.n_agents

    damage_history = np.zeros((no_seq, n, n_years), dtype=engine.damage_dtype)
    adapted_history = np.zeros((no_seq, n, n_years), dtype=bool)
    eu_adapt_history = (
        np.full((no_seq, n, n_years), np.nan, dtype=np.float32) if track_eu else None
    )
    eu_do_nothing_history = (
        np.full((no_seq, n, n_years), np.nan, dtype=np.float32) if track_eu else None
    )

    for s in range(no_seq):
        model = FloodAdaptSLRModel(
            engine=engine,
            slr_values=slr_values,
            seed=base_seed + s,
            interp_method=interp_method,
            track_eu=track_eu,
        )
        model.run_model()  # <-- time owned by model.step() ticks

        damage_history[s] = model.damage_history
        adapted_history[s] = model.adapted_history
        if track_eu:
            eu_adapt_history[s] = model.eu_adapt_history
            eu_do_nothing_history[s] = model.eu_do_nothing_history

    results = {
        "damage_history": damage_history,
        "adapted_history": adapted_history,
        "adoption_fraction": adapted_history.mean(axis=1),
    }
    if track_eu:
        results["eu_adapt_history"] = eu_adapt_history
        results["eu_do_nothing_history"] = eu_do_nothing_history
    return results
