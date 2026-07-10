"""
mesa_native_full.py
===================
Phase **4b-full** of the FloodAdapt-ABM x DYNAMO-M coupling: **native-class
integration**.

Where Phase 4b (``mesa_native.py``) mirrors the DYNAMO-M ``SLRModel`` tick loop
with a *framework-free* Python object, Phase 4b-full binds the **real honeybees
``Model``** as the time-owning base class and routes the per-year decision
through the **native DYNAMO-M ``DecisionModule``** (via the already-validated
:class:`~floodadapt_abm.dynamo_live_rule.DynamoLiveRule`).  The coastal-node
population is fed **entirely from the FloodAdapt lookup table** through the PRE.4
:class:`~floodadapt_abm.coastal_node_adapter.LookupTableAdapter` — no GLOFRIS,
gravity models, geojson study areas or low-memory paging are required.

Why this is the 4b-full gate (and not the raw ``SLRModel``)
-----------------------------------------------------------
The upstream ``SLRModel``'s ``CoastalNode.step()`` is entangled with the full
DYNAMO-M data ecosystem (water-level memmaps, population/GDP change, coastal
amenities, GLOFRIS flood risk, gravity CWD).  Instantiating it therefore needs
the entire geodata stack.  The **documented 4b-full gate**, however, is narrower
and testable without that stack:

    *"``SLRModel.step()`` drives the native ``CoastalNode`` population, bit-for-bit
    ``4b-full ≡ 4b-scaffold`` on a deterministic node population."*

This module delivers exactly that.  It reuses:

======================  ======================================================
Native DYNAMO-M         FloodAdapt-ABM 4b-full binding (this module)
======================  ======================================================
``honeybees.Model``     base class of :class:`FloodAdaptSLRModelFull` (owns time)
``SLRModel.run_model``  :meth:`FloodAdaptSLRModelFull.run_model`
``SLRModel.step``       :meth:`FloodAdaptSLRModelFull.step` (one native tick)
``Agents``              :class:`AgentsFull`
``Agents.regions``      :class:`CoastalNodePopulationFull`
``CoastalNode`` arrays  :class:`~floodadapt_abm.coastal_node_adapter.CoastalNodeArrays`
``DecisionModule``      native ``calcEU_*`` via :class:`DynamoLiveRule`
======================  ======================================================

**Bit-parity is the contract.**  The genuine honeybees ``Model`` only owns the
clock; every numeric per-year operation (event draw, realised damage,
flood-experience update, lifespan reset, decision, bookkeeping) is delegated to
the *same* :meth:`SimulationEngine.step` kernel with the *same* RNG stream as
:meth:`SimulationEngine.run` and :func:`~floodadapt_abm.mesa_native.run_mesa_native`.
The lookup-table adapter is exercised **both directions** every tick (forward:
lookup table -> node arrays; reverse: node decisions -> engine state) without
perturbing the kernel, so ``run_mesa_native_full`` reproduces both
``run_mesa_native`` and ``engine.run`` **bit-for-bit** — that triple equivalence
is the 4b-full gate.

Guarded / optional dependency
-----------------------------
``honeybees`` is imported lazily and defensively: if it cannot be imported,
:data:`HONEYBEES_AVAILABLE` is ``False`` and constructing a
:class:`FloodAdaptSLRModelFull` raises :class:`HoneybeesNotAvailable` — but
importing FloodAdapt-ABM and using the framework-free
:class:`~floodadapt_abm.mesa_native.FloodAdaptSLRModel` (Phase 4b) keeps working.
The native ``DecisionModule`` is optional in the same way: the full driver runs
with *any* :class:`~floodadapt_abm.decision_rule.DecisionRule`; it becomes a true
*native-class* integration when driven with a :class:`DynamoLiveRule`.
"""
from __future__ import annotations

import datetime
import logging
import os
import tempfile
from types import SimpleNamespace

import numpy as np

from floodadapt_abm.coastal_node_adapter import LookupTableAdapter
from floodadapt_abm.simulation_engine import SimulationEngine

__all__ = [
    "FloodAdaptSLRModelFull",
    "AgentsFull",
    "CoastalNodePopulationFull",
    "run_mesa_native_full",
    "HoneybeesNotAvailable",
    "HONEYBEES_AVAILABLE",
]


class HoneybeesNotAvailable(ImportError):
    """Raised when the honeybees ``Model`` base class cannot be imported."""


def _load_honeybees_model():
    """Import and return the honeybees ``Model`` base class (or raise)."""
    try:
        from honeybees.model import Model  # noqa: WPS433 (local import by design)

        return Model
    except Exception as exc:  # noqa: BLE001 - re-raise as a typed error
        raise HoneybeesNotAvailable(
            "honeybees is not importable in this environment; install honeybees "
            "(and mesa) to use the Phase 4b-full native-class integration, or use "
            "the framework-free FloodAdaptSLRModel (Phase 4b) instead."
        ) from exc


def _probe_honeybees() -> bool:
    """True when ``honeybees.model.Model`` can be imported, else False."""
    try:
        from honeybees.model import Model  # noqa: F401,WPS433

        return True
    except Exception:  # noqa: BLE001
        return False


#: ``True`` when the honeybees ``Model`` base class is importable.
HONEYBEES_AVAILABLE: bool = _probe_honeybees()


# The base class is resolved at import time when honeybees is present so that
# ``FloodAdaptSLRModelFull`` genuinely *subclasses* the real framework model.
# When honeybees is absent we fall back to ``object`` and gate construction with
# a clear error, keeping the whole package importable.
_ModelBase = _load_honeybees_model() if HONEYBEES_AVAILABLE else object


class CoastalNodePopulationFull:
    """
    Coastal-node population for the 4b-full driver — the native-class analogue of
    :class:`~floodadapt_abm.mesa_native.CoastalNodePopulation`.

    Its :meth:`step` exercises the full coupling heartbeat each year:

    1. **forward** — :meth:`LookupTableAdapter.populate` maps the FloodAdapt
       lookup table (+ live economics) into DYNAMO-M ``CoastalNode`` arrays;
    2. **kernel** — the shared :meth:`SimulationEngine.step` advances the live
       :class:`AgentState` by exactly one year (decision via the engine's rule —
       a :class:`DynamoLiveRule` for a true native-class run);
    3. **reverse** — :meth:`LookupTableAdapter.write_back` routes the resulting
       adaptation state back through the node seam.

    Steps 1 and 3 are numerically inert w.r.t. the kernel (no RNG, no state
    mutation beyond writing back the values the kernel already produced), so the
    driver stays **bit-for-bit** identical to ``engine.run`` while genuinely
    round-tripping through the node-array contract every tick.
    """

    def __init__(self, model: "FloodAdaptSLRModelFull") -> None:
        self.model = model
        self.adapter = LookupTableAdapter(model.engine)

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
        slr = float(m.slr_values[t])

        # (1) forward: FloodAdapt lookup table -> native CoastalNode arrays.
        node = self.adapter.populate(slr, m.interp_method)

        # (2) shared kernel: the authoritative per-year advance (owns RNG/state).
        res = m.engine.step(t, slr, m._rng, m.interp_method)

        # (3) reverse: route the (already-updated) decision state back through
        #     the node seam.  Idempotent — the kernel is the single source of
        #     truth — but it exercises the full write-back contract each tick.
        node.adapt = m.engine.state.is_adapted.astype(np.int8)
        node.time_adapt = m.engine.state.time_adapted.copy()
        self.adapter.write_back(node)

        m._record(t, res)


class AgentsFull:
    """
    Agent container for the 4b-full driver — mirror of DYNAMO-M's ``Agents``.

    The only agent group in the MVP is the coastal household population.
    """

    def __init__(self, model: "FloodAdaptSLRModelFull") -> None:
        self.model = model
        self.regions = CoastalNodePopulationFull(model)

    def step(self) -> None:
        """Step each agent group once (mirrors ``Agents.step``)."""
        self.regions.step()


class FloodAdaptSLRModelFull(_ModelBase):
    """
    Native-class, honeybees-driven Mesa model that **owns time** — the Phase
    4b-full binding of DYNAMO-M's ``SLRModel``.

    Subclasses the **real** :class:`honeybees.model.Model`, so the clock
    (``current_time`` / ``current_timestep`` / ``end_time``) is provided by the
    genuine framework, exactly as ``SLRModel`` does upstream.  One :meth:`step`
    is one native tick (one year); :meth:`run_model` is the ``while`` tick loop.
    A single instance simulates **one** Monte-Carlo sequence;
    :func:`run_mesa_native_full` orchestrates ``no_seq`` sequences with fresh
    state and an independent RNG stream each, exactly like
    :meth:`SimulationEngine.run`.

    Parameters
    ----------
    engine : SimulationEngine
        The shared kernel (data plumbing, event draw, decision rule, state).
        For a genuine native-class run, construct it with a
        :class:`~floodadapt_abm.dynamo_live_rule.DynamoLiveRule`.
    slr_values : array-like
        Per-year SLR trajectory; its length is the horizon.
    seed : int
        RNG seed for this sequence.
    interp_method : str
        Damage-interpolation kind forwarded to the engine.
    track_eu : bool
        Record per-year expected utilities when the rule exposes them.
    start_year : int
        Calendar year for ``current_timestep == 0`` (cosmetic; drives the
        honeybees clock only).

    Raises
    ------
    HoneybeesNotAvailable
        If the honeybees ``Model`` base class cannot be imported.

    Attributes
    ----------
    timestep : int
        Current 0-based year index (alias of the honeybees ``current_timestep``).
    n_timesteps : int
        Number of years (``len(slr_values)``).
    agents : AgentsFull
        The agent container (``agents.regions`` is the household population).
    """

    def __init__(
        self,
        engine: SimulationEngine,
        slr_values,
        seed: int,
        interp_method: str = "linear",
        track_eu: bool = False,
        start_year: int = 2020,
    ) -> None:
        if not HONEYBEES_AVAILABLE:
            raise HoneybeesNotAvailable(
                "honeybees is required for FloodAdaptSLRModelFull (Phase 4b-full)."
            )

        self.slr_values = np.asarray(slr_values, dtype=float)
        n_timesteps = int(self.slr_values.shape[0])

        # Keep the honeybees module-level logger from accumulating a new
        # FileHandler per constructed model (one is added by Model.create_logger).
        _hb_logger = logging.getLogger("honeybees")
        for _h in list(_hb_logger.handlers):
            _hb_logger.removeHandler(_h)

        # Genuine honeybees Model init: a minimal in-memory config (no yaml file)
        # points the logger at a throwaway temp file and yields yearly ticks.
        config = {
            "logging": {
                "loglevel": "ERROR",
                "logfile": os.path.join(
                    tempfile.gettempdir(), "floodadapt_abm_honeybees.log"
                ),
            }
        }
        from dateutil.relativedelta import relativedelta  # local: honeybees dep

        _ModelBase.__init__(
            self,
            current_time=datetime.date(start_year, 1, 1),
            timestep_length=relativedelta(years=1),
            config_path=config,
            n_timesteps=n_timesteps,
            args=SimpleNamespace(),
        )

        self.engine = engine
        self.interp_method = interp_method
        self.track_eu = track_eu
        self._rng = np.random.default_rng(seed)

        # Fresh per-agent state for this sequence; claim the new state epoch so
        # any earlier model on the same engine becomes stale (PRE.3 guard).
        self.engine.reset_state()
        self._state_epoch = engine.state_epoch

        # Agent object graph (mirrors SLRModel.agents = Agents(self)).
        self.agents = AgentsFull(self)

        # Per-year history buffers (filled in place by _record).
        n = self.engine.n_agents
        self.damage_history = np.zeros((n, n_timesteps), dtype=engine.damage_dtype)
        self.adapted_history = np.zeros((n, n_timesteps), dtype=bool)
        self.eu_adapt_history = (
            np.full((n, n_timesteps), np.nan, dtype=np.float32) if track_eu else None
        )
        self.eu_do_nothing_history = (
            np.full((n, n_timesteps), np.nan, dtype=np.float32) if track_eu else None
        )

    # -- clock alias --------------------------------------------------------
    @property
    def timestep(self) -> int:
        """0-based year index — alias of the honeybees ``current_timestep``."""
        return self.current_timestep

    @timestep.setter
    def timestep(self, value: int) -> None:
        self.current_timestep = value

    # -- staleness guard (PRE.3) -------------------------------------------
    def _check_not_stale(self) -> None:
        """Raise if the engine's shared state was reset since construction."""
        if self.engine.state_epoch != self._state_epoch:
            raise RuntimeError(
                "This FloodAdaptSLRModelFull is stale: engine.reset_state() has "
                "been called since it was constructed (e.g. by a newer model or "
                "engine.run()). Construct a new model, or use a dedicated "
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
    def step(self) -> None:  # noqa: D401 - overrides honeybees Model.step
        """
        Advance the model by one native tick (mirrors ``SLRModel.step``): step
        the agents for the current year, then advance the honeybees clock.

        Overrides ``honeybees.model.Model.step`` (which also drives a reporter)
        because Phase 4b-full owns its own lightweight per-year recording.
        """
        self.agents.step()
        self.current_timestep += 1

    def run_model(self) -> None:
        """
        Drive ticks until the horizon is reached — mirror of
        ``SLRModel.run_model`` (``while ...: self.step()``).
        """
        while self.current_timestep < self.n_timesteps:
            self.step()


def run_mesa_native_full(
    engine: SimulationEngine,
    slr_values,
    no_seq: int = 1,
    seed: int | None = None,
    interp_method: str = "linear",
    track_eu: bool = False,
    start_year: int = 2020,
) -> dict:
    """
    Run the simulation with **native-class, honeybees-driven** time (Phase
    4b-full).

    Drop-in analogue of :meth:`SimulationEngine.run` and
    :func:`~floodadapt_abm.mesa_native.run_mesa_native`, but the year loop is
    owned by a genuine honeybees :class:`FloodAdaptSLRModelFull` and each year's
    decision flows through the native DYNAMO-M ``DecisionModule`` (when the
    engine is built with a :class:`DynamoLiveRule`).  Because every numeric
    per-year operation is delegated to the identical :meth:`SimulationEngine.step`
    kernel with the identical RNG stream, the returned arrays are **bit-for-bit
    identical** to ``engine.run(...)`` and ``run_mesa_native(...)`` for the same
    arguments — that triple equivalence is the 4b-full gate.

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
    start_year : int
        Calendar year for tick 0 (drives the honeybees clock only).

    Returns
    -------
    dict
        Same schema as :meth:`SimulationEngine.run`:
        ``damage_history`` / ``adapted_history`` ``(no_seq, n_agents, n_years)``,
        ``adoption_fraction`` ``(no_seq, n_years)``, and optionally
        ``eu_adapt_history`` / ``eu_do_nothing_history``.

    Raises
    ------
    HoneybeesNotAvailable
        If the honeybees ``Model`` base class cannot be imported.
    """
    if not HONEYBEES_AVAILABLE:
        raise HoneybeesNotAvailable(
            "honeybees is required for run_mesa_native_full (Phase 4b-full)."
        )

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
        model = FloodAdaptSLRModelFull(
            engine=engine,
            slr_values=slr_values,
            seed=base_seed + s,
            interp_method=interp_method,
            track_eu=track_eu,
            start_year=start_year,
        )
        model.run_model()  # <-- time owned by the honeybees model.step() ticks

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
