"""
coastal_node_adapter.py
=======================
PRE.4 (Phase 4b-full pre-flight): prototype of the **lookup-table ->
CoastalNode adapter** — the single genuinely new modelling artefact of the
4b-full migration (roadmap step 3 in
``docs/20260709_proposed_development_architecture_steps.md`` Sec.7.2).

In native DYNAMO-M each region is a ``CoastalNode`` holding per-household
arrays (``node.n``, ``node.property_value``, ``node.damages_coastal_cells``,
``node.adapt``, ``node.time_adapt``); each tick, ``calcEU_*`` consumes those
arrays.  In 4b-full the same arrays must be **populated from the FloodAdapt
lookup table** instead of from GLOFRIS/DataDrive — and the adaptation
decisions must flow back.  This module prototypes exactly that seam against
the validated FloodAdapt-ABM kernels, with **no honeybees/mesa dependency**,
so the mapping can be verified bit-for-bit before the real environment lift
(4b-full steps 1-2) is attempted.

Forward direction  (FloodAdapt -> DYNAMO-M):
    ``populate(slr_value)`` interpolates the engine's per-event damage
    catalogues at SLR_t and packs them into a :class:`CoastalNodeArrays`
    using DYNAMO-M's conventions (events-first ``(n_events, n)`` matrices,
    ``property_value`` = ``max_pot_dmg``, ``p_floods`` = event frequencies,
    ``adapt`` as an int array, geom_id ending in ``_flood_plain``).

Reverse direction  (DYNAMO-M -> FloodAdapt):
    ``write_back(node)`` routes ``node.adapt`` / ``node.time_adapt`` back
    into the engine's live :class:`AgentState` — the "decisions flow back"
    half of the reverse-coupling heartbeat.

The bit-parity contract is executable via :func:`round_trip_check` and the
tests in ``tests/test_coastal_node_adapter.py``: every array must survive the
FloodAdapt -> node -> FloodAdapt round trip **bit-identically**, so that the
eventual 4b-full ≡ 4b-scaffold gate stays attributable to the environment
swap alone (one variable at a time, as in the 4a -> 4b progression).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from floodadapt_abm.simulation_engine import SimulationEngine


@dataclass
class CoastalNodeArrays:
    """
    Dependency-free mirror of the native DYNAMO-M ``CoastalNode`` array set
    (the fields consumed by ``calcEU_do_nothing`` / ``calcEU_adapt`` and the
    adaptation bookkeeping in ``coastal_nodes.py``).

    Attributes
    ----------
    geom_id : str
        Region identifier; native code filters on the ``'_flood_plain'``
        suffix, so the adapter enforces it.
    n : int
        Number of households in the node (mirrors ``CoastalNode.n``).
    object_ids : np.ndarray
        FloodAdapt ``object_id`` per household — the join key back to the
        lookup table (and to ``AgentState`` row order).
    property_value : np.ndarray[float64]
        ``max_pot_dmg`` per household (mirrors ``node.property_value``).
    wealth, income : np.ndarray[float32]
        Household economics (mirrors ``node.wealth`` / ``node.income``).
    damages_coastal_cells : np.ndarray[float32], shape (n_events, n)
        Per-event no-measures damages at the populated SLR, **events-first**
        (the native convention; FloodAdapt-ABM uses agents-first internally).
    damages_coastal_cells_adapt : np.ndarray[float32], shape (n_events, n)
        Per-event damages under the floodproof strategy.
    p_floods : np.ndarray[float64], shape (n_events,)
        Exceedance probabilities (= event frequencies = 1/RP).
    adapt : np.ndarray[int8]
        Adaptation status per household (native stores 0/1 ints).
    time_adapt : np.ndarray[int32]
        Adaptation age per household (drives the lifespan-dryproof reset).
    slr_value : float
        The SLR at which the damage arrays were interpolated.
    """

    geom_id: str
    n: int
    object_ids: np.ndarray
    property_value: np.ndarray
    wealth: np.ndarray
    income: np.ndarray
    damages_coastal_cells: np.ndarray
    damages_coastal_cells_adapt: np.ndarray
    p_floods: np.ndarray
    adapt: np.ndarray
    time_adapt: np.ndarray
    slr_value: float


class LookupTableAdapter:
    """
    Maps between a :class:`SimulationEngine` (FloodAdapt lookup-table world)
    and a :class:`CoastalNodeArrays` (native DYNAMO-M node world).

    Parameters
    ----------
    engine : SimulationEngine
        The engine whose data layer and live ``AgentState`` are adapted.
    geom_id : str
        Node identifier.  A ``'_flood_plain'`` suffix is appended when absent
        (native node filtering convention).
    """

    def __init__(self, engine: SimulationEngine, geom_id: str = "floodadapt_flood_plain") -> None:
        self.engine = engine
        if not geom_id.endswith("_flood_plain"):
            geom_id = f"{geom_id}_flood_plain"
        self.geom_id = geom_id

    # -- forward: lookup table -> native node arrays --------------------------
    def populate(self, slr_value: float, interp_method: str = "linear") -> CoastalNodeArrays:
        """
        Build the native node arrays from the lookup table at ``slr_value``.

        Uses the engine's own (validated) interpolation kernel, then re-packs
        into DYNAMO-M conventions.  The engine's live state supplies the
        household economics and current adaptation status, so a node populated
        mid-run reflects the run's actual cohort.
        """
        eng = self.engine
        d_no, d_fp = eng.prepare_damages(slr_value, interp_method)
        state = eng.state
        return CoastalNodeArrays(
            geom_id=self.geom_id,
            n=eng.n_agents,
            object_ids=np.asarray(eng.object_ids).copy(),
            property_value=np.asarray(eng.max_pot_dmg, dtype=np.float64).copy(),
            wealth=state.wealth.copy(),
            income=state.income.copy(),
            # native convention is events-first; .T is a transpose copy here
            damages_coastal_cells=np.ascontiguousarray(d_no.T),
            damages_coastal_cells_adapt=np.ascontiguousarray(d_fp.T),
            p_floods=np.asarray(eng._event_freqs, dtype=np.float64).copy(),
            adapt=state.is_adapted.astype(np.int8),
            time_adapt=state.time_adapted.copy(),
            slr_value=float(slr_value),
        )

    # -- reverse: native node decisions -> engine state ----------------------
    def write_back(self, node: CoastalNodeArrays) -> None:
        """
        Route the node's adaptation state back into the engine's live
        ``AgentState`` (the decisions-flow-back half of the coupling loop).

        Raises
        ------
        ValueError
            If the node does not align with the engine's agent population
            (size or ``object_id`` order mismatch) — the same silent
            misalignment risk flagged for ``read_impacts_dataset``.
        """
        eng = self.engine
        if node.n != eng.n_agents:
            raise ValueError(
                f"node has {node.n} households but engine has {eng.n_agents} agents"
            )
        if not np.array_equal(node.object_ids, np.asarray(eng.object_ids)):
            raise ValueError("node.object_ids order does not match engine.object_ids")
        eng.state.is_adapted[:] = node.adapt.astype(bool)
        eng.state.time_adapted[:] = node.time_adapt


def round_trip_check(engine: SimulationEngine, slr_value: float,
                     interp_method: str = "linear") -> dict:
    """
    Executable bit-parity contract for the adapter (the PRE.4 gate).

    Populates a node from the engine, then verifies every mapped array is
    **bit-identical** to its source, and that ``write_back`` restores a
    mutated adaptation state exactly.

    Returns
    -------
    dict
        Per-check booleans plus ``"all_pass"``.
    """
    adapter = LookupTableAdapter(engine)
    node = adapter.populate(slr_value, interp_method)
    d_no, d_fp = engine.prepare_damages(slr_value, interp_method)

    checks = {
        "geom_id_flood_plain_suffix": node.geom_id.endswith("_flood_plain"),
        "n_matches": node.n == engine.n_agents,
        "object_ids_identical": bool(np.array_equal(node.object_ids, engine.object_ids)),
        "property_value_identical": bool(np.array_equal(node.property_value, engine.max_pot_dmg)),
        "damages_no_adapt_identical": bool(np.array_equal(node.damages_coastal_cells.T, d_no)),
        "damages_adapt_identical": bool(np.array_equal(node.damages_coastal_cells_adapt.T, d_fp)),
        "p_floods_identical": bool(np.array_equal(node.p_floods, engine._event_freqs)),
        "adapt_matches_state": bool(
            np.array_equal(node.adapt.astype(bool), engine.state.is_adapted)
        ),
    }

    # reverse direction: mutate on the node side, write back, compare exactly
    rng = np.random.default_rng(0)
    node.adapt = (rng.random(node.n) < 0.5).astype(np.int8)
    node.time_adapt = rng.integers(0, 40, node.n).astype(np.int32)
    adapter.write_back(node)
    checks["write_back_adapt_identical"] = bool(
        np.array_equal(engine.state.is_adapted, node.adapt.astype(bool))
    )
    checks["write_back_time_adapt_identical"] = bool(
        np.array_equal(engine.state.time_adapted, node.time_adapt)
    )

    checks["all_pass"] = all(checks.values())
    return checks
