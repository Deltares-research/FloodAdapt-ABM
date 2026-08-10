"""
agent_state.py
==============
Standardised per-agent state container for the unified ``SimulationEngine``
(Phase 3 step-wise refactoring).

Before this refactor, ``ABMSimulator`` and ``DynamoDecisionBridge`` each kept
their own loose set of per-agent arrays (``is_floodproofed`` vs ``is_adapted``,
separate ``flood_timer`` / ``risk_perception`` handling, no shared
``time_adapted``).  ``AgentState`` collapses these into one vectorised,
NumPy-first container passed to every :class:`~floodadapt_abm.decision_rule.DecisionRule`.

The container is deliberately a plain mutable dataclass of parallel arrays (all
shape ``(n_agents,)``) rather than a per-agent object, to keep the hot decision
path fully vectorised.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AgentState:
    """
    Vectorised per-agent state (all arrays have shape ``(n_agents,)``).

    Attributes
    ----------
    wealth : np.ndarray[float32]
        Household wealth per agent.
    income : np.ndarray[float32]
        Annual income per agent.
    risk_perception : np.ndarray[float32]
        Current subjective risk-perception multiplier per agent.
    flood_timer : np.ndarray[int32]
        Years since each agent last experienced a flood.  Large values decay
        ``risk_perception`` toward ``risk_perc_min``.
    is_adapted : np.ndarray[bool]
        Current adaptation (dry-floodproofing) status per agent.
    time_adapted : np.ndarray[int32]
        Age of each agent's current adaptation in years.  ``0`` for
        never-adapted agents; incremented each year an agent remains adapted;
        reset when the measure expires (``>= lifespan_dryproof``) and the agent
        un-adapts.  This is the field that enables the lifespan-dryproof reset
        absent from the original bridge.
    last_flood_severity : np.ndarray[float32]
        Damage severity (``realised / max_pot_dmg``, in ``[0, 1]``) of each
        agent's most recent significant flood.  ``0`` for agents never
        flooded.  Used by ``perception_mode="severity"`` to scale the
        post-flood risk-perception peak; ignored in binary mode.
    is_insured : np.ndarray[bool]
        Current flood-insurance status per agent (re-decided every year).
        Always ``False`` while ``include_insurance`` is off.
    """

    wealth: np.ndarray
    income: np.ndarray
    risk_perception: np.ndarray
    flood_timer: np.ndarray
    is_adapted: np.ndarray
    time_adapted: np.ndarray
    last_flood_severity: np.ndarray | None = None
    is_insured: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Default the newer state arrays for backward-compatible construction."""
        n = int(self.wealth.shape[0])
        if self.last_flood_severity is None:
            self.last_flood_severity = np.zeros(n, dtype=np.float32)
        if self.is_insured is None:
            self.is_insured = np.zeros(n, dtype=bool)

    @property
    def n_agents(self) -> int:
        """Number of agents (length of the state arrays)."""
        return int(self.wealth.shape[0])

    @classmethod
    def initial(
        cls,
        n_agents: int,
        income: np.ndarray,
        wealth: np.ndarray,
        risk_perc_min: float,
        initial_flood_timer: int = 99,
    ) -> "AgentState":
        """
        Build a fresh state for ``n_agents`` at the start of a run.

        All agents start un-adapted, with ``flood_timer`` set to a large value
        (``initial_flood_timer``) so their initial ``risk_perception`` sits at
        the ``risk_perc_min`` floor, matching ``DynamoDecisionBridge``'s
        original initialisation.

        Parameters
        ----------
        n_agents : int
            Number of agents.
        income, wealth : np.ndarray
            Per-agent economic arrays, shape ``(n_agents,)``.
        risk_perc_min : float
            Minimum risk-perception multiplier used as the initial value.
        initial_flood_timer : int
            Initial years-since-flood for every agent.  Default ``99``.
        """
        return cls(
            wealth=np.asarray(wealth, dtype=np.float32).copy(),
            income=np.asarray(income, dtype=np.float32).copy(),
            risk_perception=np.full(n_agents, risk_perc_min, dtype=np.float32),
            flood_timer=np.full(n_agents, initial_flood_timer, dtype=np.int32),
            is_adapted=np.zeros(n_agents, dtype=bool),
            time_adapted=np.zeros(n_agents, dtype=np.int32),
            last_flood_severity=np.zeros(n_agents, dtype=np.float32),
            is_insured=np.zeros(n_agents, dtype=bool),
        )

    def copy(self) -> "AgentState":
        """Return a deep copy (all arrays copied)."""
        return AgentState(
            wealth=self.wealth.copy(),
            income=self.income.copy(),
            risk_perception=self.risk_perception.copy(),
            flood_timer=self.flood_timer.copy(),
            is_adapted=self.is_adapted.copy(),
            time_adapted=self.time_adapted.copy(),
            last_flood_severity=self.last_flood_severity.copy(),
            is_insured=self.is_insured.copy(),
        )
