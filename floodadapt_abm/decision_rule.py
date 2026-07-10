"""
decision_rule.py
================
Pluggable decision rules for the unified ``SimulationEngine`` (Phase 3
step-wise refactoring).

``SimulationEngine`` owns *time and data* (NetCDF loading, interpolation,
stochastic event drawing, state tracking, the year loop); a ``DecisionRule``
owns *behaviour* (whether each household adapts this year).  Swapping the rule
is the only change needed to switch between the legacy threshold heuristic and
the DYNAMO-M SEU science, without touching the engine.

Rules provided
--------------
ThresholdRule
    Reproduces the legacy ``ABMSimulator`` heuristic: adapt when the realised
    damage this year exceeds ``damage_threshold * max_pot_dmg``.
SEURule
    Wraps the validated DYNAMO-M Subjective-Expected-Utility kernels
    (``_calc_eu_do_nothing`` / ``_calc_eu_adapt`` from
    ``dynamo_decision_bridge``): adapt when ``EU_adapt > EU_do_nothing`` and the
    agent is not already adapted (affordability is encoded inside the adapt
    kernel as ``EU_adapt = -inf``).

Design constraints (enforced)
-----------------------------
* **No FloodAdapt or DYNAMO-M imports inside rule kernels** — rules operate on
  plain NumPy arrays.
* **Vectorised** — no per-household Python loops in the hot path.
* **Backward compatible** — ``ThresholdRule`` ignores the SEU-only arguments,
  so the shared ``should_adapt`` signature serves both rules.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from floodadapt_abm.agent_state import AgentState
from floodadapt_abm.coupling_config import DecisionConfig
from floodadapt_abm._core.dynamo_decision_bridge import (
    _calc_eu_adapt,
    _calc_eu_do_nothing,
)


class DecisionRule(ABC):
    """
    Abstract base class for household adaptation decision rules.

    A rule is constructed once from a :class:`DecisionConfig` (scalar
    behavioural parameters) and then queried each year via
    :meth:`should_adapt`.

    The ``should_adapt`` signature is intentionally wide enough to serve both
    an ex-post heuristic (``ThresholdRule``, which uses ``damages_this_year``)
    and the ex-ante SEU science (``SEURule``, which integrates the full
    ``damages_no_adapt`` / ``damages_adapt`` catalogues).  A rule ignores the
    arguments it does not need.
    """

    def __init__(self, config: DecisionConfig):
        self.config = config

    def clone(self, rng_seed: int | None = None) -> "DecisionRule":
        """
        Return an independent copy of this rule for parallel execution.

        The copy shares the (read-only) ``config`` but gets its own random
        generator and its own per-call diagnostic slots, so it can run in a
        separate worker without racing on shared state.  ``rng_seed`` seeds the
        fresh generator for reproducibility; rules that draw no random numbers
        (e.g. ``error_interval == 0``) are unaffected by it.
        """
        import copy as _copy

        new = _copy.copy(self)
        if hasattr(new, "_rng"):
            new._rng = np.random.default_rng(rng_seed)
        if hasattr(new, "last_eu_adapt"):
            new.last_eu_adapt = None
            new.last_eu_do_nothing = None
        return new

    @abstractmethod
    def should_adapt(
        self,
        agent_state: AgentState,
        damages_this_year: np.ndarray,   # (n_agents,)  realised damage this year
        damages_no_adapt: np.ndarray,    # (n_agents, n_events) catalogue @ SLR_t, no measures
        damages_adapt: np.ndarray,       # (n_agents, n_events) catalogue @ SLR_t, floodproofed
        event_freqs: np.ndarray,         # (n_events,)  exceedance probs (= 1/RP)
        max_pot_dmg: np.ndarray,         # (n_agents,)
        adaptation_costs: np.ndarray,    # (n_agents,)  annualised loan repayment
    ) -> np.ndarray:                     # (n_agents,) bool
        """
        Decide which currently non-adapted agents newly adapt this year.

        Returns
        -------
        newly_adapted : np.ndarray[bool], shape (n_agents,)
            ``True`` for agents that switch to adapted *this* year.  Must be
            ``False`` for agents already adapted (``agent_state.is_adapted``).
        """
        raise NotImplementedError


class ThresholdRule(DecisionRule):
    """
    Legacy reactive heuristic (the rule the coupling replaces).

    An agent adapts once the realised damage it suffered *this year* exceeds a
    fixed fraction of its maximum potential damage::

        adapt  if  damages_this_year / max_pot_dmg > damage_threshold

    This reproduces ``ABMSimulator._simulate_damage_history`` bit-for-bit
    (same masking on ``not_adapted & max_pot_dmg > 0``).  All SEU-specific
    arguments are ignored.

    Parameters
    ----------
    config : DecisionConfig
        Used only for interface uniformity; the threshold itself is passed
        separately (defaults to ``0.30``, the legacy value).
    damage_threshold : float
        Fraction of ``max_pot_dmg`` above which an agent adapts.  Default
        ``0.30``.
    """

    def __init__(self, config: DecisionConfig, damage_threshold: float = 0.30):
        super().__init__(config)
        self.damage_threshold = damage_threshold

    def should_adapt(
        self,
        agent_state: AgentState,
        damages_this_year: np.ndarray,
        damages_no_adapt: np.ndarray,
        damages_adapt: np.ndarray,
        event_freqs: np.ndarray,
        max_pot_dmg: np.ndarray,
        adaptation_costs: np.ndarray,
    ) -> np.ndarray:
        not_adapted = ~agent_state.is_adapted
        with_pot_dmg = max_pot_dmg > 0
        valid = not_adapted & with_pot_dmg

        newly_adapted = np.zeros(agent_state.n_agents, dtype=bool)
        newly_adapted[valid] = (
            damages_this_year[valid] / max_pot_dmg[valid]
        ) > self.damage_threshold
        return newly_adapted


class SEURule(DecisionRule):
    """
    DYNAMO-M Subjective Expected Utility decision rule (the MVP science).

    Wraps the *validated* SEU kernels ported into ``dynamo_decision_bridge``
    (``_calc_eu_do_nothing`` / ``_calc_eu_adapt``), so this rule produces the
    same numbers that the Phase-1 Sec.12.1 battery and the native-DYNAMO-M
    cross-check validated.  An agent adapts when::

        EU_adapt > EU_do_nothing   and   not already adapted

    Affordability is *not* re-checked here: it is encoded inside
    ``_calc_eu_adapt`` as ``EU_adapt = -inf`` when the annualised cost exceeds
    ``income * expenditure_cap`` (avoids logic drift — architecture Sec.4.4).

    Parameters
    ----------
    config : DecisionConfig
        SEU behavioural parameters (``risk_aversion``, ``discount_rate``,
        ``decision_horizon``, ``loan_duration``, ``expenditure_cap``,
        ``amenity_weight``, ``error_interval`` …).
    rng : np.random.Generator or None
        Generator for the stochastic error terms.  Only used when
        ``config.error_interval > 0``.  When ``None`` a default generator is
        created (deterministic when ``error_interval == 0``).
    amenity_value : np.ndarray or None
        Optional per-agent amenity value (shape ``(n_agents,)``).  ``None``
        (default) uses zeros, matching the validated MVP configuration
        (``amenity`` is a post-MVP per-agent extension).

    Notes
    -----
    The last computed expected utilities are exposed as ``self.last_eu_adapt``
    and ``self.last_eu_do_nothing`` for diagnostics (``eu_history``).
    """

    _RISK_PERC_BASE: float = 1.6  # matches dynamo_decision_bridge

    def __init__(
        self,
        config: DecisionConfig,
        rng: np.random.Generator | None = None,
        amenity_value: np.ndarray | None = None,
    ):
        super().__init__(config)
        self._rng = rng if rng is not None else np.random.default_rng()
        self._amenity_value = (
            None if amenity_value is None
            else np.asarray(amenity_value, dtype=np.float32)
        )
        self.last_eu_adapt: np.ndarray | None = None
        self.last_eu_do_nothing: np.ndarray | None = None

    def should_adapt(
        self,
        agent_state: AgentState,
        damages_this_year: np.ndarray,
        damages_no_adapt: np.ndarray,
        damages_adapt: np.ndarray,
        event_freqs: np.ndarray,
        max_pot_dmg: np.ndarray,
        adaptation_costs: np.ndarray,
    ) -> np.ndarray:
        n_agents = agent_state.n_agents
        cfg = self.config

        # DYNAMO-M convention: expected-damage matrices shaped (n_events, n_agents)
        exp_dmg_no_measures = np.ascontiguousarray(
            damages_no_adapt.T, dtype=np.float32
        )
        exp_dmg_floodproof = np.ascontiguousarray(
            damages_adapt.T, dtype=np.float32
        )
        p_floods = np.asarray(event_freqs, dtype=np.float32)

        amenity_value = (
            self._amenity_value
            if self._amenity_value is not None
            else np.zeros(n_agents, dtype=np.float32)
        )

        T = np.full(n_agents, cfg.decision_horizon, dtype=np.int32)

        if cfg.error_interval > 0:
            error_terms = self._rng.uniform(
                1.0 - cfg.error_interval,
                1.0 + cfg.error_interval,
                size=n_agents,
            ).astype(np.float32)
        else:
            error_terms = np.ones(n_agents, dtype=np.float32)

        eu_do_nothing = _calc_eu_do_nothing(
            n_agents=n_agents,
            wealth=agent_state.wealth,
            income=agent_state.income,
            amenity_value=amenity_value,
            amenity_weight=cfg.amenity_weight,
            risk_perception=agent_state.risk_perception,
            expected_damages=exp_dmg_no_measures,
            adapted=agent_state.is_adapted.astype(np.int32),
            p_floods=p_floods,
            T=T,
            r=cfg.discount_rate,
            sigma=cfg.risk_aversion,
            error_terms=error_terms,
        )

        eu_adapt = _calc_eu_adapt(
            n_agents=n_agents,
            wealth=agent_state.wealth,
            income=agent_state.income,
            expenditure_cap=cfg.expenditure_cap,
            amenity_value=amenity_value,
            amenity_weight=cfg.amenity_weight,
            risk_perception=agent_state.risk_perception,
            expected_damages_adapt=exp_dmg_floodproof,
            adaptation_costs=np.asarray(adaptation_costs, dtype=np.float32),
            time_adapted=agent_state.time_adapted.astype(np.int32),
            loan_duration=cfg.loan_duration,
            p_floods=p_floods,
            T=T,
            r=cfg.discount_rate,
            sigma=cfg.risk_aversion,
            error_terms=error_terms,
        )

        self.last_eu_do_nothing = np.asarray(eu_do_nothing).copy()
        self.last_eu_adapt = np.asarray(eu_adapt).copy()

        newly_adapted = (eu_adapt - eu_do_nothing > 0) & (~agent_state.is_adapted)
        return newly_adapted
