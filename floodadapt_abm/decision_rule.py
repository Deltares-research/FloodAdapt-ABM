"""
decision_rule.py
================
Pluggable decision rules for the unified ``SimulationEngine``.

``SimulationEngine`` owns *time and data* (NetCDF loading, interpolation,
stochastic event drawing, state tracking, the year loop); a ``DecisionRule``
owns *behaviour* (whether each household adapts this year).  Swapping the rule
is the only change needed to switch between the legacy threshold heuristic and
the DYNAMO-M SEU science, without touching the engine.

Rules provided, by status
-------------------------
Every rule carries a machine-readable ``STATUS`` class attribute saying how it
is meant to be used.  The preferred path is the live coupling to native
DYNAMO-M; the others are a parity-gated port, a baseline, and legacy code.

============================  ==============  ==================================
Rule                          ``STATUS``      Use it for
============================  ==============  ==================================
``DynamoLiveRule``            preferred       Application runs.  Calls the
(``dynamo_live_rule.py``)                     **native** DYNAMO-M
                                              ``DecisionModule`` for both
                                              floodproofing and insurance.
``SEURule``                   reference       Same science, pure-NumPy port,
                                              parity-gated against the
                                              preferred rule.  Used when
                                              DYNAMO-M is absent, and required
                                              for ``engine.run(n_jobs>1)``.
``ThresholdRule``             experiment      Legacy baseline: adapt when this
                                              year's damage exceeds
                                              ``damage_threshold *
                                              max_pot_dmg``.  Ignores income,
                                              perception and insurance.
============================  ==============  ==================================

Use :func:`floodadapt_abm.dynamo_live_rule.preferred_decision_rule` to get the
preferred rule for the current environment, with an automatic, parity-verified
fallback to ``SEURule`` when DYNAMO-M is not installed.

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
    _calc_eu_insure,
)

#: Action codes returned by :meth:`DecisionRule.decide`.
ACTION_DO_NOTHING: int = 0
ACTION_ADAPT: int = 1
ACTION_INSURE: int = 2

# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------
# Every rule and driver carries a ``STATUS`` tag saying how it is meant to be
# used.  The tag is machine-readable (so tests, docs and user code can query it)
# and mirrors the roles stated in ``docs/architecture.md``.

#: Recommended for application runs: the live coupling to native DYNAMO-M
#: (household adaptation *and* insurance, and any future DYNAMO-M coupling).
STATUS_PREFERRED: str = "preferred"

#: Validated pure-NumPy port of the preferred path.  Bit-parity-gated against it
#: (relative EU error < 1e-4, identical actions), so results are interchangeable.
#: Used automatically when DYNAMO-M is not installed, and required for the
#: parallel Monte-Carlo backend (``engine.run(n_jobs>1)``), which the native
#: module does not support.
STATUS_REFERENCE: str = "reference"

#: Baseline or comparison behaviour, not the coupled science.  Kept so the
#: pre-coupling model can be reproduced and compared against.
STATUS_EXPERIMENT: str = "experiment"

#: Retained only to prove the bit-parity contract; not an application path.
STATUS_VERIFICATION: str = "verification"

#: Superseded; kept for backward compatibility only.
STATUS_DEPRECATED: str = "deprecated"


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

    Attributes
    ----------
    STATUS : str
        How this rule is meant to be used; one of :data:`STATUS_PREFERRED`,
        :data:`STATUS_REFERENCE`, :data:`STATUS_EXPERIMENT`,
        :data:`STATUS_VERIFICATION` or :data:`STATUS_DEPRECATED`.  Third-party
        subclasses inherit :data:`STATUS_EXPERIMENT` unless they override it.
        Use :func:`floodadapt_abm.dynamo_live_rule.preferred_decision_rule` to
        obtain the preferred rule for the current environment.
    """

    #: Default for third-party subclasses; concrete rules below override it.
    STATUS: str = STATUS_EXPERIMENT

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
        if hasattr(new, "last_eu_insure"):
            new.last_eu_insure = None
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

    def decide(
        self,
        agent_state: AgentState,
        damages_this_year: np.ndarray,
        damages_no_adapt: np.ndarray,
        damages_adapt: np.ndarray,
        event_freqs: np.ndarray,
        max_pot_dmg: np.ndarray,
        adaptation_costs: np.ndarray,
        insurance_premium: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Choose one action per agent: do nothing, adapt, or insure.

        The default implementation delegates to :meth:`should_adapt` and
        never insures, so existing two-way rules (including third-party
        subclasses) work unchanged.  Rules that support insurance
        (:class:`SEURule`, ``DynamoLiveRule``) override this method.

        Parameters
        ----------
        agent_state, damages_this_year, damages_no_adapt, damages_adapt,
        event_freqs, max_pot_dmg, adaptation_costs
            As in :meth:`should_adapt`.
        insurance_premium : np.ndarray or None
            Annual premium per agent for the coming year; ``None`` means
            insurance is not offered (the option is skipped entirely).

        Returns
        -------
        actions : np.ndarray[int8], shape (n_agents,)
            ``ACTION_DO_NOTHING`` (0), ``ACTION_ADAPT`` (1), or
            ``ACTION_INSURE`` (2) per agent.
        """
        newly_adapted = self.should_adapt(
            agent_state=agent_state,
            damages_this_year=damages_this_year,
            damages_no_adapt=damages_no_adapt,
            damages_adapt=damages_adapt,
            event_freqs=event_freqs,
            max_pot_dmg=max_pot_dmg,
            adaptation_costs=adaptation_costs,
        )
        return np.where(newly_adapted, ACTION_ADAPT, ACTION_DO_NOTHING).astype(
            np.int8
        )


class ThresholdRule(DecisionRule):
    """
    Legacy reactive heuristic (the rule the coupling replaces).

    An agent adapts once the realised damage it suffered *this year* exceeds a
    fixed fraction of its maximum potential damage::

        adapt  if  damages_this_year / max_pot_dmg > damage_threshold

    This reproduces ``ABMSimulator._simulate_damage_history`` bit-for-bit
    (same masking on ``not_adapted & max_pot_dmg > 0``).  All SEU-specific
    arguments are ignored.

    .. note::
       **Status: experiment.**  This is a baseline for comparison, not the
       coupled science.  It reacts *after* damage occurs and ignores income,
       affordability, risk perception and insurance.  Use it to reproduce the
       pre-coupling model or as a reference arm; use the preferred rule
       (:class:`~floodadapt_abm.dynamo_live_rule.DynamoLiveRule`) for
       application runs.

    Parameters
    ----------
    config : DecisionConfig
        Used only for interface uniformity; the threshold itself is passed
        separately (defaults to ``0.30``, the legacy value).
    damage_threshold : float
        Fraction of ``max_pot_dmg`` above which an agent adapts.  Default
        ``0.30``.
    """

    STATUS: str = STATUS_EXPERIMENT

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
    DYNAMO-M Subjective Expected Utility decision rule (pure-NumPy port).

    Wraps the *validated* SEU kernels ported into ``dynamo_decision_bridge``
    (``_calc_eu_do_nothing`` / ``_calc_eu_adapt`` / ``_calc_eu_insure``).  An
    agent adapts when::

        EU_adapt > EU_do_nothing   and   not already adapted

    Affordability is *not* re-checked here: it is encoded inside
    ``_calc_eu_adapt`` as ``EU_adapt = -inf`` when the annualised cost exceeds
    ``income * expenditure_cap`` (avoids logic drift).

    .. note::
       **Status: reference.**  This is the parity-gated port of the preferred
       rule (:class:`~floodadapt_abm.dynamo_live_rule.DynamoLiveRule`): the two
       agree to a relative EU error < 1e-4 and produce identical actions, so
       their results are interchangeable.  Prefer this rule when

       * DYNAMO-M is not installed (it is an optional dependency), or
       * the run needs ``engine.run(n_jobs>1)``: the native module keeps
         mutable per-call state on one shared object, so it is not safe to
         clone across worker threads, whereas this rule is.

       Otherwise use the preferred rule.  See
       :func:`~floodadapt_abm.dynamo_live_rule.preferred_decision_rule`.

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

    STATUS: str = STATUS_REFERENCE

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
        self.last_eu_insure: np.ndarray | None = None

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
        """Two-way decision: delegate to :meth:`decide` without insurance."""
        actions = self.decide(
            agent_state=agent_state,
            damages_this_year=damages_this_year,
            damages_no_adapt=damages_no_adapt,
            damages_adapt=damages_adapt,
            event_freqs=event_freqs,
            max_pot_dmg=max_pot_dmg,
            adaptation_costs=adaptation_costs,
            insurance_premium=None,
        )
        return actions == ACTION_ADAPT

    def decide(
        self,
        agent_state: AgentState,
        damages_this_year: np.ndarray,
        damages_no_adapt: np.ndarray,
        damages_adapt: np.ndarray,
        event_freqs: np.ndarray,
        max_pot_dmg: np.ndarray,
        adaptation_costs: np.ndarray,
        insurance_premium: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        SEU three-way decision (do nothing / adapt / insure).

        One stochastic error-terms array is drawn per call and reused across
        all EU evaluations (mirroring native DYNAMO-M, where
        ``error_terms_stay`` is sampled once per node step).  When
        ``insurance_premium`` is ``None`` the insurance branch is skipped
        entirely and the decision reduces bit-exactly to the historical
        two-way rule.

        Native comparison semantics (``coastal_nodes.py:1938-1952``)::

            adapt  = (EU_adapt  > EU_do_nothing) & (EU_adapt >= EU_insure)
            insure = (EU_insure > EU_do_nothing) & (EU_insure > EU_adapt)

        both restricted to currently non-adapted agents (keeping physical
        floodproofing sticky is a documented deviation from native, which
        lets adapted agents switch to insurance).
        """
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

        if insurance_premium is None:
            # Historical two-way comparison, bit-exact.
            self.last_eu_insure = None
            newly_adapted = (
                (eu_adapt - eu_do_nothing > 0) & (~agent_state.is_adapted)
            )
            return np.where(
                newly_adapted, ACTION_ADAPT, ACTION_DO_NOTHING
            ).astype(np.int8)

        eu_insure = _calc_eu_insure(
            n_agents=n_agents,
            wealth=agent_state.wealth,
            income=agent_state.income,
            expenditure_cap=cfg.expenditure_cap,
            amenity_value=amenity_value,
            amenity_weight=cfg.amenity_weight,
            risk_perception=agent_state.risk_perception,
            expected_damages=exp_dmg_no_measures,
            premium=np.asarray(insurance_premium, dtype=np.float32),
            p_floods=p_floods,
            T=T,
            r=cfg.discount_rate,
            sigma=cfg.risk_aversion,
            error_terms=error_terms,
            deductible=cfg.insurance_deductible,
        )
        self.last_eu_insure = np.asarray(eu_insure).copy()

        not_adapted = ~agent_state.is_adapted
        adapt = (
            (eu_adapt > eu_do_nothing) & (eu_adapt >= eu_insure) & not_adapted
        )
        insure = (
            (eu_insure > eu_do_nothing) & (eu_insure > eu_adapt) & not_adapted
        )
        actions = np.full(n_agents, ACTION_DO_NOTHING, dtype=np.int8)
        actions[adapt] = ACTION_ADAPT
        actions[insure] = ACTION_INSURE
        return actions
