"""
dynamo_live_rule.py
===================
The *live* coupling to DYNAMO-M: a decision rule that drives the **native**
``DecisionModule`` instead of the pure-NumPy kernels ported into
:mod:`floodadapt_abm._core.dynamo_decision_bridge`.

Role: preferred decision rule
-----------------------------
``DynamoLiveRule`` is the **recommended rule for application runs**, and the
seam through which any future DYNAMO-M coupling should arrive.  It calls the
upstream ``calcEU_do_nothing`` / ``calcEU_adapt`` / ``calcEU_insure`` with the
arrays the bridge assembles, so both household floodproofing and insurance are
decided by native DYNAMO-M code.  Paired with
:func:`~floodadapt_abm.mesa_native_full.run_mesa_native_full` (a real honeybees
``Model`` owning the clock) it forms the fully native path.

It doubles as the **parity oracle**: running it against the ported
:class:`~floodadapt_abm.decision_rule.SEURule` on an identical agent state must
yield the same expected utilities, and therefore identical decisions.  That
cross-check is what proves the port has not drifted from upstream.

Use :func:`preferred_decision_rule` to obtain this rule with an automatic,
parity-verified fallback to ``SEURule`` when DYNAMO-M is absent, or when the
configuration uses per-agent (risk-based) insurance premiums, which the native
kernel cannot express.

Why no full Mesa model is needed
--------------------------------
``calcEU_adapt`` (``decision_module.py`` lines 114-368) and
``calcEU_do_nothing`` (369-471) are near-pure array functions: they depend only
on the ``@njit`` static ``IterateThroughFlood`` and on ``self.error_terms_stay``
- **not** on ``self.model`` / ``self.agents``.  No full Mesa model is required.

Guarded / optional dependency
-----------------------------
The native module's top-level import
``from gravity_models.read_gravity_model import read_gravity_model`` only
resolves when the ``DYNAMO-M/DYNAMO-M`` package directory is on ``sys.path``.
This module therefore imports DYNAMO-M **lazily and defensively**:

* the import path can be supplied via the ``dynamo_path`` constructor argument
  or the ``DYNAMO_M_PATH`` environment variable (falling back to the
  conventional ``c:\\repos\\DYNAMO-M\\DYNAMO-M``);
* if DYNAMO-M cannot be imported, :data:`DYNAMO_M_AVAILABLE` is ``False`` and
  constructing a :class:`DynamoLiveRule` raises a clear
  :class:`DynamoMNotAvailable` error - **but importing FloodAdapt-ABM and using
  ``ThresholdRule`` / ``SEURule`` keeps working**.

Bit-parity configuration
-------------------------
For an exact cross-check set ``error_interval = 0`` (so
``error_terms_stay == 1``) and keep ``amenity_value = 0``.  Under that
configuration ``DynamoLiveRule`` and ``SEURule`` agree to float32 rounding and
produce identical boolean decisions.
"""
from __future__ import annotations

import importlib
import os
import sys
from types import SimpleNamespace

import numpy as np

from floodadapt_abm.agent_state import AgentState
from floodadapt_abm.coupling_config import DecisionConfig
from floodadapt_abm.decision_rule import (
    DecisionRule,
    SEURule,
    STATUS_PREFERRED,
)

__all__ = [
    "DynamoLiveRule",
    "DynamoMNotAvailable",
    "DYNAMO_M_AVAILABLE",
    "resolve_dynamo_path",
    "load_native_decision_module",
    "preferred_decision_rule",
]

#: Conventional checkout location of the DYNAMO-M *package* directory (the inner
#: ``DYNAMO-M/DYNAMO-M`` folder that contains ``decision_module.py`` and the
#: ``gravity_models`` package).
_DEFAULT_DYNAMO_PATH = r"c:\repos\DYNAMO-M\DYNAMO-M"


class DynamoMNotAvailable(ImportError):
    """Raised when the native DYNAMO-M ``DecisionModule`` cannot be imported."""


def resolve_dynamo_path(dynamo_path: str | None = None) -> str:
    """
    Resolve the DYNAMO-M package directory.

    Resolution order: explicit ``dynamo_path`` argument, then the
    ``DYNAMO_M_PATH`` environment variable, then :data:`_DEFAULT_DYNAMO_PATH`.
    """
    return (
        dynamo_path
        or os.environ.get("DYNAMO_M_PATH")
        or _DEFAULT_DYNAMO_PATH
    )


def load_native_decision_module(dynamo_path: str | None = None):
    """
    Import and return the native DYNAMO-M ``DecisionModule`` class.

    The DYNAMO-M package directory is prepended to ``sys.path`` (idempotently)
    so the module-level ``gravity_models`` import resolves.

    Raises
    ------
    DynamoMNotAvailable
        If the path does not exist or the import fails for any reason.
    """
    path = resolve_dynamo_path(dynamo_path)
    if not os.path.isdir(path):
        raise DynamoMNotAvailable(
            f"DYNAMO-M package directory not found: {path!r}. Set the "
            "DYNAMO_M_PATH environment variable or pass dynamo_path=... to "
            "DynamoLiveRule."
        )
    if path not in sys.path:
        sys.path.append(path)
    try:
        module = importlib.import_module("decision_module")
        return module.DecisionModule
    except Exception as exc:  # noqa: BLE001 - re-raise as a typed error
        raise DynamoMNotAvailable(
            f"Failed to import native DYNAMO-M DecisionModule from {path!r}: "
            f"{exc}"
        ) from exc


def _probe_availability() -> bool:
    """
    Lightweight check that DYNAMO-M *looks* importable, WITHOUT importing it or
    mutating ``sys.path``.  The real import (and any ``sys.path`` change) is
    deferred to :func:`load_native_decision_module`, called when a
    :class:`DynamoLiveRule` is actually constructed.
    """
    path = resolve_dynamo_path()
    return (
        os.path.isfile(os.path.join(path, "decision_module.py"))
        and os.path.isdir(os.path.join(path, "gravity_models"))
    )


#: ``True`` when the native DYNAMO-M ``DecisionModule`` is importable in this
#: environment.  Probed once at import time using the default resolution.
DYNAMO_M_AVAILABLE: bool = _probe_availability()


def _build_stub_model(error_interval: float, seed: int) -> SimpleNamespace:
    """
    Build the minimal object graph ``DecisionModule`` needs.

    ``DecisionModule.__init__`` reads ``model.settings['decisions']
    ['error_interval']`` and ``sample_error_terms`` uses
    ``model.random_module.random_state``.  We provide both, plus an empty
    ``args`` namespace, so the module constructs without a full Mesa model.
    """
    return SimpleNamespace(
        settings={"decisions": {"error_interval": error_interval}},
        random_module=SimpleNamespace(random_state=np.random.default_rng(seed)),
        args=SimpleNamespace(),
    )


class DynamoLiveRule(DecisionRule):
    """
    Decision rule that delegates to the **native** DYNAMO-M ``DecisionModule``.

    Calls upstream ``calcEU_do_nothing`` / ``calcEU_adapt`` / ``calcEU_insure``
    instead of the ported kernels, so household floodproofing **and** insurance
    are decided by native DYNAMO-M code.

    .. note::
       **Status: preferred.**  This is the recommended rule for application
       runs, and the seam through which any future DYNAMO-M coupling (migration,
       the government agent) should arrive.  Combine it with
       :func:`~floodadapt_abm.mesa_native_full.run_mesa_native_full`, where a
       real honeybees ``Model`` owns the clock, for the fully native path.

       Two practical constraints.  It needs a DYNAMO-M checkout (an optional
       dependency; see ``dynamo_path`` / ``DYNAMO_M_PATH``), and it **cannot
       represent per-agent insurance premiums**: native ``calcEU_insure``
       discounts ``premium.mean()``, so ``insurance_pricing="risk_based"``
       must use :class:`~floodadapt_abm.decision_rule.SEURule` instead
       (:meth:`decide` raises rather than silently averaging).
       :func:`preferred_decision_rule` applies exactly this policy.

       Parallel runs are safe: :meth:`clone` gives each worker its own native
       module.  The native kernels hold the GIL, though, so a parallel native
       run is correct but does not scale across threads.

    Parameters
    ----------
    config : DecisionConfig
        SEU behavioural parameters (identical semantics to ``SEURule``).
    dynamo_path : str or None
        Location of the DYNAMO-M package directory.  ``None`` uses the
        ``DYNAMO_M_PATH`` environment variable or the conventional default.
    amenity_value : np.ndarray or None
        Optional per-agent amenity value.  ``None`` uses zeros (the validated
        MVP / bit-parity configuration).
    rng : np.random.Generator or None
        Generator for the stochastic error terms when
        ``config.error_interval > 0``.  Ignored when ``error_interval == 0``
        (the bit-parity configuration, ``error_terms_stay == 1``).
    geom_id : str
        Label forwarded to the native methods (used only in their diagnostic
        prints).

    Raises
    ------
    DynamoMNotAvailable
        If the native ``DecisionModule`` cannot be imported.

    Notes
    -----
    The last computed expected utilities are exposed as ``self.last_eu_adapt``
    and ``self.last_eu_do_nothing`` (mirroring ``SEURule``), so the rule plugs
    straight into ``SimulationEngine`` with ``track_eu=True``.
    """

    STATUS: str = STATUS_PREFERRED

    def __init__(
        self,
        config: DecisionConfig,
        dynamo_path: str | None = None,
        amenity_value: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        geom_id: str = "floodadapt_abm",
    ):
        super().__init__(config)
        decision_module_cls = load_native_decision_module(dynamo_path)

        self.geom_id = geom_id
        self._rng = rng if rng is not None else np.random.default_rng()
        self._amenity_value = (
            None if amenity_value is None
            else np.asarray(amenity_value, dtype=np.float32)
        )
        # Retained so clone() can build an independent native module per worker
        # instead of sharing this one (see clone()).
        self._dynamo_path = dynamo_path
        self._decision_module_cls = decision_module_cls

        self._dm = self._new_decision_module()

        self.last_eu_adapt: np.ndarray | None = None
        self.last_eu_do_nothing: np.ndarray | None = None
        self.last_eu_insure: np.ndarray | None = None

    # -----------------------------------------------------------------------
    def _new_decision_module(self):
        """
        Instantiate a fresh native ``DecisionModule`` against a stub model.

        ``agents`` is unused by the ``calcEU_*`` methods (only by
        ``load_gravity_models``, which this adapter never calls), so a minimal
        stub suffices.
        """
        stub_model = _build_stub_model(
            error_interval=self.config.error_interval, seed=0
        )
        return self._decision_module_cls(agents=None, model=stub_model)

    def clone(self, rng_seed: int | None = None) -> "DynamoLiveRule":
        """
        Return a worker-safe copy with its **own** native ``DecisionModule``.

        The base-class ``clone`` is a shallow copy, which would leave every
        parallel worker sharing one native module.  That is unsafe here because
        :meth:`should_adapt` assigns ``self._dm.error_terms_stay`` on every
        call, so concurrent sequences would overwrite each other's error terms.
        Building a fresh module per clone removes that race.

        Note on performance: the native kernels are ``@njit`` **without**
        ``nogil=True``, so they hold the GIL. Cloning makes
        ``engine.run(n_jobs>1)`` *correct* with this rule, but the native work
        still serialises across threads. Use
        :class:`~floodadapt_abm.decision_rule.SEURule` (vectorised NumPy, which
        releases the GIL) when parallel throughput matters; the two are
        parity-gated, so results are interchangeable.
        """
        new = super().clone(rng_seed=rng_seed)
        new._dm = new._new_decision_module()
        return new

    # -----------------------------------------------------------------------
    def _error_terms(self, n_agents: int) -> np.ndarray:
        cfg = self.config
        if cfg.error_interval > 0:
            return self._rng.uniform(
                1.0 - cfg.error_interval,
                1.0 + cfg.error_interval,
                size=n_agents,
            ).astype(np.float32)
        return np.ones(n_agents, dtype=np.float32)

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

        # Native calcEU_* read self.error_terms_stay directly.  We set it here
        # (rather than calling sample_error_terms, which needs regions).
        self._dm.error_terms_stay = self._error_terms(n_agents)

        eu_do_nothing = self._dm.calcEU_do_nothing(
            geom_id=self.geom_id,
            n_agents=n_agents,
            wealth=np.asarray(agent_state.wealth, dtype=np.float32),
            income=np.asarray(agent_state.income, dtype=np.float32),
            amenity_value=amenity_value,
            amenity_weight=cfg.amenity_weight,
            risk_perception=np.asarray(
                agent_state.risk_perception, dtype=np.float32
            ),
            expected_damages=exp_dmg_no_measures,
            adapted=agent_state.is_adapted.astype(np.int32),
            p_floods=p_floods,
            T=T,
            r=cfg.discount_rate,
            sigma=cfg.risk_aversion,
        )

        eu_adapt = self._dm.calcEU_adapt(
            geom_id=self.geom_id,
            n_agents=n_agents,
            wealth=np.asarray(agent_state.wealth, dtype=np.float32),
            income=np.asarray(agent_state.income, dtype=np.float32),
            expendature_cap=cfg.expenditure_cap,  # native spelling
            amenity_value=amenity_value,
            amenity_weight=cfg.amenity_weight,
            risk_perception=np.asarray(
                agent_state.risk_perception, dtype=np.float32
            ),
            expected_damages_adapt=exp_dmg_floodproof,
            adaptation_costs=np.asarray(adaptation_costs, dtype=np.float32),
            time_adapted=agent_state.time_adapted.astype(np.int32),
            loan_duration=cfg.loan_duration,
            p_floods=p_floods,
            T=T,
            r=cfg.discount_rate,
            sigma=cfg.risk_aversion,
        )

        self.last_eu_do_nothing = np.asarray(eu_do_nothing).copy()
        self.last_eu_adapt = np.asarray(eu_adapt).copy()

        newly_adapted = (
            (self.last_eu_adapt - self.last_eu_do_nothing > 0)
            & (~agent_state.is_adapted)
        )
        return newly_adapted

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
        Native three-way decision (do nothing / adapt / insure).

        Runs :meth:`should_adapt` first — which sets
        ``self._dm.error_terms_stay`` once and computes the two native EUs —
        then, when a premium is offered, adds the native ``calcEU_insure``
        (which reuses the same ``error_terms_stay``, matching native
        semantics of one error draw per node step) and applies the native
        three-way comparison (``coastal_nodes.py:1938-1952``), restricted to
        non-adapted agents (floodproofing stays sticky — same documented
        deviation as :class:`~floodadapt_abm.decision_rule.SEURule`).

        Raises
        ------
        ValueError
            If ``insurance_premium`` varies between agents.  Native
            ``calcEU_insure`` discounts ``premium.mean()``
            (``decision_module.py:337``), because native's insurer only ever
            issues one flat community premium.  Handing it per-agent premiums
            would silently charge every household the pool average, so
            risk-based pricing must use ``SEURule`` instead, whose kernel
            discounts each agent's own premium.
        """
        from floodadapt_abm.decision_rule import (
            ACTION_ADAPT,
            ACTION_DO_NOTHING,
            ACTION_INSURE,
        )

        if insurance_premium is not None:
            prem = np.asarray(insurance_premium, dtype=np.float64)
            if prem.size > 1 and not np.allclose(prem, prem.flat[0]):
                raise ValueError(
                    "DynamoLiveRule received a per-agent insurance premium, but "
                    "native calcEU_insure discounts premium.mean() and cannot "
                    "express per-agent pricing: every household would be charged "
                    "the pool average. Use SEURule for insurance_pricing="
                    '"risk_based" (it is parity-gated against this rule under '
                    "the flat community premium), or keep "
                    'insurance_pricing="community".'
                )

        newly_adapted = self.should_adapt(
            agent_state=agent_state,
            damages_this_year=damages_this_year,
            damages_no_adapt=damages_no_adapt,
            damages_adapt=damages_adapt,
            event_freqs=event_freqs,
            max_pot_dmg=max_pot_dmg,
            adaptation_costs=adaptation_costs,
        )
        if insurance_premium is None:
            self.last_eu_insure = None
            return np.where(
                newly_adapted, ACTION_ADAPT, ACTION_DO_NOTHING
            ).astype(np.int8)

        n_agents = agent_state.n_agents
        cfg = self.config
        exp_dmg_no_measures = np.ascontiguousarray(
            damages_no_adapt.T, dtype=np.float32
        )
        amenity_value = (
            self._amenity_value
            if self._amenity_value is not None
            else np.zeros(n_agents, dtype=np.float32)
        )
        T = np.full(n_agents, cfg.decision_horizon, dtype=np.int32)

        eu_insure = self._dm.calcEU_insure(
            geom_id=self.geom_id,
            n_agents=n_agents,
            wealth=np.asarray(agent_state.wealth, dtype=np.float32),
            income=np.asarray(agent_state.income, dtype=np.float32),
            expendature_cap=cfg.expenditure_cap,  # native spelling
            amenity_value=amenity_value,
            amenity_weight=cfg.amenity_weight,
            risk_perception=np.asarray(
                agent_state.risk_perception, dtype=np.float32
            ),
            expected_damages=exp_dmg_no_measures,
            premium=np.asarray(insurance_premium, dtype=np.float32),
            p_floods=np.asarray(event_freqs, dtype=np.float32),
            T=T,
            r=cfg.discount_rate,
            sigma=cfg.risk_aversion,
            deductable=cfg.insurance_deductible,  # native spelling
        )
        self.last_eu_insure = np.asarray(eu_insure).copy()

        eu_do_nothing = self.last_eu_do_nothing
        eu_adapt = self.last_eu_adapt
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


def preferred_decision_rule(
    config: DecisionConfig,
    dynamo_path: str | None = None,
    amenity_value: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> DecisionRule:
    """
    Return the preferred decision rule for this configuration and environment.

    Encodes the project's rule policy in one place instead of hand-written
    ``if DYNAMO_M_AVAILABLE`` branches.  :class:`DynamoLiveRule` (the live
    coupling to the native DYNAMO-M ``DecisionModule``) is returned whenever it
    is both **available** and **able to express the configuration**.  The
    parity-gated :class:`~floodadapt_abm.decision_rule.SEURule` is returned
    otherwise; the two agree to a relative EU error < 1e-4 and produce
    identical actions, so results stay comparable either way.

    The port is selected in exactly two cases:

    1. **DYNAMO-M is not installed.** It is an optional dependency.
    2. **Per-agent insurance premiums are configured**
       (``include_insurance=True`` with ``insurance_pricing`` other than
       ``"community"``).  Native ``calcEU_insure`` discounts
       ``premium.mean()`` (``decision_module.py:337``) because native's insurer
       only ever issues one flat community premium, so it cannot represent
       risk-based pricing at all.  This is a capability limit, not a
       performance choice.

    Parallel runs are safe with either rule: :meth:`DynamoLiveRule.clone` gives
    each worker its own native module.  Note though that the native kernels are
    ``@njit`` without ``nogil=True`` and therefore hold the GIL, so a parallel
    run driven by the native rule is correct but does not scale across threads.
    Pass the port explicitly when parallel throughput matters.

    Parameters
    ----------
    config : DecisionConfig
        SEU behavioural parameters, passed to whichever rule is built.  Its
        ``include_insurance`` / ``insurance_pricing`` fields decide whether the
        native rule can be used at all.
    dynamo_path : str or None
        Location of the DYNAMO-M package directory.  ``None`` uses the
        ``DYNAMO_M_PATH`` environment variable or the conventional default.
    amenity_value : np.ndarray or None
        Optional per-agent amenity value, forwarded to the rule.
    rng : np.random.Generator or None
        Generator for the stochastic error terms when
        ``config.error_interval > 0``.

    Returns
    -------
    rule : DecisionRule
        A ``DynamoLiveRule`` when it is available and expressive enough, else
        an ``SEURule``.  Inspect ``rule.STATUS`` to see which one you got.

    Examples
    --------
    >>> from floodadapt_abm import CouplingConfig, preferred_decision_rule
    >>> cfg = CouplingConfig()
    >>> rule = preferred_decision_rule(cfg.decision)
    >>> rule.STATUS in ("preferred", "reference")
    True
    """
    per_agent_premium = (
        getattr(config, "include_insurance", False)
        and getattr(config, "insurance_pricing", "community") != "community"
    )
    if not per_agent_premium:
        try:
            return DynamoLiveRule(
                config,
                dynamo_path=dynamo_path,
                amenity_value=amenity_value,
                rng=rng,
            )
        except DynamoMNotAvailable:
            pass  # optional dependency absent: fall through to the port
    return SEURule(config, rng=rng, amenity_value=amenity_value)
