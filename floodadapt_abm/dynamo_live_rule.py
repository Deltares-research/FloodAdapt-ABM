"""
dynamo_live_rule.py
===================
Phase 4a of the FloodAdapt-ABM x DYNAMO-M coupling: a *live* decision rule that
drives the **native** DYNAMO-M ``DecisionModule`` instead of the pure-NumPy
kernels ported into :mod:`floodadapt_abm._core.dynamo_decision_bridge`.

-----------------------
``DynamoLiveRule`` is a thin adapter that calls the upstream
``DecisionModule.calcEU_do_nothing`` / ``DecisionModule.calcEU_adapt`` with the
same arrays the bridge assembles.  Its primary role is to guarantee that the
ported :class:`~floodadapt_abm.decision_rule.SEURule` has **not drifted** from
upstream DYNAMO-M: running both on an identical agent state must yield the same
expected utilities (and therefore identical adaptation decisions).  This is the
mechanism that executes the Phase-1 cross-check gate.

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
from floodadapt_abm.decision_rule import DecisionRule

__all__ = [
    "DynamoLiveRule",
    "DynamoMNotAvailable",
    "DYNAMO_M_AVAILABLE",
    "resolve_dynamo_path",
    "load_native_decision_module",
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

    Drop-in replacement for :class:`~floodadapt_abm.decision_rule.SEURule` that
    calls upstream ``calcEU_do_nothing`` / ``calcEU_adapt`` instead of the
    ported kernels.  Used as a parity oracle (and as the seam for a future
    fully Mesa-native integration, Phase 4b).

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

        # Instantiate the native module against a minimal stub model.  agents is
        # unused by calcEU_* (only by load_gravity_models, which we never call).
        stub_model = _build_stub_model(
            error_interval=config.error_interval, seed=0
        )
        self._dm = decision_module_cls(agents=None, model=stub_model)

        self.last_eu_adapt: np.ndarray | None = None
        self.last_eu_do_nothing: np.ndarray | None = None

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
