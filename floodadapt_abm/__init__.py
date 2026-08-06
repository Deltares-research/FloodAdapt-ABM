"""
floodadapt_abm
==============
FloodAdapt-ABM: agent-based flood adaptation simulation coupled with
the DYNAMO-M Subjective Expected Utility decision framework.

The preferred path
------------------
**The recommended way to run a simulation is the fully native coupling**:
``run_mesa_native_full`` (a real honeybees ``Model`` owns the clock) driving
``DynamoLiveRule`` (the native DYNAMO-M ``DecisionModule`` decides both
floodproofing and insurance)::

    from floodadapt_abm import (
        SimulationEngine, CouplingConfig, preferred_decision_rule,
        run_mesa_native_full,
    )

    config = CouplingConfig()
    engine = SimulationEngine(ds=ds, config=config)
    engine.decision_rule = preferred_decision_rule(config.decision)
    results = run_mesa_native_full(engine, slr_values, no_seq=5, seed=42)

``preferred_decision_rule`` returns ``DynamoLiveRule`` when DYNAMO-M is
installed, and otherwise falls back to the parity-gated ``SEURule`` port, whose
results are interchangeable (relative EU error < 1e-4, identical actions).
Any future DYNAMO-M coupling (migration, the government agent) should arrive
through the same seam and inherit the same preference.

Everything else is a comparison baseline or a verification path.

Public API — roles at a glance
------------------------------
Decision rules (``rule.STATUS`` carries the tag):

==========================  ==============  ==================================
Symbol                      ``STATUS``      Role
==========================  ==============  ==================================
``DynamoLiveRule``          preferred       Native DYNAMO-M decisions
                                            (floodproofing + insurance)
``SEURule``                 reference       Parity-gated NumPy port; fallback,
                                            and required for ``n_jobs>1``
``ThresholdRule``           experiment      Legacy damage-threshold baseline
==========================  ==============  ==================================

Drivers (which object owns the clock):

==============================  ==============  ==============================
Symbol                          Status          Role
==============================  ==============  ==============================
``run_mesa_native_full`` /      preferred       Real honeybees ``Model`` owns
``FloodAdaptSLRModelFull``                      the clock.  Application runs.
``SimulationEngine``            kernel          The single numeric kernel every
                                                driver delegates to, plus the
                                                parallel Monte-Carlo backend
                                                ``run(n_jobs=...)`` used for
                                                experiments and sweeps.
``run_mesa_native`` /           verification    Framework-free mirror of the
``FloodAdaptSLRModel``                          tick loop; bit-parity gate.
``ABMSimulator``                deprecated      Legacy simulator, backward
                                                compatibility only.
==============================  ==============  ==============================

All three drivers delegate every numeric operation to ``SimulationEngine.step``
with the same RNG stream, and are gated to agree bit-for-bit.

AgentState
    Per-agent state container for the engine.
CouplingConfig / DecisionConfig / NetCDFMappingConfig
    Configuration dataclasses.  ``CouplingConfig.legacy()`` reproduces the
    pre-2026-07 behaviour bit-exactly (verification harnesses pin it).

Note on setup_lookup_table
--------------------------
``setup_lookup_table`` (stage 1 pipeline) is intentionally NOT imported
here because it requires the full ``flood-adapt`` library, which is not
available in all environments (e.g. the ``dynamom`` conda env).

Import it explicitly when needed::

    from floodadapt_abm.setup_lookup_table import create_lookup_table

Internal plumbing (_core, not recommended for direct use)
---------------------------------------------------------
DynamoDecisionBridge
    Internal data-plumbing layer; composed by ``SimulationEngine``.
"""

from floodadapt_abm.abm_simulator import ABMSimulator
from floodadapt_abm.coupling_config import (
    CouplingConfig,
    DecisionConfig,
    NetCDFMappingConfig,
)
from floodadapt_abm.agent_state import AgentState
from floodadapt_abm.decision_rule import (
    ACTION_ADAPT,
    ACTION_DO_NOTHING,
    ACTION_INSURE,
    STATUS_DEPRECATED,
    STATUS_EXPERIMENT,
    STATUS_PREFERRED,
    STATUS_REFERENCE,
    STATUS_VERIFICATION,
    DecisionRule,
    SEURule,
    ThresholdRule,
)
from floodadapt_abm.simulation_engine import SimulationEngine
from floodadapt_abm.event_utils import draw_year_events, generate_event_sequences

# For backward compat: DynamoDecisionBridge moved to _core, but re-export here
from floodadapt_abm._core import DynamoDecisionBridge

# PREFERRED decision rule: the live coupling to native DYNAMO-M. Import is
# guarded so the package still works when DYNAMO-M is not installed/importable
# (DYNAMO_M_AVAILABLE is False, and constructing DynamoLiveRule then raises
# DynamoMNotAvailable). preferred_decision_rule() applies the fallback policy.
from floodadapt_abm.dynamo_live_rule import (
    DynamoLiveRule,
    DynamoMNotAvailable,
    DYNAMO_M_AVAILABLE,
    preferred_decision_rule,
)

# Phase 4b: Mesa-native driving (time-ownership inversion). Framework-free
# mirror of DYNAMO-M's SLRModel.step() tick loop; reuses the shared kernels.
from floodadapt_abm.mesa_native import (
    FloodAdaptSLRModel,
    Agents as MesaAgents,
    CoastalNodePopulation,
    run_mesa_native,
)

# MAIN ENGINE (Phase 4b-full, native-class integration): subclasses the real
# honeybees Model (owns time); decisions can route through the native DYNAMO-M
# DecisionModule via DynamoLiveRule. honeybees is a core dependency; the import
# guard is retained defensively (HONEYBEES_AVAILABLE False -> construction
# raises HoneybeesNotAvailable with an actionable message).
from floodadapt_abm.mesa_native_full import (
    FloodAdaptSLRModelFull,
    AgentsFull,
    CoastalNodePopulationFull,
    run_mesa_native_full,
    HoneybeesNotAvailable,
    HONEYBEES_AVAILABLE,
)

__all__ = [
    "SimulationEngine",
    "ABMSimulator",
    "DecisionRule",
    "ThresholdRule",
    "SEURule",
    "ACTION_DO_NOTHING",
    "ACTION_ADAPT",
    "ACTION_INSURE",
    "STATUS_PREFERRED",
    "STATUS_REFERENCE",
    "STATUS_EXPERIMENT",
    "STATUS_VERIFICATION",
    "STATUS_DEPRECATED",
    "DynamoLiveRule",
    "DynamoMNotAvailable",
    "DYNAMO_M_AVAILABLE",
    "preferred_decision_rule",
    "FloodAdaptSLRModel",
    "MesaAgents",
    "CoastalNodePopulation",
    "run_mesa_native",
    "FloodAdaptSLRModelFull",
    "AgentsFull",
    "CoastalNodePopulationFull",
    "run_mesa_native_full",
    "HoneybeesNotAvailable",
    "HONEYBEES_AVAILABLE",
    "AgentState",
    "CouplingConfig",
    "DecisionConfig",
    "NetCDFMappingConfig",
    "draw_year_events",
    "generate_event_sequences",
    "DynamoDecisionBridge",  # backward compat
]

