"""
floodadapt_abm
==============
FloodAdapt-ABM: agent-based flood adaptation simulation coupled with
the DYNAMO-M Subjective Expected Utility decision framework.

Public API
----------
SimulationEngine (recommended)
    Unified simulation engine with pluggable decision rules
    (Strategy Pattern).  Owns time, data plumbing, event generation, and
    lifespan-dryproof reset logic.  Use this for new code.
ABMSimulator
    **Deprecated** legacy threshold-rule simulator (stage-2 pipeline),
    retained for backward compatibility and the Gate-1 regression test.
    New code should use ``SimulationEngine`` + ``ThresholdRule`` instead.
DecisionRule / ThresholdRule / SEURule
    Pluggable decision rules for ``SimulationEngine``.
AgentState
    Per-agent state container for the engine.
CouplingConfig / DecisionConfig / NetCDFMappingConfig
    Configuration dataclasses.

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
from floodadapt_abm.decision_rule import DecisionRule, ThresholdRule, SEURule
from floodadapt_abm.simulation_engine import SimulationEngine
from floodadapt_abm.event_utils import draw_year_events, generate_event_sequences

# For backward compat: DynamoDecisionBridge moved to _core, but re-export here
from floodadapt_abm._core import DynamoDecisionBridge

# Phase 4a: live DYNAMO-M parity rule. Import is guarded so the package still
# works when DYNAMO-M is not installed/importable (DYNAMO_M_AVAILABLE is False,
# and constructing DynamoLiveRule then raises DynamoMNotAvailable).
from floodadapt_abm.dynamo_live_rule import (
    DynamoLiveRule,
    DynamoMNotAvailable,
    DYNAMO_M_AVAILABLE,
)

# Phase 4b: Mesa-native driving (time-ownership inversion). Framework-free
# mirror of DYNAMO-M's SLRModel.step() tick loop; reuses the shared kernels.
from floodadapt_abm.mesa_native import (
    FloodAdaptSLRModel,
    Agents as MesaAgents,
    CoastalNodePopulation,
    run_mesa_native,
)

# Phase 4b-full: native-class integration. Subclasses the real honeybees Model
# (owns time) and routes decisions through the native DYNAMO-M DecisionModule
# via DynamoLiveRule, feeding a deterministic node population from the FloodAdapt
# lookup table through the PRE.4 adapter. Guarded so the package imports even
# when honeybees is absent (HONEYBEES_AVAILABLE is False; construction raises).
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
    "DynamoLiveRule",
    "DynamoMNotAvailable",
    "DYNAMO_M_AVAILABLE",
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

