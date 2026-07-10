"""
Internal data-plumbing layer for the FloodAdapt-ABM coupling.

This package (``_core``) is not part of the public API. Users should call
``SimulationEngine`` instead. The bridge and lookup utilities are composed
internally for data loading, interpolation, and per-agent state management.
"""
from floodadapt_abm._core.dynamo_decision_bridge import (
    DynamoDecisionBridge,
    _calc_eu_adapt,
    _calc_eu_do_nothing,
    _iterate_through_flood,
)
from floodadapt_abm._core.lookup_utils import (
    interpolate_damage_at_slr,
    interpolate_damage_matrix,
)

__all__ = [
    "DynamoDecisionBridge",
    "_calc_eu_adapt",
    "_calc_eu_do_nothing",
    "_iterate_through_flood",
    "interpolate_damage_at_slr",
    "interpolate_damage_matrix",
]

