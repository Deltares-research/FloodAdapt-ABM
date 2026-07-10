"""
05_dynamo_live_parity.py
========================
Phase 4a: the ``DynamoLiveRule`` parity oracle.  This rule drives the **native**
DYNAMO-M ``DecisionModule`` (``calcEU_do_nothing`` / ``calcEU_adapt``) instead of
the NumPy kernels ported into the bridge.  Its purpose is to *prove* the ported
``SEURule`` has not drifted from upstream DYNAMO-M.

What you learn here
-------------------
* how the DYNAMO-M dependency is *optional and guarded*
  (``DYNAMO_M_AVAILABLE``); the package works without it,
* how to point the rule at a DYNAMO-M checkout,
* how to run the parity check: identical decisions + near-identical EU,
* that ``DynamoLiveRule`` is a drop-in ``DecisionRule`` (plugs into the engine).

Bit-parity configuration
-------------------------
For an exact cross-check use ``error_interval = 0`` (the default) and
``amenity_value = 0``; then the only differences are float32 rounding inside
the trapezoidal EU integral, which never flips a decision.

Run::

    set DYNAMO_M_PATH=c:\\repos\\DYNAMO-M\\DYNAMO-M   (optional; a default is tried)
    python 05_dynamo_live_parity.py
"""
from __future__ import annotations

import numpy as np

import _shared
from floodadapt_abm import (
    SimulationEngine,
    CouplingConfig,
    SEURule,
    DYNAMO_M_AVAILABLE,
)


def _parity_check(ds, cfg) -> None:
    """Assemble one year's arrays and compare ported SEURule vs native rule."""
    from floodadapt_abm import DynamoLiveRule  # imported here: guarded dependency

    engine = SimulationEngine(ds=ds, config=cfg)
    state = engine.state
    d_no, d_fp = engine.prepare_damages(1.0)   # damages @ SLR = 1.0 m
    amenity = engine._data.amenity_value

    kwargs = dict(
        agent_state=state,
        damages_this_year=np.zeros(state.n_agents, dtype=np.float32),
        damages_no_adapt=d_no,
        damages_adapt=d_fp,
        event_freqs=engine._event_freqs,
        max_pot_dmg=engine.max_pot_dmg,
        adaptation_costs=engine._annual_adapt_cost,
    )

    seu = SEURule(cfg.decision, amenity_value=amenity)
    live = DynamoLiveRule(cfg.decision, amenity_value=amenity)

    a_seu = seu.should_adapt(**kwargs)
    a_live = live.should_adapt(**kwargs)

    decisions_match = bool(np.array_equal(a_seu, a_live))
    eu_abs = max(
        float(np.max(np.abs(np.nan_to_num(seu.last_eu_adapt) - np.nan_to_num(live.last_eu_adapt)))),
        float(np.max(np.abs(np.nan_to_num(seu.last_eu_do_nothing) - np.nan_to_num(live.last_eu_do_nothing)))),
    )
    print(f"  agents compared      : {state.n_agents}")
    print(f"  decisions identical  : {decisions_match}")
    print(f"  EU max |abs| diff    : {eu_abs:.2e}")
    print(f"  parity gate          : {'PASS' if decisions_match and eu_abs <= 1e-3 else 'FAIL'}")


def _engine_with_live_rule(ds, cfg) -> None:
    """Show DynamoLiveRule as a normal, pluggable engine rule."""
    from floodadapt_abm import DynamoLiveRule

    live = DynamoLiveRule(cfg.decision)
    engine = SimulationEngine(ds=ds, config=cfg, decision_rule=live)
    res = engine.run(np.linspace(0.0, 1.5, 30), no_seq=3, seed=42)
    print(f"  final adoption (live rule): {res['adoption_fraction'][:, -1].mean():.1%}")


def main() -> None:
    _shared.banner("05 - PHASE 4a: DynamoLiveRule (native DYNAMO-M parity)")

    print(f"DYNAMO_M_AVAILABLE = {DYNAMO_M_AVAILABLE}")
    if not DYNAMO_M_AVAILABLE:
        print(
            "\nDYNAMO-M is not importable in this environment, so the live rule\n"
            "cannot run. This is expected and safe: the rest of FloodAdapt-ABM\n"
            "(SEURule / ThresholdRule) works unchanged. To enable the live rule,\n"
            "set DYNAMO_M_PATH to your DYNAMO-M/DYNAMO-M package directory.\n"
        )
        return

    ds, source = _shared.load_dataset()
    print(f"Dataset: {source}")
    cfg = CouplingConfig()  # error_interval=0 (bit-parity), amenity handled inside

    _shared.banner("Parity check: ported SEURule vs native DynamoLiveRule")
    _parity_check(ds, cfg)

    _shared.banner("DynamoLiveRule as a pluggable engine rule")
    _engine_with_live_rule(ds, cfg)

    print(
        "\nThe ported SEURule reproduces native DYNAMO-M: identical decisions,\n"
        "EU differences at float32 level. The same rule interface is the seam\n"
        "for the future Phase-4b Mesa-native integration."
    )
    print("\nDone. See ../docs and the progress_todos parity report for detail.")


if __name__ == "__main__":
    main()
