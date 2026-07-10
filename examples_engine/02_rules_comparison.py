"""
02_rules_comparison.py
======================
The Strategy Pattern in action: swap the *behaviour* without touching the
engine.  Runs the same scenario twice — once with the DYNAMO-M ``SEURule`` and
once with the legacy ``ThresholdRule`` — and compares the outcomes.

What you learn here
-------------------
* the engine (time + data) is fixed; only the ``decision_rule`` changes,
* ``SEURule`` is ex-ante (forward-looking utility), ``ThresholdRule`` is
  ex-post (reactive to realised damage),
* how to pull expected-utility diagnostics with ``track_eu=True``.

Run::

    python 02_rules_comparison.py
"""
from __future__ import annotations

import numpy as np

import _shared
from floodadapt_abm import (
    SimulationEngine,
    CouplingConfig,
    SEURule,
    ThresholdRule,
)

SLR = np.linspace(0.0, 1.5, 30)
NO_SEQ = 5
SEED = 42


def _run(ds, cfg, rule=None, track_eu=False):
    engine = SimulationEngine(ds=ds, config=cfg, decision_rule=rule)
    return engine, engine.run(SLR, no_seq=NO_SEQ, seed=SEED, track_eu=track_eu)


def main() -> None:
    _shared.banner("02 - RULES COMPARISON: SEURule vs ThresholdRule")
    ds, source = _shared.load_dataset()
    print(f"Dataset: {source}")
    cfg = CouplingConfig()

    # Scenario 1 - SEURule (default rule; ask for EU diagnostics).
    _shared.banner("Scenario 1: SEURule (DYNAMO-M SEU science)")
    eng_seu, res_seu = _run(ds, cfg, rule=None, track_eu=True)
    print(f"Agents: {eng_seu.n_agents}")
    print(f"Final adoption : {res_seu['adoption_fraction'][:, -1].mean():.1%}")
    print(f"Total damage   : ${res_seu['damage_history'].sum():,.0f}")
    print(f"Mean EU(adapt) : {np.nanmean(res_seu['eu_adapt_history']):,.0f}")
    print(f"Mean EU(stay)  : {np.nanmean(res_seu['eu_do_nothing_history']):,.0f}")

    # Scenario 2 - ThresholdRule (legacy heuristic, same engine + config).
    _shared.banner("Scenario 2: ThresholdRule (legacy 0.3 heuristic)")
    thresh = ThresholdRule(cfg.decision, damage_threshold=0.30)
    eng_thr, res_thr = _run(ds, cfg, rule=thresh)
    print(f"Final adoption : {res_thr['adoption_fraction'][:, -1].mean():.1%}")
    print(f"Total damage   : ${res_thr['damage_history'].sum():,.0f}")

    # Side by side.
    _shared.banner("Comparison")
    print(f"{'metric':<28}{'SEURule':>16}{'ThresholdRule':>16}")
    print(f"{'final adoption':<28}"
          f"{res_seu['adoption_fraction'][:, -1].mean():>15.1%} "
          f"{res_thr['adoption_fraction'][:, -1].mean():>15.1%}")
    print(f"{'total damage ($)':<28}"
          f"{res_seu['damage_history'].sum():>16,.0f}"
          f"{res_thr['damage_history'].sum():>16,.0f}")
    print(
        "\nSEURule adapts before damage is realised (utility maximisation);\n"
        "ThresholdRule only reacts once a big loss occurs. Same engine, same\n"
        "data, different rule => different dynamics."
    )
    print("\nDone. Next: 03_custom_rule.py")


if __name__ == "__main__":
    main()
