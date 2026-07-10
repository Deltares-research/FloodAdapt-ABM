"""
01_quickstart.py
================
The smallest possible end-to-end run of the unified ``SimulationEngine``.

What you learn here
-------------------
* how to load a lookup table and build a :class:`SimulationEngine`,
* that the engine uses the DYNAMO-M ``SEURule`` by default,
* how to drive time with an SLR trajectory (``engine.run``),
* the shape and meaning of the three result arrays.

Run::

    python 01_quickstart.py
"""
from __future__ import annotations

import numpy as np

import _shared  # bootstraps sys.path + provides a dataset
from floodadapt_abm import SimulationEngine, CouplingConfig


def main() -> None:
    _shared.banner("01 - QUICKSTART: one engine, one run")

    # 1. Data: real Charleston table if available, else a synthetic stand-in.
    ds, source = _shared.load_dataset()
    print(f"Dataset: {source}")

    # 2. Configuration (defaults are Charleston / settings.yml calibrated).
    cfg = CouplingConfig()

    # 3. Engine — with no decision_rule argument it defaults to SEURule
    #    (the validated DYNAMO-M Subjective-Expected-Utility science).
    engine = SimulationEngine(ds=ds, config=cfg)
    print(f"Residential agents: {engine.n_agents}")

    # 4. Time is driven by the length of the SLR trajectory (one value/year).
    #    Here: 30 years, sea level rising linearly 0 -> 1.5 m.
    slr_trajectory = np.linspace(0.0, 1.5, 30)

    # 5. Run 5 Monte-Carlo sequences (independent random weather histories).
    results = engine.run(slr_trajectory, no_seq=5, seed=42)

    # 6. Read the results.
    dmg = results["damage_history"]        # (no_seq, n_agents, n_years)
    adapted = results["adapted_history"]   # (no_seq, n_agents, n_years) bool
    adoption = results["adoption_fraction"]  # (no_seq, n_years)

    print("\nResult shapes (no_seq, n_agents, n_years):")
    print(f"  damage_history   : {dmg.shape}")
    print(f"  adapted_history  : {adapted.shape}")
    print(f"  adoption_fraction: {adoption.shape}  (no_seq, n_years)")

    print("\nHeadline numbers (averaged over sequences):")
    print(f"  Final adoption fraction: {adoption[:, -1].mean():.1%}")
    print(f"  Mean total damage/seq  : ${dmg.sum(axis=(1, 2)).mean():,.0f}")

    print("\nDone. Next: 02_rules_comparison.py")


if __name__ == "__main__":
    main()
